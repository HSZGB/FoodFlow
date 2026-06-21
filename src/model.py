from __future__ import annotations

import math

import torch
from torch import nn


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    masked = logits.masked_fill(~mask, -1e9)
    probs = torch.softmax(masked, dim=dim) * mask.float()
    return probs / probs.sum(dim=dim, keepdim=True).clamp_min(1e-8)


class DynamicKGAttentionRecommender(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_pois: int,
        n_entities: int,
        n_relations: int,
        basic_dim: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        use_time_bias: bool = True,
        use_relation_attention: bool = True,
        use_basic_features: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.use_time_bias = use_time_bias
        self.use_relation_attention = use_relation_attention
        self.use_basic_features = use_basic_features

        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.poi_emb = nn.Embedding(n_pois, embed_dim)
        self.entity_emb = nn.Embedding(n_entities, embed_dim, padding_idx=0)
        self.rel_emb = nn.Embedding(n_relations, embed_dim, padding_idx=0)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.e_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.r_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_vec = nn.Linear(embed_dim, 1, bias=False)

        self.poi_q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.poi_e_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.poi_r_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.poi_attn_vec = nn.Linear(embed_dim, 1, bias=False)

        self.user_norm = nn.LayerNorm(embed_dim)
        self.poi_norm = nn.LayerNorm(embed_dim)

        rank_basic_dim = basic_dim if use_basic_features else 0
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 3 + rank_basic_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for emb in [self.user_emb, self.poi_emb, self.entity_emb, self.rel_emb]:
            nn.init.xavier_uniform_(emb.weight)
            if emb.padding_idx is not None:
                with torch.no_grad():
                    emb.weight[emb.padding_idx].zero_()

    def encode_dynamic_user(
        self,
        user: torch.Tensor,
        interest_e: torch.Tensor,
        interest_r: torch.Tensor,
        interest_w: torch.Tensor,
        interest_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.user_emb(user)
        e = self.entity_emb(interest_e)
        r = self.rel_emb(interest_r)
        if self.use_relation_attention:
            logits = self.attn_vec(
                torch.tanh(self.q_proj(q).unsqueeze(1) + self.e_proj(e) + self.r_proj(r))
            ).squeeze(-1)
        else:
            logits = (self.q_proj(q).unsqueeze(1) * self.e_proj(e)).sum(-1) / math.sqrt(self.embed_dim)
        if self.use_time_bias:
            logits = logits + torch.log1p(interest_w.clamp_min(0.0))
        alpha = masked_softmax(logits, interest_mask, dim=1)
        interest = torch.sum(alpha.unsqueeze(-1) * (e + r), dim=1)
        h = self.user_norm(q + interest)
        return h, alpha

    def encode_poi(self, poi: torch.Tensor, attr_e: torch.Tensor, attr_r: torch.Tensor, attr_mask: torch.Tensor) -> torch.Tensor:
        base = self.poi_emb(poi)
        e = self.entity_emb(attr_e)
        r = self.rel_emb(attr_r)
        logits = self.poi_attn_vec(
            torch.tanh(self.poi_q_proj(base).unsqueeze(1) + self.poi_e_proj(e) + self.poi_r_proj(r))
        ).squeeze(-1)
        alpha = masked_softmax(logits, attr_mask, dim=1)
        attr = torch.sum(alpha.unsqueeze(-1) * (e + r), dim=1)
        return self.poi_norm(base + attr)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        h_u, alpha = self.encode_dynamic_user(
            batch["user"],
            batch["interest_e"],
            batch["interest_r"],
            batch["interest_w"],
            batch["interest_mask"],
        )
        h_v = self.encode_poi(batch["poi"], batch["attr_e"], batch["attr_r"], batch["attr_mask"])
        parts = [h_u, h_v, h_u * h_v]
        if self.use_basic_features:
            parts.append(batch["basic"])
        logits = self.mlp(torch.cat(parts, dim=-1)).squeeze(-1)
        return logits, {"interest_alpha": alpha}


class MatrixFactorization(nn.Module):
    def __init__(self, n_users: int, n_pois: int, embed_dim: int = 64) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.poi_emb = nn.Embedding(n_pois, embed_dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.poi_bias = nn.Embedding(n_pois, 1)
        self.global_bias = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.user_emb.weight, std=0.02)
        nn.init.normal_(self.poi_emb.weight, std=0.02)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.poi_bias.weight)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        u = self.user_emb(batch["user"])
        v = self.poi_emb(batch["poi"])
        score = (u * v).sum(-1)
        score = score + self.user_bias(batch["user"]).squeeze(-1) + self.poi_bias(batch["poi"]).squeeze(-1) + self.global_bias
        return score, {}
