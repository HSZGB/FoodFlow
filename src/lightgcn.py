from __future__ import annotations

import torch
from torch import nn


class LightGCN(nn.Module):
    """Minimal LightGCN baseline for user-POI bipartite graphs."""

    def __init__(self, n_users: int, n_items: int, edge_index: torch.Tensor, embed_dim: int = 64, n_layers: int = 2) -> None:
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_nodes = n_users + n_items
        self.n_layers = n_layers
        self.edge_index = edge_index.long()
        self.embedding = nn.Embedding(self.n_nodes, embed_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)

        deg = torch.zeros(self.n_nodes)
        src, dst = self.edge_index
        deg.index_add_(0, src, torch.ones_like(src, dtype=torch.float32))
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
        self.register_buffer("deg", deg.clamp_min(1.0))

    def propagate_once(self, x: torch.Tensor) -> torch.Tensor:
        src, dst = self.edge_index
        norm = (self.deg[src].rsqrt() * self.deg[dst].rsqrt()).unsqueeze(-1)
        out = torch.zeros_like(x)
        out.index_add_(0, dst, x[src] * norm)
        out.index_add_(0, src, x[dst] * norm)
        return out

    def all_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding.weight
        layers = [x]
        for _ in range(self.n_layers):
            x = self.propagate_once(x)
            layers.append(x)
        out = torch.stack(layers, dim=0).mean(dim=0)
        return out[: self.n_users], out[self.n_users :]

    def score(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        user_emb, item_emb = self.all_embeddings()
        return (user_emb[users] * item_emb[items]).sum(-1)
