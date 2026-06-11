from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import PreparedData
from .rerank import estimate_user_merchant_eta, fairness_scores, minmax, supply_score_for_merchant


@dataclass
class RecommendationResult:
    name: str
    recommendations: dict[str, list[str]]
    scores: dict[str, dict[str, float]]


class BaseRecommender:
    name = "Base"

    def fit(self, data: PreparedData) -> "BaseRecommender":
        self.data = data
        self.history = data.history_by_user()
        train_counts = data.orders_train["wm_poi_id"].astype(str).value_counts()
        known_merchants = set(data.merchant_ids)
        active = [m for m in train_counts.index.astype(str).tolist() if m in known_merchants]
        self.active_ids = active or data.merchant_ids
        self.active_set = set(self.active_ids)
        self.merchant_ids = active[:400] or data.merchant_ids[:400]
        self.merchant_set = set(self.merchant_ids)
        self.popular = train_counts.reindex(self.merchant_ids).fillna(0)
        self.popular_list = self.popular.sort_values(ascending=False).index.astype(str).tolist()
        return self

    def recommend_for_user(self, user_id: str, k: int, period: str = "lunch") -> tuple[list[str], dict[str, float]]:
        raise NotImplementedError

    def recommend(self, users: list[str], k: int, periods: dict[str, str] | None = None) -> RecommendationResult:
        recs: dict[str, list[str]] = {}
        scores: dict[str, dict[str, float]] = {}
        periods = periods or {}
        for user_id in users:
            items, item_scores = self.recommend_for_user(user_id, k, periods.get(user_id, "lunch"))
            recs[user_id] = items[:k]
            scores[user_id] = item_scores
        return RecommendationResult(self.name, recs, scores)

    def _remove_seen_and_backfill(self, user_id: str, ranked: list[str], k: int) -> list[str]:
        # Food delivery has strong repeat consumption, so historical merchants remain valid candidates.
        out = []
        for item in ranked + self.popular_list:
            if item not in out:
                out.append(item)
            if len(out) >= k:
                break
        return out

    def _personalized_candidates(self, user_id: str) -> list[str]:
        history = [m for m in self.history.get(user_id, []) if m in self.active_set]
        return list(dict.fromkeys(history + self.merchant_ids))


class RandomRecommender(BaseRecommender):
    name = "Random"

    def __init__(self, seed: int = 42):
        self.seed = seed

    def recommend_for_user(self, user_id: str, k: int, period: str = "lunch") -> tuple[list[str], dict[str, float]]:
        digest = hashlib.md5(f"{self.seed}:{user_id}".encode("utf-8")).hexdigest()
        rng = np.random.default_rng(int(digest[:8], 16))
        candidates = self.merchant_ids.copy()
        rng.shuffle(candidates)
        recs = self._remove_seen_and_backfill(user_id, candidates, k)
        return recs, {m: float(k - i) for i, m in enumerate(recs)}


class PopularRecommender(BaseRecommender):
    name = "Popular"

    def recommend_for_user(self, user_id: str, k: int, period: str = "lunch") -> tuple[list[str], dict[str, float]]:
        recs = self._remove_seen_and_backfill(user_id, self.popular_list, k)
        return recs, {m: float(self.popular.get(m, 0)) for m in recs}


class RepeatRecommender(BaseRecommender):
    name = "Repeat"

    def recommend_for_user(self, user_id: str, k: int, period: str = "lunch") -> tuple[list[str], dict[str, float]]:
        counts = Counter(self.history.get(user_id, []))
        ranked = [m for m, _ in counts.most_common()]
        # Repeat is intentionally allowed to include seen merchants because food delivery has strong reorder behavior.
        out = []
        for item in ranked + self.popular_list:
            if item not in out:
                out.append(item)
            if len(out) >= k:
                break
        return out, {m: float(counts.get(m, self.popular.get(m, 0) * 0.01)) for m in out}


class ItemCFRecommender(BaseRecommender):
    name = "ItemCF"

    def fit(self, data: PreparedData) -> "ItemCFRecommender":
        super().fit(data)
        train = data.orders_train[data.orders_train["wm_poi_id"].astype(str).isin(self.active_set)].copy()
        user_items = train.groupby("user_id")["wm_poi_id"].apply(lambda s: list(dict.fromkeys(s.astype(str))))
        item_counts = Counter(train["wm_poi_id"].astype(str))
        co_counts: Counter[tuple[str, str]] = Counter()
        for items in user_items:
            # Very heavy users add little signal and can dominate runtime, so cap each basket.
            capped = items[:80]
            for i, item_i in enumerate(capped):
                for item_j in capped[i + 1 :]:
                    if item_i == item_j:
                        continue
                    a, b = sorted((item_i, item_j))
                    co_counts[(a, b)] += 1
        temp: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for (item_i, item_j), count in co_counts.items():
            sim = count / np.sqrt(item_counts[item_i] * item_counts[item_j])
            temp[item_i].append((item_j, float(sim)))
            temp[item_j].append((item_i, float(sim)))
        self.similar = {
            item: dict(sorted(values, key=lambda x: x[1], reverse=True)[:120]) for item, values in temp.items()
        }
        return self

    def recommend_for_user(self, user_id: str, k: int, period: str = "lunch") -> tuple[list[str], dict[str, float]]:
        scores = Counter()
        seen = self.history.get(user_id, [])
        for item in seen:
            for other, sim in self.similar.get(item, {}).items():
                scores[other] += sim
        ranked = [m for m, _ in scores.most_common()] + self.popular_list
        recs = self._remove_seen_and_backfill(user_id, ranked, k)
        return recs, {m: float(scores.get(m, self.popular.get(m, 0) * 0.001)) for m in recs}


class BPRMFRecommender(BaseRecommender):
    name = "BPR-MF"

    def __init__(self, factors: int = 24, epochs: int = 4, lr: float = 0.035, reg: float = 0.002, seed: int = 42):
        self.factors = factors
        self.epochs = epochs
        self.lr = lr
        self.reg = reg
        self.seed = seed

    def fit(self, data: PreparedData) -> "BPRMFRecommender":
        super().fit(data)
        train = data.orders_train[data.orders_train["wm_poi_id"].astype(str).isin(self.active_set)].copy()
        users = sorted(train["user_id"].astype(str).unique())
        items = sorted(train["wm_poi_id"].astype(str).unique())
        self.user_index = {u: i for i, u in enumerate(users)}
        self.item_index = {m: i for i, m in enumerate(items)}
        self.index_item = {i: m for m, i in self.item_index.items()}
        rng = np.random.default_rng(self.seed)
        self.user_factors = rng.normal(0, 0.08, (len(users), self.factors))
        self.item_factors = rng.normal(0, 0.08, (len(items), self.factors))
        positives = [(self.user_index[u], self.item_index[m]) for u, m in train[["user_id", "wm_poi_id"]].astype(str).itertuples(index=False)]
        user_pos = defaultdict(set)
        for u, i in positives:
            user_pos[u].add(i)
        if not positives:
            return self
        item_count = len(items)
        for _ in range(self.epochs):
            rng.shuffle(positives)
            for u, i in positives:
                j = int(rng.integers(0, item_count))
                tries = 0
                while j in user_pos[u] and tries < 20:
                    j = int(rng.integers(0, item_count))
                    tries += 1
                x = float(np.dot(self.user_factors[u], self.item_factors[i] - self.item_factors[j]))
                grad = 1.0 / (1.0 + np.exp(x))
                u_vec = self.user_factors[u].copy()
                i_vec = self.item_factors[i].copy()
                j_vec = self.item_factors[j].copy()
                self.user_factors[u] += self.lr * (grad * (i_vec - j_vec) - self.reg * u_vec)
                self.item_factors[i] += self.lr * (grad * u_vec - self.reg * i_vec)
                self.item_factors[j] += self.lr * (-grad * u_vec - self.reg * j_vec)
        return self

    def recommend_for_user(self, user_id: str, k: int, period: str = "lunch") -> tuple[list[str], dict[str, float]]:
        if user_id not in self.user_index:
            recs = self._remove_seen_and_backfill(user_id, self.popular_list, k)
            return recs, {m: float(self.popular.get(m, 0)) for m in recs}
        u = self.user_index[user_id]
        scores_arr = self.item_factors @ self.user_factors[u]
        ranked = [self.index_item[i] for i in np.argsort(-scores_arr)]
        recs = self._remove_seen_and_backfill(user_id, ranked, k)
        return recs, {m: float(scores_arr[self.item_index[m]]) if m in self.item_index else 0.0 for m in recs}


class UserOnlyRecommender(BaseRecommender):
    name = "UserOnly"

    def fit(self, data: PreparedData) -> "UserOnlyRecommender":
        super().fit(data)
        self.merchants = data.merchants.set_index("wm_poi_id", drop=False)
        self.users = data.users.set_index("user_id", drop=False)
        merged = data.orders_train.merge(data.merchants[["wm_poi_id", "primary_first_tag_id"]], on="wm_poi_id", how="left")
        cat_counts = merged.groupby(["user_id", "primary_first_tag_id"]).size().reset_index(name="cnt")
        cat_counts["share"] = cat_counts["cnt"] / cat_counts.groupby("user_id")["cnt"].transform("sum")
        self.user_cat_counts = {
            str(user_id): dict(zip(group["primary_first_tag_id"].astype(str), group["share"].astype(float)))
            for user_id, group in cat_counts.groupby("user_id", sort=False)
        }
        self.quality = dict(zip(data.merchants["wm_poi_id"].astype(str), minmax(data.merchants["poi_score"]).astype(float)))
        self.pop_norm = dict(zip(data.merchants["wm_poi_id"].astype(str), minmax(data.merchants["order_count"]).astype(float)))
        if "ord_period_name" in data.orders_train.columns:
            period_counts = data.orders_train.groupby(["ord_period_name", "wm_poi_id"]).size().reset_index(name="cnt")
            period_counts["score"] = period_counts["cnt"] / period_counts.groupby("ord_period_name")["cnt"].transform("max")
            self.period_pop = {
                str(period): dict(zip(group["wm_poi_id"].astype(str), group["score"].astype(float)))
                for period, group in period_counts.groupby("ord_period_name", sort=False)
            }
        else:
            self.period_pop = {}
        return self

    def user_score(self, user_id: str, merchant_id: str, period: str = "lunch") -> float:
        merchant = self.merchants.loc[merchant_id]
        user = self.users.loc[user_id] if user_id in self.users.index else pd.Series(dtype=object)
        category = merchant.get("primary_first_tag_id", "unknown")
        category_pref = float(self.user_cat_counts.get(user_id, {}).get(category, 0.0))
        repeat = np.log1p(Counter(self.history.get(user_id, [])).get(merchant_id, 0)) / np.log(5)
        user_price = float(user.get("avg_order_price", user.get("avg_pay_amt", 35)) or 35)
        merchant_price = float(merchant.get("avg_order_price", 35) or 35)
        price_fit = 1.0 - min(abs(user_price - merchant_price) / max(user_price, merchant_price, 1.0), 1.0)
        period_fit = float(self.period_pop.get(period, {}).get(merchant_id, 0.0))
        novelty = 1.0 - float(self.pop_norm.get(merchant_id, 0.0))
        quality = float(self.quality.get(merchant_id, 0.5))
        return float(0.20 * category_pref + 0.52 * repeat + 0.10 * price_fit + 0.06 * period_fit + 0.07 * quality + 0.05 * novelty)

    def recommend_for_user(self, user_id: str, k: int, period: str = "lunch") -> tuple[list[str], dict[str, float]]:
        candidates = self._personalized_candidates(user_id)
        scores = {m: self.user_score(user_id, m, period) for m in candidates}
        ranked = sorted(scores, key=scores.get, reverse=True)
        recs = self._remove_seen_and_backfill(user_id, ranked, k)
        return recs, {m: scores[m] for m in recs}


class OursFullRecommender(UserOnlyRecommender):
    name = "Ours-Full"

    def fit(self, data: PreparedData) -> "OursFullRecommender":
        super().fit(data)
        self.fair = fairness_scores(data.merchants)
        self.data = data
        return self

    def recommend_for_user(self, user_id: str, k: int, period: str = "lunch") -> tuple[list[str], dict[str, float]]:
        user_row = self.users.loc[user_id] if user_id in self.users.index else pd.Series(dtype=object)
        scores = {}
        for m in self._personalized_candidates(user_id):
            merchant = self.merchants.loc[m]
            user_score = self.user_score(user_id, m, period)
            eta = estimate_user_merchant_eta(user_row, merchant, period)
            eta_score = 1.0 - min(eta / 70.0, 1.0)
            supply = supply_score_for_merchant(merchant)
            scores[m] = float(0.62 * user_score + 0.18 * self.fair.get(m, 0.0) + 0.14 * eta_score + 0.06 * supply)
        ranked = sorted(scores, key=scores.get, reverse=True)
        recs = self._remove_seen_and_backfill(user_id, ranked, k)
        return recs, {m: scores[m] for m in recs}


def build_recommenders(seed: int = 42) -> list[BaseRecommender]:
    return [
        RandomRecommender(seed=seed),
        PopularRecommender(),
        RepeatRecommender(),
        ItemCFRecommender(),
        BPRMFRecommender(seed=seed),
        UserOnlyRecommender(),
        OursFullRecommender(),
    ]
