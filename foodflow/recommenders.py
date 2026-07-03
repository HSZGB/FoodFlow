from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import PreparedData
from .rerank import estimate_user_merchant_eta, fairness_scores, minmax, supply_score_for_merchant


SEQ_TUNED_WEIGHTS = {
    "fast_recency": 0.142601,
    "slow_recency": 0.093624,
    "repeat": 0.412158,
    "transition": 0.247023,
    "category": 0.091250,
    "popularity": 0.008945,
    "quality": 0.004398,
}

SEQ_TUNED_XQUAD_WEIGHTS = {
    "fast_recency": 0.270998,
    "slow_recency": 0.106017,
    "repeat": 0.279395,
    "transition": 0.274380,
    "category": 0.048099,
    "popularity": 0.005218,
    "quality": 0.015892,
}

SEQUENCE_FEATURE_NAMES = [
    "fast_recency",
    "slow_recency",
    "repeat",
    "transition",
    "category",
    "popularity",
    "quality",
]

TRIPARTITE_COMPONENT_NAMES = [
    "user_score",
    "merchant_fairness",
    "eta_score",
    "supply_score",
]


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


class SequentialHybridRecommender(UserOnlyRecommender):
    name = "Seq-Hybrid"

    def fit(self, data: PreparedData) -> "SequentialHybridRecommender":
        super().fit(data)
        train = data.orders_train.copy()
        train["user_id"] = train["user_id"].astype(str)
        train["wm_poi_id"] = train["wm_poi_id"].astype(str)
        if "order_timestamp" in train.columns:
            train["order_timestamp"] = pd.to_numeric(train["order_timestamp"], errors="coerce").fillna(0)
            train = train.sort_values(["user_id", "order_timestamp"])
        else:
            train = train.sort_values(["user_id"])

        self.user_item_counts = {
            str(user_id): Counter(group["wm_poi_id"].astype(str))
            for user_id, group in train.groupby("user_id", sort=False)
        }
        self.recent_by_user: dict[str, list[str]] = {}
        transition_counts: dict[str, Counter[str]] = defaultdict(Counter)
        for user_id, group in train.groupby("user_id", sort=False):
            seq = [item for item in group["wm_poi_id"].astype(str).tolist() if item in self.active_set]
            if not seq:
                continue
            self.recent_by_user[str(user_id)] = seq
            for prev, nxt in zip(seq, seq[1:]):
                if prev != nxt:
                    transition_counts[prev][nxt] += 1

        self.transitions: dict[str, dict[str, float]] = {}
        for item, counts in transition_counts.items():
            total = float(sum(counts.values()))
            if total <= 0:
                continue
            self.transitions[item] = {
                other: count / total for other, count in counts.most_common(120)
            }
        max_pop = float(np.log1p(max(float(self.popular.max()), 1.0)))
        self.pop_log = {m: float(np.log1p(self.popular.get(m, 0)) / max_pop) for m in self.active_ids}
        return self

    def _recency_scores(self, seq: list[str], merchant_id: str) -> tuple[float, float]:
        for age, item in enumerate(reversed(seq)):
            if item == merchant_id:
                return float(np.exp(-age / 6.0)), float(np.exp(-age / 12.0))
        return 0.0, 0.0

    def _transition_score(self, seq: list[str], merchant_id: str) -> float:
        score = 0.0
        for offset, item in enumerate(reversed(seq[-5:]), start=1):
            score = max(score, (0.85 ** offset) * self.transitions.get(item, {}).get(merchant_id, 0.0))
        return float(score)

    def _sequential_candidates(self, user_id: str) -> list[str]:
        seq = self.recent_by_user.get(user_id, [])
        candidates: list[str] = []
        candidates.extend(list(dict.fromkeys(reversed(seq[-40:]))))
        for item in reversed(seq[-8:]):
            candidates.extend(list(self.transitions.get(item, {}).keys())[:80])
        candidates.extend(self.popular_list)
        return [m for m in dict.fromkeys(candidates) if m in self.merchants.index]

    def user_score(self, user_id: str, merchant_id: str, period: str = "lunch") -> float:
        seq = self.recent_by_user.get(user_id, [])
        counts = self.user_item_counts.get(user_id, Counter())
        merchant = self.merchants.loc[merchant_id]
        category = merchant.get("primary_first_tag_id", "unknown")
        category_pref = float(self.user_cat_counts.get(user_id, {}).get(category, 0.0))
        repeat = np.log1p(counts.get(merchant_id, 0)) / np.log(5)
        recency_fast, recency_slow = self._recency_scores(seq, merchant_id)
        transition = self._transition_score(seq, merchant_id)
        popularity = float(self.pop_log.get(merchant_id, 0.0))
        quality = float(self.quality.get(merchant_id, 0.5))
        return float(
            0.25 * recency_fast
            + 0.12 * recency_slow
            + 0.30 * repeat
            + 0.23 * transition
            + 0.05 * category_pref
            + 0.03 * popularity
            + 0.02 * quality
        )

    def recommend_for_user(self, user_id: str, k: int, period: str = "lunch") -> tuple[list[str], dict[str, float]]:
        candidates = self._sequential_candidates(user_id)
        scores: dict[str, float] = {}
        for merchant_id in candidates:
            scores[merchant_id] = self.user_score(user_id, merchant_id, period)
        ranked = sorted(scores, key=scores.get, reverse=True)
        recs = self._remove_seen_and_backfill(user_id, ranked, k)
        return recs, {m: scores.get(m, float(self.popular.get(m, 0))) for m in recs}


def _normalize_tripartite_components(
    components_by_item: dict[str, dict[str, float]],
    weights: dict[str, float],
) -> dict[str, dict[str, float]]:
    if not components_by_item:
        return components_by_item
    frame = pd.DataFrame.from_dict(components_by_item, orient="index")
    for name in TRIPARTITE_COMPONENT_NAMES:
        values = pd.to_numeric(frame.get(name, 0.0), errors="coerce").fillna(0.0)
        lo = float(values.min())
        hi = float(values.max())
        if hi - lo <= 1e-12:
            normalized = pd.Series(np.zeros(len(values)), index=values.index)
        else:
            normalized = (values - lo) / (hi - lo)
        frame[f"{name}_norm"] = normalized.astype(float)
    weight_sum = max(float(sum(weights.values())), 1e-12)
    frame["raw_final_score"] = pd.to_numeric(frame.get("final_score", 0.0), errors="coerce").fillna(0.0)
    frame["final_score"] = sum(
        float(weight) * frame[f"{name}_norm"] for name, weight in weights.items()
    ) / weight_sum
    return {
        str(index): {key: float(value) for key, value in row.dropna().items()}
        for index, row in frame.iterrows()
    }


class WeightedSequentialRecommender(SequentialHybridRecommender):
    name = "Seq-Weighted"

    def __init__(self, name: str = "Seq-Weighted", weights: dict[str, float] | None = None):
        self.name = name
        self.seq_weights = weights or SEQ_TUNED_WEIGHTS

    def _sequence_feature_values(self, user_id: str, merchant_id: str) -> dict[str, float]:
        seq = self.recent_by_user.get(user_id, [])
        counts = self.user_item_counts.get(user_id, Counter())
        merchant = self.merchants.loc[merchant_id]
        category = merchant.get("primary_first_tag_id", "unknown")
        recency_fast, recency_slow = self._recency_scores(seq, merchant_id)
        return {
            "fast_recency": float(recency_fast),
            "slow_recency": float(recency_slow),
            "repeat": float(np.log1p(counts.get(merchant_id, 0)) / np.log(5)),
            "transition": float(self._transition_score(seq, merchant_id)),
            "category": float(self.user_cat_counts.get(user_id, {}).get(category, 0.0)),
            "popularity": float(self.pop_log.get(merchant_id, 0.0)),
            "quality": float(self.quality.get(merchant_id, 0.5)),
        }

    def user_score(self, user_id: str, merchant_id: str, period: str = "lunch") -> float:
        values = self._sequence_feature_values(user_id, merchant_id)
        weight_sum = max(float(sum(self.seq_weights.values())), 1e-12)
        return float(sum(self.seq_weights[key] * values.get(key, 0.0) for key in self.seq_weights) / weight_sum)


class SeqTunedRecommender(WeightedSequentialRecommender):
    def __init__(self):
        super().__init__(name="Seq-Tuned", weights=SEQ_TUNED_WEIGHTS)


class LightGBMRankerRecommender(WeightedSequentialRecommender):
    name = "LightGBM-LTR"

    def __init__(
        self,
        seed: int = 42,
        max_train_users: int = 1000,
        candidate_limit: int = 160,
        n_estimators: int = 80,
    ):
        super().__init__(name=self.name, weights=SEQ_TUNED_WEIGHTS)
        self.seed = seed
        self.max_train_users = max_train_users
        self.candidate_limit = candidate_limit
        self.n_estimators = n_estimators
        self.model = None

    def _feature_vector(self, user_id: str, merchant_id: str) -> list[float]:
        values = self._sequence_feature_values(user_id, merchant_id)
        return [float(values.get(name, 0.0)) for name in SEQUENCE_FEATURE_NAMES]

    def fit(self, data: PreparedData) -> "LightGBMRankerRecommender":
        super().fit(data)
        try:
            from lightgbm import LGBMRanker
        except ImportError:
            self.model = None
            return self

        users = [user_id for user_id, counts in self.user_item_counts.items() if counts]
        rng = np.random.default_rng(self.seed)
        if len(users) > self.max_train_users:
            users = rng.choice(users, size=self.max_train_users, replace=False).astype(str).tolist()

        features: list[list[float]] = []
        labels: list[int] = []
        groups: list[int] = []
        for user_id in users:
            positives = set(self.user_item_counts.get(user_id, Counter()))
            candidates = self._sequential_candidates(user_id)[: self.candidate_limit]
            if not candidates:
                continue
            user_features: list[list[float]] = []
            user_labels: list[int] = []
            for merchant_id in candidates:
                label = int(merchant_id in positives)
                user_features.append(self._feature_vector(user_id, merchant_id))
                user_labels.append(label)
            if not any(user_labels) or all(user_labels):
                continue
            features.extend(user_features)
            labels.extend(user_labels)
            groups.append(len(user_labels))

        if not features or not groups:
            self.model = None
            return self

        self.model = LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=self.n_estimators,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=20,
            subsample=0.85,
            colsample_bytree=0.9,
            random_state=self.seed,
            verbosity=-1,
        )
        self.model.fit(np.asarray(features, dtype=float), np.asarray(labels, dtype=int), group=groups)
        return self

    def recommend_for_user(self, user_id: str, k: int, period: str = "lunch") -> tuple[list[str], dict[str, float]]:
        if self.model is None:
            return super().recommend_for_user(user_id, k, period)
        candidates = self._sequential_candidates(user_id)
        if not candidates:
            return super().recommend_for_user(user_id, k, period)
        features = np.asarray([self._feature_vector(user_id, merchant_id) for merchant_id in candidates], dtype=float)
        predictions = self.model.predict(features)
        scores = {merchant_id: float(score) for merchant_id, score in zip(candidates, predictions)}
        ranked = sorted(scores, key=scores.get, reverse=True)
        recs = self._remove_seen_and_backfill(user_id, ranked, k)
        return recs, {m: scores.get(m, float(self.popular.get(m, 0))) for m in recs}


class LogisticLTRRecommender(WeightedSequentialRecommender):
    name = "Logistic-LTR"

    def __init__(
        self,
        seed: int = 42,
        max_train_users: int = 1000,
        candidate_limit: int = 160,
        max_iter: int = 250,
    ):
        super().__init__(name=self.name, weights=SEQ_TUNED_WEIGHTS)
        self.seed = seed
        self.max_train_users = max_train_users
        self.candidate_limit = candidate_limit
        self.max_iter = max_iter
        self.model = None
        self.backend = "logistic"
        self.feature_importances_: dict[str, float] = {}

    def _feature_vector(self, user_id: str, merchant_id: str) -> list[float]:
        values = self._sequence_feature_values(user_id, merchant_id)
        return [float(values.get(name, 0.0)) for name in SEQUENCE_FEATURE_NAMES]

    def fit(self, data: PreparedData) -> "LogisticLTRRecommender":
        super().fit(data)
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError:
            self.model = None
            self.backend = "seq_tuned_fallback"
            return self

        users = [user_id for user_id, counts in self.user_item_counts.items() if counts]
        rng = np.random.default_rng(self.seed)
        if len(users) > self.max_train_users:
            users = rng.choice(users, size=self.max_train_users, replace=False).astype(str).tolist()

        features: list[list[float]] = []
        labels: list[int] = []
        for user_id in users:
            positives = set(self.user_item_counts.get(user_id, Counter()))
            candidates = self._sequential_candidates(user_id)[: self.candidate_limit]
            if not candidates:
                continue
            user_labels = [int(merchant_id in positives) for merchant_id in candidates]
            if not any(user_labels) or all(user_labels):
                continue
            features.extend([self._feature_vector(user_id, merchant_id) for merchant_id in candidates])
            labels.extend(user_labels)

        if not features or len(set(labels)) < 2:
            self.model = None
            self.backend = "seq_tuned_fallback"
            return self

        self.model = LogisticRegression(
            class_weight="balanced",
            max_iter=self.max_iter,
            random_state=self.seed,
            solver="lbfgs",
        )
        self.model.fit(np.asarray(features, dtype=float), np.asarray(labels, dtype=int))
        coefs = np.abs(self.model.coef_[0])
        denom = float(coefs.sum()) or 1.0
        self.feature_importances_ = {
            name: float(value / denom) for name, value in zip(SEQUENCE_FEATURE_NAMES, coefs)
        }
        return self

    def recommend_for_user(self, user_id: str, k: int, period: str = "lunch") -> tuple[list[str], dict[str, float]]:
        if self.model is None:
            return super().recommend_for_user(user_id, k, period)
        candidates = self._sequential_candidates(user_id)
        if not candidates:
            return super().recommend_for_user(user_id, k, period)
        features = np.asarray([self._feature_vector(user_id, merchant_id) for merchant_id in candidates], dtype=float)
        if len(getattr(self.model, "classes_", [])) < 2:
            return super().recommend_for_user(user_id, k, period)
        positive_index = int(np.where(self.model.classes_ == 1)[0][0])
        predictions = self.model.predict_proba(features)[:, positive_index]
        scores = {merchant_id: float(score) for merchant_id, score in zip(candidates, predictions)}
        ranked = sorted(scores, key=scores.get, reverse=True)
        recs = self._remove_seen_and_backfill(user_id, ranked, k)
        return recs, {m: scores.get(m, float(self.popular.get(m, 0))) for m in recs}


class SeqTripartiteRecommender(SequentialHybridRecommender):
    name = "Seq-Tripartite"

    def __init__(
        self,
        user_weight: float = 0.95,
        fairness_weight: float = 0.02,
        eta_weight: float = 0.02,
        supply_weight: float = 0.01,
    ):
        self.user_weight = user_weight
        self.fairness_weight = fairness_weight
        self.eta_weight = eta_weight
        self.supply_weight = supply_weight

    def fit(self, data: PreparedData) -> "SeqTripartiteRecommender":
        super().fit(data)
        self.fair = fairness_scores(data.merchants)
        self.data = data
        return self

    def component_scores(self, user_id: str, merchant_id: str, period: str = "lunch") -> dict[str, float]:
        user_row = self.users.loc[user_id] if user_id in self.users.index else pd.Series(dtype=object)
        merchant = self.merchants.loc[merchant_id]
        user_score = self.user_score(user_id, merchant_id, period)
        eta = estimate_user_merchant_eta(user_row, merchant, period)
        eta_score = 1.0 - min(eta / 70.0, 1.0)
        supply = supply_score_for_merchant(merchant)
        fair = float(self.fair.get(merchant_id, 0.0))
        final = float(
            self.user_weight * user_score
            + self.fairness_weight * fair
            + self.eta_weight * eta_score
            + self.supply_weight * supply
        )
        return {
            "user_score": float(user_score),
            "merchant_fairness": fair,
            "eta_minutes": float(eta),
            "eta_score": float(eta_score),
            "supply_score": float(supply),
            "final_score": final,
        }

    def _component_scores_for_candidates(
        self,
        user_id: str,
        candidates: list[str],
        period: str = "lunch",
    ) -> dict[str, dict[str, float]]:
        components = {merchant_id: self.component_scores(user_id, merchant_id, period) for merchant_id in candidates}
        weights = {
            "user_score": self.user_weight,
            "merchant_fairness": self.fairness_weight,
            "eta_score": self.eta_weight,
            "supply_score": self.supply_weight,
        }
        return _normalize_tripartite_components(components, weights)

    def recommend_for_user(self, user_id: str, k: int, period: str = "lunch") -> tuple[list[str], dict[str, float]]:
        components = self._component_scores_for_candidates(user_id, self._sequential_candidates(user_id), period)
        scores = {merchant_id: values["final_score"] for merchant_id, values in components.items()}
        ranked = sorted(scores, key=scores.get, reverse=True)
        recs = self._remove_seen_and_backfill(user_id, ranked, k)
        return recs, {m: scores[m] for m in recs}


class SeqXQuadRecommender(SequentialHybridRecommender):
    name = "Seq-xQuAD"

    def __init__(self, diversity_weight: float = 0.22, tail_weight: float = 0.08):
        self.diversity_weight = diversity_weight
        self.tail_weight = tail_weight

    def _rerank_xquad(self, scores: dict[str, float], k: int) -> list[str]:
        if not scores:
            return []
        candidates = sorted(scores, key=scores.get, reverse=True)[:80]
        min_score = min(scores[m] for m in candidates)
        max_score = max(scores[m] for m in candidates)
        scale = max(max_score - min_score, 1e-9)
        relevance_weight = max(1.0 - self.diversity_weight - self.tail_weight, 0.0)
        selected: list[str] = []
        covered_categories: set[str] = set()
        while len(selected) < k and len(selected) < len(candidates):
            best_item = None
            best_score = -float("inf")
            for merchant_id in candidates:
                if merchant_id in selected:
                    continue
                relevance = (scores[merchant_id] - min_score) / scale
                merchant = self.merchants.loc[merchant_id]
                category = str(merchant.get("primary_first_tag_id", "unknown"))
                category_gain = 0.0 if category in covered_categories else 1.0
                order_count = float(merchant.get("order_count", 0) or 0)
                tail_gain = 1.0 / (1.0 + np.log1p(order_count))
                value = (
                    relevance_weight * relevance
                    + self.diversity_weight * category_gain
                    + self.tail_weight * tail_gain
                )
                if value > best_score:
                    best_item = merchant_id
                    best_score = value
            if best_item is None:
                break
            selected.append(best_item)
            covered_categories.add(str(self.merchants.loc[best_item].get("primary_first_tag_id", "unknown")))
        return selected

    def recommend_for_user(self, user_id: str, k: int, period: str = "lunch") -> tuple[list[str], dict[str, float]]:
        candidates = self._sequential_candidates(user_id)
        scores = {merchant_id: self.user_score(user_id, merchant_id, period) for merchant_id in candidates}
        recs = self._rerank_xquad(scores, k)
        recs = self._remove_seen_and_backfill(user_id, recs, k)
        return recs, {m: scores.get(m, float(self.popular.get(m, 0))) for m in recs}


class SeqTunedXQuadRecommender(SeqXQuadRecommender):
    name = "Seq-Tuned-xQuAD"

    def __init__(self, diversity_weight: float = 0.22, tail_weight: float = 0.08):
        super().__init__(diversity_weight=diversity_weight, tail_weight=tail_weight)
        self.seq_weights = SEQ_TUNED_XQUAD_WEIGHTS

    def user_score(self, user_id: str, merchant_id: str, period: str = "lunch") -> float:
        seq = self.recent_by_user.get(user_id, [])
        counts = self.user_item_counts.get(user_id, Counter())
        merchant = self.merchants.loc[merchant_id]
        category = merchant.get("primary_first_tag_id", "unknown")
        recency_fast, recency_slow = self._recency_scores(seq, merchant_id)
        values = {
            "fast_recency": float(recency_fast),
            "slow_recency": float(recency_slow),
            "repeat": float(np.log1p(counts.get(merchant_id, 0)) / np.log(5)),
            "transition": float(self._transition_score(seq, merchant_id)),
            "category": float(self.user_cat_counts.get(user_id, {}).get(category, 0.0)),
            "popularity": float(self.pop_log.get(merchant_id, 0.0)),
            "quality": float(self.quality.get(merchant_id, 0.5)),
        }
        weight_sum = max(float(sum(self.seq_weights.values())), 1e-12)
        return float(sum(self.seq_weights[key] * values.get(key, 0.0) for key in self.seq_weights) / weight_sum)


class SeqXQuadTripartiteRecommender(SeqTripartiteRecommender):
    name = "Seq-xQuAD-Tripartite"

    def __init__(
        self,
        user_weight: float = 0.93,
        fairness_weight: float = 0.025,
        eta_weight: float = 0.03,
        supply_weight: float = 0.015,
        diversity_weight: float = 0.12,
        tail_weight: float = 0.04,
    ):
        super().__init__(user_weight, fairness_weight, eta_weight, supply_weight)
        self.diversity_weight = diversity_weight
        self.tail_weight = tail_weight

    def _rerank_xquad(self, scores: dict[str, float], k: int) -> list[str]:
        if not scores:
            return []
        candidates = sorted(scores, key=scores.get, reverse=True)[:80]
        min_score = min(scores[m] for m in candidates)
        max_score = max(scores[m] for m in candidates)
        scale = max(max_score - min_score, 1e-9)
        relevance_weight = max(1.0 - self.diversity_weight - self.tail_weight, 0.0)
        selected: list[str] = []
        covered_categories: set[str] = set()
        while len(selected) < k and len(selected) < len(candidates):
            best_item = None
            best_score = -float("inf")
            for merchant_id in candidates:
                if merchant_id in selected:
                    continue
                merchant = self.merchants.loc[merchant_id]
                category = str(merchant.get("primary_first_tag_id", "unknown"))
                category_gain = 0.0 if category in covered_categories else 1.0
                order_count = float(merchant.get("order_count", 0) or 0)
                tail_gain = 1.0 / (1.0 + np.log1p(order_count))
                relevance = (scores[merchant_id] - min_score) / scale
                value = (
                    relevance_weight * relevance
                    + self.diversity_weight * category_gain
                    + self.tail_weight * tail_gain
                )
                if value > best_score:
                    best_item = merchant_id
                    best_score = value
            if best_item is None:
                break
            selected.append(best_item)
            covered_categories.add(str(self.merchants.loc[best_item].get("primary_first_tag_id", "unknown")))
        return selected

    def recommend_for_user(self, user_id: str, k: int, period: str = "lunch") -> tuple[list[str], dict[str, float]]:
        components = self._component_scores_for_candidates(user_id, self._sequential_candidates(user_id), period)
        scores = {merchant_id: values["final_score"] for merchant_id, values in components.items()}
        recs = self._rerank_xquad(scores, k)
        recs = self._remove_seen_and_backfill(user_id, recs, k)
        return recs, {m: scores.get(m, float(self.popular.get(m, 0))) for m in recs}


class SessionSpuTripartiteRecommender(SeqXQuadTripartiteRecommender):
    name = "Session-SPU-Tripartite"

    def __init__(
        self,
        user_weight: float = 0.86,
        fairness_weight: float = 0.025,
        eta_weight: float = 0.03,
        supply_weight: float = 0.015,
        diversity_weight: float = 0.12,
        tail_weight: float = 0.04,
        session_weight: float = 0.055,
        spu_weight: float = 0.015,
    ):
        super().__init__(user_weight, fairness_weight, eta_weight, supply_weight, diversity_weight, tail_weight)
        self.session_weight = session_weight
        self.spu_weight = spu_weight

    def fit(self, data: PreparedData) -> "SessionSpuTripartiteRecommender":
        super().fit(data)
        if not data.session_interactions.empty:
            sessions = data.session_interactions.copy()
            if "split" in sessions.columns:
                sessions = sessions[sessions["split"].astype(str).eq("train")].copy()
            sessions["rank"] = pd.to_numeric(sessions.get("rank", 1), errors="coerce").fillna(1)
            sort_columns = ["user_id"]
            ascending = [True]
            if "order_timestamp" in sessions.columns:
                sessions["order_timestamp"] = pd.to_numeric(sessions["order_timestamp"], errors="coerce").fillna(0)
                sort_columns.append("order_timestamp")
                ascending.append(False)
            sort_columns.append("rank")
            ascending.append(True)
            sessions = sessions.sort_values(sort_columns, ascending=ascending)
            self.session_by_user = {
                str(user_id): list(dict.fromkeys(group["wm_poi_id"].astype(str)))
                for user_id, group in sessions.groupby("user_id", sort=False)
            }
        else:
            self.session_by_user = {}

        self.user_spu_categories: dict[str, Counter[str]] = {}
        self.merchant_spu_categories: dict[str, Counter[str]] = {}
        if not data.order_spus_train.empty and "category" in data.spus.columns:
            order_spus = data.order_spus_train.merge(
                data.spus[["wm_food_spu_id", "category"]],
                on="wm_food_spu_id",
                how="left",
            ).dropna(subset=["category"])
            self.user_spu_categories = {
                str(user_id): Counter(group["category"].astype(str))
                for user_id, group in order_spus.groupby("user_id", sort=False)
            }
            self.merchant_spu_categories = {
                str(merchant_id): Counter(group["category"].astype(str))
                for merchant_id, group in order_spus.groupby("wm_poi_id", sort=False)
            }
        return self

    def _spu_affinity(self, user_id: str, merchant_id: str) -> float:
        user_counts = self.user_spu_categories.get(user_id, Counter())
        merchant_counts = self.merchant_spu_categories.get(merchant_id, Counter())
        if not user_counts or not merchant_counts:
            return 0.0
        overlap = set(user_counts) & set(merchant_counts)
        if not overlap:
            return 0.0
        numerator = sum(min(user_counts[key], merchant_counts[key]) for key in overlap)
        denominator = max(sum(user_counts.values()), 1)
        return float(min(numerator / denominator, 1.0))

    def _session_score(self, user_id: str, merchant_id: str) -> float:
        clicked = self.session_by_user.get(user_id, [])
        for rank, item in enumerate(clicked, start=1):
            if item == merchant_id:
                return float(np.exp(-(rank - 1) / 4.0))
        return 0.0

    def _sequential_candidates(self, user_id: str) -> list[str]:
        session_candidates = [m for m in self.session_by_user.get(user_id, []) if m in self.active_set]
        spu_categories = self.user_spu_categories.get(user_id, Counter())
        spu_candidates: list[str] = []
        if spu_categories:
            favorite_categories = {cat for cat, _ in spu_categories.most_common(3)}
            for merchant_id, categories in self.merchant_spu_categories.items():
                if merchant_id in self.active_set and favorite_categories & set(categories):
                    spu_candidates.append(merchant_id)
        base = super()._sequential_candidates(user_id)
        return [m for m in dict.fromkeys(session_candidates + spu_candidates[:120] + base) if m in self.merchants.index]

    def component_scores(self, user_id: str, merchant_id: str, period: str = "lunch") -> dict[str, float]:
        components = super().component_scores(user_id, merchant_id, period)
        session_score = self._session_score(user_id, merchant_id)
        spu_score = self._spu_affinity(user_id, merchant_id)
        components["session_score"] = session_score
        components["spu_score"] = spu_score
        components["final_score"] = float(
            components["final_score"]
            + self.session_weight * session_score
            + self.spu_weight * spu_score
        )
        return components


class KGTripartiteRecommender(SessionSpuTripartiteRecommender):
    """Session-SPU 三方推荐器之上叠加轻量知识图谱兴趣信号。

    思想来自 kg-demo（动态 KG 注意力模型）的免训练近似：用户兴趣 = 对其
    历史商家 KG 属性（品类/商圈/价位）的时间衰减加权分布，商家由同一组
    KG 属性刻画，kg_score 为两者在关系加权下的匹配度。torch 版的完整
    实现与 GPU 实验结果见 kg-demo/。
    """

    name = "KG-Tripartite"

    RELATION_FIELDS = {
        "has_category": "primary_first_tag_id",
        "located_in_area": "aor_id",
        "has_price_range": "avg_order_price",
    }

    def __init__(
        self,
        user_weight: float = 0.86,
        fairness_weight: float = 0.025,
        eta_weight: float = 0.03,
        supply_weight: float = 0.015,
        diversity_weight: float = 0.12,
        tail_weight: float = 0.04,
        session_weight: float = 0.055,
        spu_weight: float = 0.015,
        kg_weight: float = 0.05,
        kg_decay: float = 6.0,
        relation_weights: dict[str, float] | None = None,
    ):
        super().__init__(
            user_weight,
            fairness_weight,
            eta_weight,
            supply_weight,
            diversity_weight,
            tail_weight,
            session_weight,
            spu_weight,
        )
        self.kg_weight = kg_weight
        self.kg_decay = kg_decay
        self.relation_weights = relation_weights or {
            "has_category": 0.6,
            "has_price_range": 0.25,
            "located_in_area": 0.15,
        }

    def _merchant_kg_nodes(self, merchant: pd.Series) -> dict[str, str]:
        from .kg import price_bucket

        nodes: dict[str, str] = {}
        category = str(merchant.get("primary_first_tag_id", "") or "").strip()
        if category and category.lower() not in {"nan", "unknown"}:
            nodes["has_category"] = f"category:{category}"
        area = str(merchant.get("aor_id", "") or "").strip()
        if area and area.lower() not in {"nan", "unknown"}:
            nodes["located_in_area"] = f"area:{area}"
        bucket = price_bucket(merchant.get("avg_order_price"))
        if bucket != "price_unknown":
            nodes["has_price_range"] = f"price:{bucket}"
        return nodes

    def fit(self, data: PreparedData) -> "KGTripartiteRecommender":
        super().fit(data)
        self.merchant_kg_nodes: dict[str, dict[str, str]] = {
            str(merchant_id): self._merchant_kg_nodes(merchant)
            for merchant_id, merchant in self.merchants.iterrows()
        }

        orders = data.orders_train[["user_id", "wm_poi_id"] + (["order_timestamp"] if "order_timestamp" in data.orders_train.columns else [])].copy()
        orders["user_id"] = orders["user_id"].astype(str)
        orders["wm_poi_id"] = orders["wm_poi_id"].astype(str)
        if "order_timestamp" in orders.columns:
            orders = orders.sort_values(["user_id", "order_timestamp"], ascending=[True, False])

        self.user_kg_interests: dict[str, dict[str, dict[str, float]]] = {}
        for user_id, group in orders.groupby("user_id", sort=False):
            interests: dict[str, dict[str, float]] = {}
            for position, merchant_id in enumerate(group["wm_poi_id"]):
                weight = float(np.exp(-position / self.kg_decay))
                for relation, node in self.merchant_kg_nodes.get(merchant_id, {}).items():
                    bucket = interests.setdefault(relation, {})
                    bucket[node] = bucket.get(node, 0.0) + weight
            for relation, bucket in interests.items():
                total = sum(bucket.values())
                if total > 0:
                    interests[relation] = {node: value / total for node, value in bucket.items()}
            self.user_kg_interests[str(user_id)] = interests
        return self

    def _kg_affinity(self, user_id: str, merchant_id: str) -> float:
        interests = self.user_kg_interests.get(user_id)
        nodes = self.merchant_kg_nodes.get(merchant_id)
        if not interests or not nodes:
            return 0.0
        score = 0.0
        for relation, node in nodes.items():
            score += self.relation_weights.get(relation, 0.0) * interests.get(relation, {}).get(node, 0.0)
        return float(min(score, 1.0))

    def component_scores(self, user_id: str, merchant_id: str, period: str = "lunch") -> dict[str, float]:
        components = super().component_scores(user_id, merchant_id, period)
        kg_score = self._kg_affinity(user_id, merchant_id)
        components["kg_score"] = kg_score
        components["final_score"] = float(components["final_score"] + self.kg_weight * kg_score)
        return components


class TripartiteRerankRecommender(UserOnlyRecommender):
    name = "Tripartite-Rerank"

    def __init__(
        self,
        name: str = "Tripartite-Rerank",
        user_weight: float = 0.62,
        fairness_weight: float = 0.18,
        eta_weight: float = 0.14,
        supply_weight: float = 0.06,
    ):
        self.name = name
        self.user_weight = user_weight
        self.fairness_weight = fairness_weight
        self.eta_weight = eta_weight
        self.supply_weight = supply_weight

    def fit(self, data: PreparedData) -> "TripartiteRerankRecommender":
        super().fit(data)
        self.fair = fairness_scores(data.merchants)
        self.data = data
        return self

    def component_scores(self, user_id: str, merchant_id: str, period: str = "lunch") -> dict[str, float]:
        user_row = self.users.loc[user_id] if user_id in self.users.index else pd.Series(dtype=object)
        merchant = self.merchants.loc[merchant_id]
        user_score = self.user_score(user_id, merchant_id, period)
        eta = estimate_user_merchant_eta(user_row, merchant, period)
        eta_score = 1.0 - min(eta / 70.0, 1.0)
        supply = supply_score_for_merchant(merchant)
        fair = float(self.fair.get(merchant_id, 0.0))
        final = float(
            self.user_weight * user_score
            + self.fairness_weight * fair
            + self.eta_weight * eta_score
            + self.supply_weight * supply
        )
        return {
            "user_score": float(user_score),
            "merchant_fairness": fair,
            "eta_minutes": float(eta),
            "eta_score": float(eta_score),
            "supply_score": float(supply),
            "final_score": final,
        }

    def recommend_for_user(self, user_id: str, k: int, period: str = "lunch") -> tuple[list[str], dict[str, float]]:
        scores = {}
        for m in self._personalized_candidates(user_id):
            scores[m] = self.component_scores(user_id, m, period)["final_score"]
        ranked = sorted(scores, key=scores.get, reverse=True)
        recs = self._remove_seen_and_backfill(user_id, ranked, k)
        return recs, {m: scores[m] for m in recs}


class OursBalancedRecommender(TripartiteRerankRecommender):
    def __init__(self):
        super().__init__(
            name="Ours-Balanced",
            user_weight=0.72,
            fairness_weight=0.08,
            eta_weight=0.14,
            supply_weight=0.06,
        )


class OursFullRecommender(TripartiteRerankRecommender):
    def __init__(self):
        super().__init__(
            name="Ours-Full",
            user_weight=0.62,
            fairness_weight=0.18,
            eta_weight=0.14,
            supply_weight=0.06,
        )


def lightgbm_available() -> bool:
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        return False
    return True


def learned_ltr_model_name() -> str:
    return "LightGBM-LTR" if lightgbm_available() else "Logistic-LTR"


def build_learned_ltr_recommender(seed: int = 42) -> BaseRecommender:
    if lightgbm_available():
        return LightGBMRankerRecommender(seed=seed)
    return LogisticLTRRecommender(seed=seed)


def build_recommenders(seed: int = 42) -> list[BaseRecommender]:
    learned_seq_recommender = build_learned_ltr_recommender(seed=seed)
    return [
        PopularRecommender(),
        BPRMFRecommender(seed=seed),
        UserOnlyRecommender(),
        learned_seq_recommender,
        SeqTunedRecommender(),
        SeqXQuadTripartiteRecommender(),
        SessionSpuTripartiteRecommender(),
        KGTripartiteRecommender(),
    ]
