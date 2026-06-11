from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class PreparedData:
    users: pd.DataFrame
    merchants: pd.DataFrame
    orders_train: pd.DataFrame
    orders_test: pd.DataFrame
    test_interactions: pd.DataFrame
    spus: pd.DataFrame

    @classmethod
    def load(cls, processed_dir: Path | str) -> "PreparedData":
        base = Path(processed_dir)
        return cls(
            users=pd.read_csv(base / "users.csv", dtype=str).infer_objects(),
            merchants=pd.read_csv(base / "merchants.csv", dtype=str).infer_objects(),
            orders_train=pd.read_csv(base / "orders_train.csv", dtype=str).infer_objects(),
            orders_test=pd.read_csv(base / "orders_test.csv", dtype=str).infer_objects(),
            test_interactions=pd.read_csv(base / "test_interactions.csv", dtype=str).infer_objects(),
            spus=pd.read_csv(base / "spus.csv", dtype=str).infer_objects(),
        ).coerce()

    def coerce(self) -> "PreparedData":
        for df in [self.users, self.merchants, self.orders_train, self.orders_test, self.test_interactions, self.spus]:
            for col in df.columns:
                if col.endswith("_id") or col in {"user_id", "wm_poi_id", "wm_food_spu_id", "wm_order_id"}:
                    df[col] = df[col].astype(str)
        for col in ["order_count", "avg_order_price", "poi_score", "delivery_comment_avg_score", "food_comment_avg_score", "lng", "lat"]:
            if col in self.merchants.columns:
                self.merchants[col] = pd.to_numeric(self.merchants[col], errors="coerce")
        for col in ["avg_pay_amt", "history_orders", "avg_order_price", "lng", "lat"]:
            if col in self.users.columns:
                self.users[col] = pd.to_numeric(self.users[col], errors="coerce")
        for df in [self.orders_train, self.orders_test]:
            for col in ["order_price", "order_timestamp"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
        return self

    @property
    def merchant_ids(self) -> list[str]:
        return self.merchants["wm_poi_id"].astype(str).tolist()

    @property
    def user_ids(self) -> list[str]:
        return self.users["user_id"].astype(str).tolist()

    def truth_by_user(self) -> dict[str, set[str]]:
        return (
            self.test_interactions.groupby("user_id")["wm_poi_id"]
            .apply(lambda s: set(s.astype(str)))
            .to_dict()
        )

    def history_by_user(self) -> dict[str, list[str]]:
        return (
            self.orders_train.groupby("user_id")["wm_poi_id"]
            .apply(lambda s: s.astype(str).tolist())
            .to_dict()
        )

    def user_period_by_user(self) -> dict[str, str]:
        if "ord_period_name" not in self.orders_test.columns:
            return {}
        return (
            self.orders_test.groupby("user_id")["ord_period_name"]
            .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0])
            .astype(str)
            .to_dict()
        )
