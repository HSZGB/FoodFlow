from __future__ import annotations

from pathlib import Path
from time import perf_counter

import pandas as pd

from .data import PreparedData
from .io import ensure_dir
from .metrics import evaluate_recommendations
from .recommenders import build_recommenders


def run_offline_eval(
    processed_dir: Path,
    output: Path,
    ks: list[int],
    seed: int = 42,
    user_limit: int | None = 300,
) -> pd.DataFrame:
    data = PreparedData.load(processed_dir)
    truth = data.truth_by_user()
    users = list(truth.keys())
    if user_limit and len(users) > user_limit:
        users = users[:user_limit]
        truth = {u: truth[u] for u in users}
    periods = data.user_period_by_user()
    rows = []
    for recommender in build_recommenders(seed=seed):
        started = perf_counter()
        recommender.fit(data)
        fit_done = perf_counter()
        result = recommender.recommend(users, max(ks), periods)
        rec_done = perf_counter()
        metrics = evaluate_recommendations(result.recommendations, truth, data.merchants, ks, data.history_by_user())
        metrics["model"] = result.name
        rows.append(metrics)
        print(
            f"[eval] {result.name}: fit={fit_done - started:.2f}s recommend={rec_done - fit_done:.2f}s users={len(users)}",
            flush=True,
        )
    df = pd.DataFrame(rows)
    cols = ["model"] + [c for c in df.columns if c != "model"]
    df = df[cols].sort_values("model")
    ensure_dir(output.parent)
    df.to_csv(output, index=False)
    return df
