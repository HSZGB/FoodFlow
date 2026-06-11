from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .io import ensure_dir


def _save_bar(df: pd.DataFrame, x: str, y: str, title: str, path: Path, rotate: bool = True) -> None:
    plt.figure(figsize=(10, 5.6))
    sns.barplot(data=df, x=x, y=y, color="#3B82F6")
    plt.title(title)
    plt.xlabel("")
    plt.ylabel(y)
    if rotate:
        plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def generate_figures(results_dir: Path, figures_dir: Path) -> list[Path]:
    figures_dir = ensure_dir(figures_dir)
    made: list[Path] = []
    offline_path = results_dir / "offline_metrics.csv"
    sim_path = results_dir / "simulation_metrics.csv"
    sns.set_theme(style="whitegrid", font="DejaVu Sans")

    if offline_path.exists():
        offline = pd.read_csv(offline_path)
        for metric, filename, title in [
            ("Recall@20", "offline_recall20.png", "Offline recommendation Recall@20"),
            ("NDCG@20", "offline_ndcg20.png", "Offline recommendation NDCG@20"),
            ("Coverage@20", "offline_coverage20.png", "Provider coverage@20"),
            ("ExposureGini", "offline_exposure_gini.png", "Merchant exposure Gini"),
        ]:
            if metric in offline.columns:
                out = figures_dir / filename
                _save_bar(offline, "model", metric, title, out)
                made.append(out)

        if {"NDCG@20", "ExposureGini"}.issubset(offline.columns):
            plt.figure(figsize=(7.5, 5.2))
            sns.scatterplot(data=offline, x="NDCG@20", y="ExposureGini", hue="model", s=90)
            plt.title("Accuracy-Fairness trade-off")
            plt.tight_layout()
            out = figures_dir / "tradeoff_ndcg_gini.png"
            plt.savefig(out, dpi=180)
            plt.close()
            made.append(out)

    if sim_path.exists():
        sim = pd.read_csv(sim_path)
        for metric, filename, title in [
            ("avg_eta", "simulation_avg_eta.png", "Average delivery ETA"),
            ("timeout_rate", "simulation_timeout_rate.png", "Timeout rate"),
            ("rider_load_std", "simulation_rider_load_std.png", "Rider load standard deviation"),
            ("platform_utility", "simulation_platform_utility.png", "Platform utility"),
            ("merchant_exposure_gini", "simulation_exposure_gini.png", "Simulation merchant exposure Gini"),
        ]:
            if metric in sim.columns:
                out = figures_dir / filename
                _save_bar(sim, "policy", metric, title, out)
                made.append(out)
    return made
