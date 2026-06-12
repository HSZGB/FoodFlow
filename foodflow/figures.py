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


def _save_tripartite_summary(offline: pd.DataFrame, sim: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    sns.barplot(data=offline, x="model", y="Recall@20", ax=axes[0], color="#2563EB")
    axes[0].set_title("User: Recall@20")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=35)

    if "Coverage@20" in offline.columns:
        sns.barplot(data=offline, x="model", y="Coverage@20", ax=axes[1], color="#059669")
        axes[1].set_title("Merchant: Coverage@20")
        axes[1].set_xlabel("")
        axes[1].tick_params(axis="x", rotation=35)

    sns.barplot(data=sim, x="policy", y="platform_utility", ax=axes[2], color="#B5121B")
    axes[2].set_title("Platform: Utility")
    axes[2].set_xlabel("")
    axes[2].tick_params(axis="x", rotation=35)

    fig.suptitle("FoodFlow tripartite scorecard", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


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

        if {"Recall@20", "Coverage@20"}.issubset(offline.columns):
            plt.figure(figsize=(7.5, 5.2))
            sns.scatterplot(data=offline, x="Recall@20", y="Coverage@20", hue="model", s=90)
            plt.title("Accuracy-Coverage trade-off")
            plt.tight_layout()
            out = figures_dir / "tradeoff_recall_coverage.png"
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
        if {"avg_eta", "platform_utility", "completed_orders"}.issubset(sim.columns):
            plt.figure(figsize=(8.2, 5.2))
            sns.scatterplot(
                data=sim,
                x="avg_eta",
                y="platform_utility",
                hue="policy",
                size="completed_orders",
                sizes=(70, 240),
            )
            plt.title("ETA-Utility trade-off")
            plt.tight_layout()
            out = figures_dir / "tradeoff_eta_utility.png"
            plt.savefig(out, dpi=180)
            plt.close()
            made.append(out)

    if offline_path.exists() and sim_path.exists():
        offline = pd.read_csv(offline_path)
        sim = pd.read_csv(sim_path)
        if {"model", "Recall@20", "Coverage@20"}.issubset(offline.columns) and {
            "policy",
            "platform_utility",
        }.issubset(sim.columns):
            out = figures_dir / "tripartite_scorecard.png"
            _save_tripartite_summary(offline, sim, out)
            made.append(out)
    return made
