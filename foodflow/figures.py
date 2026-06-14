from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .frontier import build_tripartite_frontier
from .io import ensure_dir


HIGHLIGHT = {
    "Seq-xQuAD-Tripartite": "#B5121B",
    "Seq-xQuAD": "#2563EB",
    "Seq-Hybrid": "#60A5FA",
    "Seq-Tripartite": "#0F766E",
    "Ours-Full": "#DC6803",
    "Ours-Balanced": "#7C3AED",
}
DEFAULT_COLOR = "#CBD5E1"
TEXT_COLOR = "#0F172A"
GRID_COLOR = "#E2E8F0"


def _color_for(label: str) -> str:
    for key, color in HIGHLIGHT.items():
        if key in str(label):
            return color
    return DEFAULT_COLOR


def _annotate_highlights(ax, df: pd.DataFrame, x: str, y: str, label_col: str) -> None:
    offsets = {
        "Seq-xQuAD-Tripartite": (8, 13),
        "Ours-Full": (8, -13),
        "Seq-xQuAD": (8, 8),
        "Seq-Hybrid": (8, -10),
        "Seq-Tripartite": (8, 10),
        "Ours-Balanced": (8, -10),
    }
    for _, row in df.iterrows():
        label = str(row[label_col])
        if not any(key in label for key in HIGHLIGHT):
            continue
        offset = next((value for key, value in offsets.items() if key in label), (7, 5))
        ax.annotate(
            label,
            (float(row[x]), float(row[y])),
            xytext=offset,
            textcoords="offset points",
            fontsize=8.2,
            color=TEXT_COLOR,
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=GRID_COLOR, alpha=0.9),
        )


def _save_bar(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    path: Path,
    higher_better: bool = True,
) -> None:
    plot_df = df[[x, y]].copy()
    plot_df[y] = pd.to_numeric(plot_df[y], errors="coerce").fillna(0.0)
    plot_df = plot_df.sort_values(y, ascending=higher_better)
    colors = [_color_for(label) for label in plot_df[x]]

    fig_height = max(4.8, 0.48 * len(plot_df) + 1.6)
    fig, ax = plt.subplots(figsize=(10.5, fig_height))
    bars = ax.barh(plot_df[x].astype(str), plot_df[y], color=colors, edgecolor="white", linewidth=0.8)
    ax.set_title(title, fontsize=14, fontweight="bold", color=TEXT_COLOR, loc="left", pad=12)
    ax.set_xlabel(y, color="#475569")
    ax.set_ylabel("")
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.tick_params(axis="both", colors="#334155", labelsize=9)

    xmax = max(float(plot_df[y].max()), 1e-9)
    ax.set_xlim(0, xmax * 1.14)
    for bar, value in zip(bars, plot_df[y]):
        ax.text(
            bar.get_width() + xmax * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}" if value < 10 else f"{value:.2f}",
            va="center",
            ha="left",
            fontsize=8.5,
            color="#334155",
        )
    fig.tight_layout()
    fig.savefig(path, dpi=190, facecolor="white")
    plt.close(fig)


def _save_tripartite_summary(offline: pd.DataFrame, sim: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    panels = [
        (offline, "model", "Recall@20", "User side: Recall@20", True),
        (offline, "model", "Coverage@20", "Merchant side: Coverage@20", True),
        (sim, "policy", "platform_utility", "Platform side: Utility", True),
    ]
    for ax, (df, label_col, metric_col, title, higher_better) in zip(axes, panels):
        if metric_col not in df.columns:
            ax.axis("off")
            continue
        plot_df = df[[label_col, metric_col]].copy()
        plot_df[metric_col] = pd.to_numeric(plot_df[metric_col], errors="coerce").fillna(0.0)
        plot_df = plot_df.sort_values(metric_col, ascending=False).head(5).sort_values(metric_col)
        colors = [_color_for(label) for label in plot_df[label_col]]
        ax.barh(plot_df[label_col].astype(str), plot_df[metric_col], color=colors, edgecolor="white")
        ax.set_title(title, fontsize=12, fontweight="bold", loc="left", color=TEXT_COLOR)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(axis="x", color=GRID_COLOR)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color(GRID_COLOR)
        xmax = max(float(plot_df[metric_col].max()), 1e-9)
        ax.set_xlim(0, xmax * 1.18)
        for y_pos, value in enumerate(plot_df[metric_col]):
            ax.text(value + xmax * 0.02, y_pos, f"{value:.4f}", va="center", fontsize=8.5, color="#334155")

    fig.suptitle("FoodFlow tripartite scorecard", fontsize=16, fontweight="bold", color=TEXT_COLOR)
    fig.tight_layout()
    fig.savefig(path, dpi=190, facecolor="white")
    plt.close(fig)


def _save_frontier_plot(frontier: pd.DataFrame, path: Path) -> None:
    if frontier.empty:
        return
    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    plot_df = frontier[(frontier["Recall@20"] >= 0.35) | (frontier["platform_utility"] >= 0.45)].copy()
    dominated = plot_df[~plot_df["is_frontier"]]
    front = plot_df[plot_df["is_frontier"]]
    if not dominated.empty:
        ax.scatter(
            dominated["Recall@20"],
            dominated["platform_utility"],
            s=(dominated["on_time_rate"] * 300 + 60),
            marker="x",
            linewidths=2.0,
            color="#CBD5E1",
            label="Dominated",
            zorder=2,
        )
    for _, row in front.iterrows():
        ax.scatter(
            float(row["Recall@20"]),
            float(row["platform_utility"]),
            s=float(row["on_time_rate"]) * 320 + 80,
            color=_color_for(str(row["policy"])),
            edgecolor="white",
            linewidth=1.4,
            label=str(row["policy"]),
            zorder=3,
        )
    selected_labels = {
        "Seq-xQuAD-Tripartite": (8, 12),
        "Seq-xQuAD + MinETA": (8, -18),
        "Seq-Tripartite": (8, 7),
        "Ours-Full": (8, -16),
    }
    for _, row in plot_df.iterrows():
        label = str(row["policy"])
        if label not in selected_labels:
            continue
        ax.annotate(
            label,
            (float(row["Recall@20"]), float(row["platform_utility"])),
            xytext=selected_labels[label],
            textcoords="offset points",
            fontsize=8.8,
            color=TEXT_COLOR,
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=GRID_COLOR, alpha=0.92),
        )
    ax.set_title("Pareto view: accuracy vs platform utility", fontsize=14, fontweight="bold", loc="left")
    ax.set_xlabel("Recall@20")
    ax.set_ylabel("Platform utility")
    ax.grid(color=GRID_COLOR)
    ax.text(
        0.01,
        0.02,
        "Circle size: on-time rate. Grey x: dominated strategy.",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#475569",
    )
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=190, facecolor="white")
    plt.close(fig)


def generate_figures(results_dir: Path, figures_dir: Path) -> list[Path]:
    figures_dir = ensure_dir(figures_dir)
    made: list[Path] = []
    offline_path = results_dir / "offline_metrics.csv"
    sim_path = results_dir / "simulation_metrics.csv"
    sns.set_theme(style="whitegrid", font="DejaVu Sans")
    plt.rcParams.update(
        {
            "axes.facecolor": "#F8FAFC",
            "figure.facecolor": "white",
            "axes.edgecolor": GRID_COLOR,
            "axes.labelcolor": "#475569",
            "xtick.color": "#334155",
            "ytick.color": "#334155",
        }
    )

    if offline_path.exists():
        offline = pd.read_csv(offline_path)
        for metric, filename, title in [
            ("Recall@20", "offline_recall20.png", "Offline recommendation Recall@20"),
            ("NDCG@20", "offline_ndcg20.png", "Offline recommendation NDCG@20"),
            ("Coverage@20", "offline_coverage20.png", "Provider coverage@20"),
            ("ExposureGini", "offline_exposure_gini.png", "Merchant exposure Gini"),
            ("CategoryJSD@20", "offline_category_jsd20.png", "Category calibration JSD@20"),
        ]:
            if metric in offline.columns:
                out = figures_dir / filename
                _save_bar(
                    offline,
                    "model",
                    metric,
                    title,
                    out,
                    higher_better=metric not in {"ExposureGini", "CategoryJSD@20"},
                )
                made.append(out)

        if {"NDCG@20", "ExposureGini"}.issubset(offline.columns):
            fig, ax = plt.subplots(figsize=(8.2, 5.6))
            sns.scatterplot(
                data=offline,
                x="NDCG@20",
                y="ExposureGini",
                hue="model",
                palette={label: _color_for(label) for label in offline["model"].astype(str)},
                s=110,
                ax=ax,
            )
            ax.set_title("Accuracy-Fairness trade-off", fontsize=14, fontweight="bold", loc="left")
            ax.grid(color=GRID_COLOR)
            _annotate_highlights(ax, offline, "NDCG@20", "ExposureGini", "model")
            ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
            fig.tight_layout()
            out = figures_dir / "tradeoff_ndcg_gini.png"
            fig.savefig(out, dpi=190, facecolor="white")
            plt.close(fig)
            made.append(out)

        if {"Recall@20", "Coverage@20"}.issubset(offline.columns):
            fig, ax = plt.subplots(figsize=(8.2, 5.6))
            sns.scatterplot(
                data=offline,
                x="Recall@20",
                y="Coverage@20",
                hue="model",
                palette={label: _color_for(label) for label in offline["model"].astype(str)},
                s=110,
                ax=ax,
            )
            ax.set_title("Accuracy-Coverage trade-off", fontsize=14, fontweight="bold", loc="left")
            ax.grid(color=GRID_COLOR)
            _annotate_highlights(ax, offline, "Recall@20", "Coverage@20", "model")
            ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
            fig.tight_layout()
            out = figures_dir / "tradeoff_recall_coverage.png"
            fig.savefig(out, dpi=190, facecolor="white")
            plt.close(fig)
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
                _save_bar(
                    sim,
                    "policy",
                    metric,
                    title,
                    out,
                    higher_better=metric in {"platform_utility"},
                )
                made.append(out)
        if {"avg_eta", "platform_utility", "completed_orders"}.issubset(sim.columns):
            fig, ax = plt.subplots(figsize=(8.8, 5.6))
            sns.scatterplot(
                data=sim,
                x="avg_eta",
                y="platform_utility",
                hue="policy",
                size="completed_orders",
                sizes=(70, 240),
                palette={label: _color_for(label) for label in sim["policy"].astype(str)},
                ax=ax,
            )
            ax.set_title("ETA-Utility trade-off", fontsize=14, fontweight="bold", loc="left")
            ax.grid(color=GRID_COLOR)
            _annotate_highlights(ax, sim, "avg_eta", "platform_utility", "policy")
            ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
            fig.tight_layout()
            out = figures_dir / "tradeoff_eta_utility.png"
            fig.savefig(out, dpi=190, facecolor="white")
            plt.close(fig)
            made.append(out)

    if offline_path.exists() and sim_path.exists():
        offline = pd.read_csv(offline_path)
        sim = pd.read_csv(sim_path)
        if {"model", "Recall@20", "Coverage@20"}.issubset(offline.columns) and {
            "policy",
            "platform_utility",
        }.issubset(sim.columns):
            frontier = build_tripartite_frontier(offline, sim)
            if not frontier.empty:
                frontier.to_csv(results_dir / "tripartite_frontier.csv", index=False)
                out = figures_dir / "pareto_recall_utility.png"
                _save_frontier_plot(frontier, out)
                made.append(out)
            out = figures_dir / "tripartite_scorecard.png"
            _save_tripartite_summary(offline, sim, out)
            made.append(out)
    return made
