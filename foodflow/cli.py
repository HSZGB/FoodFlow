from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .audit import audit_data
from .config import DEFAULT_PROCESSED_DIR, DEFAULT_RAW_DIR
from .data import PreparedData
from .download import download_trd
from .evaluate import run_offline_eval
from .figures import generate_figures
from .mock_data import make_mock_trd
from .preprocess import preprocess
from .report import build_report
from .rider_data import adapt_calibration_for_food_delivery, calibration_diagnostics, estimate_rider_calibration
from .simulator import run_simulation, run_simulation_multi_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="foodflow", description="FoodFlow recommendation and delivery simulation CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download")
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    p.add_argument("--skip-graph", action="store_true", default=True)
    p.add_argument("--required-only", action="store_true")

    p = sub.add_parser("mock-data")
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    p.add_argument("--seed", type=int, default=42)

    p = sub.add_parser("preprocess")
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    p.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    p.add_argument("--sample-orders", type=int, default=50000)
    p.add_argument("--seed", type=int, default=42)

    p = sub.add_parser("eval-offline")
    p.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    p.add_argument("--output", type=Path, default=Path("outputs/results/offline_metrics.csv"))
    p.add_argument("--top-k", type=int, nargs="+", default=[10, 20])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--user-limit", type=int, default=300)

    p = sub.add_parser("simulate")
    p.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    p.add_argument("--output", type=Path, default=Path("outputs/results/simulation_metrics.csv"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quiet", action="store_true", help="Disable per-policy progress logs.")
    p.add_argument(
        "--rider-tasks",
        type=Path,
        default=None,
        help="Optional delivery task CSV, including LaDe delivery files, used to calibrate synthetic rider speed, service time, and load.",
    )
    p.add_argument(
        "--rider-calibration-profile",
        choices=["raw", "food-scaled"],
        default="raw",
        help="Use raw external-task calibration, or keep LaDe load/reliability while rescaling speed/service time to food-delivery SLA.",
    )
    p.add_argument(
        "--calibration-output",
        type=Path,
        default=None,
        help="Where to write calibration diagnostics JSON (defaults to <output stem>_calibration.json when --rider-tasks is set).",
    )
    p.add_argument(
        "--simulation-seeds",
        type=int,
        nargs="+",
        default=None,
        help="Run the simulation once per seed and aggregate mean/std/95%% CI per policy (overrides --seed).",
    )

    p = sub.add_parser("figures")
    p.add_argument("--results-dir", type=Path, default=Path("outputs/results"))
    p.add_argument("--figures-dir", type=Path, default=Path("outputs/figures"))

    p = sub.add_parser("audit-data")
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    p.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    p.add_argument("--output", type=Path, default=Path("outputs/results/data_audit.json"))
    p.add_argument("--markdown", type=Path, default=Path("docs/DATA_AUDIT.md"))

    p = sub.add_parser("report")
    p.add_argument("--results-dir", type=Path, default=Path("outputs/results"))
    p.add_argument("--figures-dir", type=Path, default=Path("outputs/figures"))
    p.add_argument("--output", type=Path, default=Path("report/实验报告.md"))
    p.add_argument("--data-note", type=Path, default=Path("data/processed/data_note.json"))
    p.add_argument("--data-audit", type=Path, default=Path("outputs/results/data_audit.json"))

    p = sub.add_parser("explain-case")
    p.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    p.add_argument("--user-id", type=str, required=True)
    p.add_argument("--merchant-id", type=str, required=True)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "download":
        paths = download_trd(args.raw_dir, skip_graph=args.skip_graph, required_only=args.required_only)
        print(f"Downloaded/verified {len(paths)} files under {args.raw_dir}")
    elif args.command == "mock-data":
        make_mock_trd(args.raw_dir, seed=args.seed)
        print(f"Mock TRD-like files written under {args.raw_dir}")
    elif args.command == "preprocess":
        preprocess(args.raw_dir, args.processed_dir, sample_orders=args.sample_orders, seed=args.seed)
        print(f"Processed files written under {args.processed_dir}")
    elif args.command == "eval-offline":
        df = run_offline_eval(args.processed_dir, args.output, args.top_k, seed=args.seed, user_limit=args.user_limit)
        print(df.to_string(index=False))
    elif args.command == "simulate":
        data = PreparedData.load(args.processed_dir)
        rider_calibration = None
        if args.rider_tasks:
            tasks = pd.read_csv(args.rider_tasks)
            raw_calibration = estimate_rider_calibration(tasks)
            rider_calibration = raw_calibration
            if args.rider_calibration_profile == "food-scaled":
                rider_calibration = adapt_calibration_for_food_delivery(raw_calibration)
            diagnostics = calibration_diagnostics(
                tasks,
                rider_calibration,
                raw=raw_calibration,
                profile=args.rider_calibration_profile,
                seed=args.seed,
            )
            diagnostics["source"] = str(args.rider_tasks)
            calibration_output = args.calibration_output or args.output.with_name(f"{args.output.stem}_calibration.json")
            calibration_output.parent.mkdir(parents=True, exist_ok=True)
            calibration_output.write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"Calibration diagnostics written to {calibration_output}")
        if args.simulation_seeds:
            df = run_simulation_multi_seed(
                data,
                seeds=args.simulation_seeds,
                verbose=not args.quiet,
                rider_calibration=rider_calibration,
            )
        else:
            df = run_simulation(data, seed=args.seed, verbose=not args.quiet, rider_calibration=rider_calibration)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(df.to_string(index=False))
    elif args.command == "figures":
        paths = generate_figures(args.results_dir, args.figures_dir)
        print(f"Generated {len(paths)} figures under {args.figures_dir}")
    elif args.command == "audit-data":
        audit = audit_data(args.raw_dir, args.processed_dir, args.output, args.markdown)
        print(
            f"Data audit written to {args.output}; train_mode={audit['train_mode']} "
            f"sample_fraction={audit['train_sample_fraction']:.4f}"
        )
    elif args.command == "report":
        path = build_report(args.results_dir, args.figures_dir, args.output, args.data_note, args.data_audit)
        print(f"Report written to {path}")
    elif args.command == "explain-case":
        from .explain import explain_recommendation

        data = PreparedData.load(args.processed_dir)
        print(explain_recommendation(data, args.user_id, args.merchant_id))
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
