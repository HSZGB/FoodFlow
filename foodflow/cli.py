from __future__ import annotations

import argparse
from pathlib import Path

from .config import DEFAULT_PROCESSED_DIR, DEFAULT_RAW_DIR
from .data import PreparedData
from .download import download_trd
from .evaluate import run_offline_eval
from .figures import generate_figures
from .mock_data import make_mock_trd
from .preprocess import preprocess
from .report import build_report
from .simulator import run_simulation


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

    p = sub.add_parser("figures")
    p.add_argument("--results-dir", type=Path, default=Path("outputs/results"))
    p.add_argument("--figures-dir", type=Path, default=Path("outputs/figures"))

    p = sub.add_parser("report")
    p.add_argument("--results-dir", type=Path, default=Path("outputs/results"))
    p.add_argument("--figures-dir", type=Path, default=Path("outputs/figures"))
    p.add_argument("--output", type=Path, default=Path("report/实验报告.md"))
    p.add_argument("--data-note", type=Path, default=Path("data/processed/data_note.json"))

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
        df = run_simulation(data, seed=args.seed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(df.to_string(index=False))
    elif args.command == "figures":
        paths = generate_figures(args.results_dir, args.figures_dir)
        print(f"Generated {len(paths)} figures under {args.figures_dir}")
    elif args.command == "report":
        path = build_report(args.results_dir, args.figures_dir, args.output, args.data_note)
        print(f"Report written to {path}")
    elif args.command == "explain-case":
        from .explain import explain_recommendation

        data = PreparedData.load(args.processed_dir)
        print(explain_recommendation(data, args.user_id, args.merchant_id))
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
