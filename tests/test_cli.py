from foodflow.cli import build_parser


def test_simulate_accepts_rider_task_calibration_file():
    args = build_parser().parse_args(["simulate", "--rider-tasks", "lade_sample.csv"])

    assert str(args.rider_tasks) == "lade_sample.csv"
