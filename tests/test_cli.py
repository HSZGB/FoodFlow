from foodflow.cli import build_parser


def test_simulate_accepts_rider_task_calibration_file():
    args = build_parser().parse_args(["simulate", "--rider-tasks", "lade_sample.csv"])

    assert str(args.rider_tasks) == "lade_sample.csv"


def test_simulate_accepts_food_scaled_rider_calibration_profile():
    args = build_parser().parse_args(
        ["simulate", "--rider-tasks", "lade_sample.csv", "--rider-calibration-profile", "food-scaled"]
    )

    assert args.rider_calibration_profile == "food-scaled"
