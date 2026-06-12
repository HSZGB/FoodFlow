from pathlib import Path

from foodflow.audit import audit_data
from foodflow.data import PreparedData
from foodflow.evaluate import run_offline_eval
from foodflow.figures import generate_figures
from foodflow.mock_data import make_mock_trd
from foodflow.preprocess import preprocess
from foodflow.report import build_report
from foodflow.simulator import run_simulation


def test_tiny_pipeline(tmp_path: Path):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    results = tmp_path / "results"
    figures = tmp_path / "figures"
    report = tmp_path / "report.md"
    make_mock_trd(raw, seed=123, users=32, merchants=18, foods=45)
    preprocess(raw, processed, sample_orders=300, seed=123)
    offline = run_offline_eval(processed, results / "offline_metrics.csv", [10], seed=123, user_limit=25)
    assert len(offline) >= 5
    data = PreparedData.load(processed)
    sim = run_simulation(data, seed=123, requests_per_step=8, steps=3, top_k=10)
    sim.to_csv(results / "simulation_metrics.csv", index=False)
    assert len(sim) >= 4
    audit = audit_data(raw, processed, results / "data_audit.json", tmp_path / "DATA_AUDIT.md")
    assert audit["required_raw_files_present"]
    assert audit["processed_train_orders"] > 0
    made = generate_figures(results, figures)
    assert made
    build_report(results, figures, report, processed / "data_note.json", results / "data_audit.json")
    assert report.exists()
