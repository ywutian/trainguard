from trainguard.benchmark import mode_order, summarize_rows


def test_benchmark_summary_keeps_raw_measurements_and_median_range() -> None:
    rows = [
        {"mode": "sync", "elapsed_seconds": 3.0},
        {"mode": "sync", "elapsed_seconds": 1.0},
        {"mode": "sync", "elapsed_seconds": 2.0},
        {"mode": "async", "elapsed_seconds": 4.0},
        {"mode": "async", "elapsed_seconds": 5.0},
        {"mode": "async", "elapsed_seconds": 6.0},
    ]
    summary = summarize_rows(rows)
    assert summary["sync"] == {"median_seconds": 2.0, "min_seconds": 1.0, "max_seconds": 3.0}
    assert summary["async"] == {"median_seconds": 5.0, "min_seconds": 4.0, "max_seconds": 6.0}


def test_repetition_order_rotates_modes() -> None:
    assert mode_order(1) == ("none", "sync", "async")
    assert mode_order(2) == ("sync", "async", "none")
    assert mode_order(3) == ("async", "none", "sync")


def test_summary_can_measure_training_window_separately_from_launch() -> None:
    rows = [
        {"mode": "sync", "elapsed_seconds": 10.0, "training_seconds": 2.0},
        {"mode": "sync", "elapsed_seconds": 11.0, "training_seconds": 4.0},
        {"mode": "sync", "elapsed_seconds": 12.0, "training_seconds": 3.0},
    ]
    assert summarize_rows(rows, "training_seconds") == {
        "sync": {"median_seconds": 3.0, "min_seconds": 2.0, "max_seconds": 4.0}
    }
