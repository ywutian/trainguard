import importlib.util
from pathlib import Path


def test_success_status_without_complete_cpu_matrix_is_rejected() -> None:
    path = Path(__file__).parents[1] / "scripts/run_simulation_closure.py"
    spec = importlib.util.spec_from_file_location("run_simulation_closure", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert not module._acceptance_complete({
        "status": "SUCCEEDED", "reference_status": "VALIDATED", "cases": [],
    })
