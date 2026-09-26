import json


def test_storage_benchmark_checks_loaded_tensor_and_keeps_phase_measurements(tmp_path):
    from trainguard.storage_benchmark import run_storage_benchmark
    directory = run_storage_benchmark(tmp_path, sizes_mib=(1,), repetitions=1)
    result = json.loads((directory / 'storage.json').read_text())
    assert result['status'] == 'SUCCEEDED'
    assert {row['mode'] for row in result['rows']} == {'sync', 'async'}
    for row in result['rows']:
        assert row['validated']
        assert row['tensor_bytes'] == 1048576
        assert row['payload_bytes'] >= row['tensor_bytes']
        assert all(row[field] >= 0 for field in ('staging_seconds', 'upload_seconds', 'hash_commit_seconds', 'load_seconds'))
