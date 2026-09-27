import hashlib
import json
from pathlib import Path

from trainguard.config import ProjectConfig, load_config
from trainguard.controller import run
from trainguard.validation import validate_runs


def test_real_data_with_workers_accumulation_and_recovery(tmp_path):
    dataset = tmp_path / 'tokens.jsonl'
    dataset.write_text(''.join(json.dumps({'tokens': [(i + j) % 128 for j in range(25)]}) + '\n'
                               for i in range(7)))
    raw = load_config(Path(__file__).parents[1] / 'configs/cpu_demo.yaml').model_dump()
    raw['model']['dropout'] = 0.2
    raw['training'].update(total_steps=6, gradient_accumulation_steps=2, dataloader_workers=2,
                           prefetch_factor=2)
    raw['data'] = {'kind': 'jsonl', 'path': str(dataset),
                   'sha256': hashlib.sha256(dataset.read_bytes()).hexdigest(),
                   'shuffle': True, 'random_crop': True}
    config = ProjectConfig.model_validate(raw)
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config.model_dump()))
    reference, ok = run(path, tmp_path / 'runs')
    assert ok, (reference / 'launcher.log').read_text()
    raw['checkpoint'] = {'mode': 'async', 'interval_steps': 1}
    raw['fault'] = {'kind': 'worker_exit', 'step': 3, 'rank': 0, 'require_committed_step': 1}
    path.write_text(json.dumps(raw))
    recovered, ok = run(path, tmp_path / 'runs', allow_experiment=True)
    assert ok, (recovered / 'launcher.log').read_text()
    result = validate_runs(reference, recovered)
    assert result['passed'], result
    summary = json.loads((recovered / 'summary.json').read_text())
    assert summary['consumed_batches'] == 12
    assert summary['optimizer_updates'] == 6


def test_cpu_bfloat16_accumulation_is_recoverable(tmp_path):
    raw = load_config(Path(__file__).parents[1] / 'configs/cpu_demo.yaml').model_dump()
    raw['training'].update(precision='bf16', gradient_accumulation_steps=2)
    raw['model']['dropout'] = 0.2
    config = ProjectConfig.model_validate(raw)
    path = tmp_path / 'bf16.json'
    path.write_text(json.dumps(config.model_dump()))
    reference, ok = run(path, tmp_path / 'runs')
    assert ok, (reference / 'launcher.log').read_text()
    raw['checkpoint'] = {'mode': 'sync', 'interval_steps': 1}
    raw['fault'] = {'kind': 'worker_exit', 'step': 2, 'rank': 0}
    path.write_text(json.dumps(raw))
    recovered, ok = run(path, tmp_path / 'runs', allow_experiment=True)
    assert ok, (recovered / 'launcher.log').read_text()
    assert validate_runs(reference, recovered)['passed']
