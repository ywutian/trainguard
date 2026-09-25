import pytest
import torch

from trainguard.data import sample_ids_for_step, token_batch


def test_rank_samples_are_disjoint_and_replayable() -> None:
    first = sample_ids_for_step(3, rank=0, world_size=2, batch_size=2)
    second = sample_ids_for_step(3, rank=1, world_size=2, batch_size=2)
    assert first == [12, 13]
    assert second == [14, 15]
    assert set(first).isdisjoint(second)
    assert sample_ids_for_step(3, 0, 2, 2) == first


def test_token_batch_is_stable_for_sample_id() -> None:
    first = token_batch([7], sequence_length=8, vocab_size=32, seed=42)
    again = token_batch([7], sequence_length=8, vocab_size=32, seed=42)
    different = token_batch([8], sequence_length=8, vocab_size=32, seed=42)
    assert first.shape == (1, 9)
    assert torch.equal(first, again)
    assert not torch.equal(first, different)


def test_invalid_rank_is_rejected() -> None:
    with pytest.raises(ValueError):
        sample_ids_for_step(0, rank=2, world_size=2, batch_size=1)
