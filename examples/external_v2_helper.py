"""A declared local dependency for the v2 evaluation workload."""


def mix_value(value: float, salt: int) -> float:
    return (value + (salt % 17) / 31.0) % 1.0
