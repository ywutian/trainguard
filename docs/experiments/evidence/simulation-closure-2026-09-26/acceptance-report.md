# Recovery acceptance

Status: SUCCEEDED

Source: `d0334dd8bacbc8743b054d6ea3af7a28e40fe0d3a31a85ecd2bdeb7de3d5934e`

| Case | Result | Recovery count | Exact comparison |
| --- | --- | ---: | --- |
| sync-worker_exit | PASSED | 1 | True |
| async-worker_exit | PASSED | 1 | True |
| sync-save_interrupt | PASSED | 1 | True |
| async-save_interrupt | PASSED | 1 | True |
| sync-corrupt | PASSED | 1 | True |
| async-corrupt | PASSED | 1 | True |
| sync-hang | PASSED | 1 | True |
| omit-rng | PASSED | 1 | False |
| omit-optimizer | PASSED | 1 | False |
| omit-cursor | PASSED | 1 | False |

Negative controls pass only when training recovers and the exact comparison detects omitted state.

## Environment gates

Tested target: cpu / ddp / fixed 2 ranks.

CUDA and FSDP2 require their own successful campaign on actual CUDA devices. Multi-node scheduling, object storage transactions and power-loss durability require separate infrastructure acceptance.
