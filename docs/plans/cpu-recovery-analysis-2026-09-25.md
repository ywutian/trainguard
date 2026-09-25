# CPU 恢复与实验执行分析

## 目标和现状

现有路线图的 CPU 基线已完成：固定两进程 Gloo 训练、完整检查点、整组恢复、故障注入、精确校验和同步／异步保存实验。本轮先补强进程归属检查与计时口径，再决定是否扩大实验规模。当前机器运行 PyTorch 2.10.0，具有 MPS，但没有 CUDA 设备；因此本计划的验收范围是单机固定规模 CPU。

## 查阅资料后的技术判断

1. [PyTorch 2.10 分布式检查点文档](https://docs.pytorch.org/docs/2.10/distributed.checkpoint.html)说明 DCP 按 rank 写入文件、加载时需要预先分配目标状态，并且不同 PyTorch 版本之间没有检查点向后兼容保证。现有实现固定 PyTorch 2.10.0、在模型和优化器构造后加载，版本变化必须单独验证。
2. [异步检查点教程](https://docs.pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html)建议同一时间最多保留一个异步保存请求。当前实现遵守这个约束，并等待保存完成后才发布应用层提交标记；保存完成时间与训练可能重叠，不能把各阶段时间直接相加。
3. [torchrun 故障语义](https://docs.pytorch.org/docs/2.10/elastic/run.html)以整组工作进程为恢复单位。当前控制器禁用 torchrun 内部重试，由自身选择已提交的检查点并控制重启次数。控制器退出时，启动器及其工作进程都必须纳入归属检查。
4. [PyTorch 可复现性说明](https://docs.pytorch.org/docs/2.10/notes/randomness.html)明确跨版本、跨平台、CPU 与 GPU 的逐位一致性不能普遍保证。因此 `atol=0, rtol=0` 仅用于本机、同版本、同设备的固定工作负载。
5. [PyTorch 基准测试教程](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html)强调预热、重复测量和报告波动。原四步实验的训练窗口约 0.018 秒，总耗时约 1.9 秒，主要测到进程启动；扩展实验单独记录训练窗口，并保留全部原始值。

## 本轮执行和验收

- [x] 在 `tests/test_recovery_integration.py` 复现“启动器已创建、PID 未入库、工作进程未派生”窗口；测试先失败，再修正 `src/trainguard/controller.py` 的启动器归属识别，恢复时拒绝重叠进程。
- [x] 将异步保存中断纳入集成故障矩阵，验证回滚后与未中断运行的最终状态相同。
- [x] 在 `src/trainguard/trainer.py` 记录单个工作进程的训练窗口，在 `src/trainguard/benchmark.py` 同时统计总耗时与训练窗口。
- [x] 用 `configs/cpu_benchmark.yaml` 执行 3000 步、每 100 步保存、三种模式各 5 次的实验。15 次全部通过精确校验，数据见[扩展实验报告](../experiments/cpu-extended-2026-09-25.md)。各模式耗时范围重叠，不得据此声称异步保存稳定更快。
- [x] 完整检查：`uv run ruff check .` 和 `uv run pytest -q`；35 项测试通过。

## 后续执行顺序

### 1. 加强校验器对事件日志的独立检查

修改 `src/trainguard/validation.py`，让 `_effective_samples` 显式报告同一次尝试内重复的步骤事件、非整数样本 ID、缺失的 rank 日志和非法步骤顺序，不再用字典覆盖重复记录。新增 `tests/test_validation.py`，分别构造每种异常，并断言 `validate_runs` 返回失败及明确原因。保留已有故障恢复和负面实验作为回归门槛。

### 2. 覆盖控制器退出点

在 `tests/test_recovery_integration.py` 分别覆盖检查点提交后入库前、尝试完成后顶层状态发布前，以及启动器退出但工作进程仍存活。每个场景都验收：不并发启动第二组；旧进程退出后只选择校验通过的检查点；完成的尝试不会额外消耗重试次数。现有模拟退出测试已经覆盖其中一部分；新增测试应优先针对尚未覆盖的文件／数据库交界点。

### 3. 提高性能结论的可信度

在 `src/trainguard/benchmark.py` 增加可选预热轮次，并把预热记录与正式轮次分开保存。使用专用、负载稳定的机器，对至少两个模型／检查点大小重复测试，保留每次运行、训练窗口、检查点字节数和保存阶段耗时。以重叠范围、重复间波动和正确性结果共同解释数据；若噪声接近模式差异，只报告观察值，不作速度优劣结论。

### 4. 扩展前的门槛

只有在 CPU 的事件日志校验、进程退出矩阵和可重复实验稳定后，才为 CUDA／FSDP 制定独立方案。该方案必须在实际多 GPU 环境验证设备绑定、CUDA RNG、NCCL 训练通信、用于异步 DCP 的 CPU 通信后端，以及不同拓扑下的状态加载；本机 MPS 结果不能代替这些验收。
