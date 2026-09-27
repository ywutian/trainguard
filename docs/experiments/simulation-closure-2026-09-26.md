# 0.3.0 本机模拟闭环验收

日期：2026-09-26（太平洋时间；原始日志使用 UTC）。结果：**本机可验证范围全部通过**。源码指纹：`d0334dd8bacbc8743b054d6ea3af7a28e40fe0d3a31a85ecd2bdeb7de3d5934e`。

## 环境与执行

- macOS 26.5.1 / arm64；Python 3.12.12；PyTorch 2.14.0；双 rank CPU/Gloo。此机器没有可用于验收的 CUDA 设备或独立多主机环境。
- 入口：`uv sync --locked --group dev`，随后 `uv run python scripts/run_simulation_closure.py --output-root runs/simulation-closure`。
- 本次完整原始目录：`/Users/yitianwu/Documents/TrainGuard-recovery-validation/runs/simulation-closure/simulation-03dd217fc632`（约 293 MB）。其中 `test-artifacts/` 保存每项测试的原始故障与恢复目录，`acceptance/` 保存十案例运行、SQLite、检查点和逐 rank 日志。
- 可提交的轻量证据在[本报告附带目录](evidence/simulation-closure-2026-09-26/)；[证据哈希清单](evidence/simulation-closure-2026-09-26/evidence-manifest.json)记录每个文件的大小与 SHA-256。

## 门槛结果

| 门槛 | 结果 | 证据 |
| --- | --- | --- |
| 静态检查 | 通过 | [输出](evidence/simulation-closure-2026-09-26/static.txt) |
| 完整测试 | 138 通过、4 项 CUDA 条件跳过；16 条单进程 DCP 提示 | [测试输出](evidence/simulation-closure-2026-09-26/pytest.txt)、[JUnit](evidence/simulation-closure-2026-09-26/pytest.xml) |
| 独立 CPU 验收 | 10/10 通过；每案目标故障一次、恢复一次、预期 attempt 状态成立 | [逐案记录](evidence/simulation-closure-2026-09-26/acceptance.json)、[摘要](evidence/simulation-closure-2026-09-26/acceptance-report.md) |
| 打包 | wheel 与 sdist 构建通过 | [构建输出](evidence/simulation-closure-2026-09-26/build.txt) |
| 安装包身份 | 0.3.0 与源码指纹一致 | [校验输出](evidence/simulation-closure-2026-09-26/wheel.txt) |

[总门槛记录](evidence/simulation-closure-2026-09-26/gate-result.json)显示五个退出码均为 0，状态为 `SUCCEEDED`。完整原始测试目录保留了提交切点、rank 事件、候选和最终结果；轻量证据中的[两个坏元数据回退事实](evidence/simulation-closure-2026-09-26/fallback-1-result.json)与[对应故障定义](evidence/simulation-closure-2026-09-26/fallback-1-fault.json)可独立抽查，另一组见同目录的 `fallback-2-*`。

七个正常案例覆盖同步／异步 worker 退出、保存中断与损坏，以及同步挂起。恢复后的模型、优化器、调度器、步数和每 rank 有效样本序列与无故障参考完全一致。三个负控都完成训练并实际恢复：遗漏 RNG 或优化器导致最终模型／优化器摘要差异；遗漏数据游标另导致两个 rank 的有效样本与批次序列差异。缺日志、未发生故障或无恢复均不能算通过。

远程协议桥接先用真实 DCP 字节验证损坏最新版后的旧版下载与加载；更完整的双 rank 测试在本地检查点目录丢失后，从对象协议模型下载第 1 步候选，交给正式控制器续跑至第 4 步。[桥接结果](evidence/simulation-closure-2026-09-26/remote-training-restore.json)显示模型、优化器、调度器摘要与参考一致，逐 rank 有效样本比较通过。它运行了训练器和恢复控制器，但对象服务、条件写、隔离权威仍是本机模型。

第一次全套运行因测试配置把冷启动进度超时设为 5 秒，在整套 CPU 负载下过早终止两个运行，留下[失败结果](evidence/simulation-closure-2026-09-26/initial-failure-result.json)和[失败堆栈](evidence/simulation-closure-2026-09-26/initial-failure-tests.txt)；提交版堆栈只去掉了行尾空格，原始文件仍在首次运行目录。将该测试阈值改为 20 秒后，两项针对性复测通过，并完整重跑了总门槛；最终结果来自上方独立的新目录，没有拼接旧运行的成功记录。

## 结论与未验证边界

本机模拟已把固定拓扑 CPU 训练、真实双进程 DCP 保存与恢复、多启动器重建、提交切点硬退出、损坏回退、负控、远端字节桥接和可追溯证据连成可复跑的闭环。[同类系统对照与故障矩阵](../analysis/simulation-closure-2026-09-26.md)记录了采用这些判据的依据。

真实 CUDA DDP/FSDP2、跨主机调度和隔离、实际对象服务的条件写／多段上传／丢响应语义，以及指定 Linux 文件系统的断电与 I/O 错误，仍须在对应环境执行独立验收。当前本机结果不能替代这些外部条件，也不声明弹性 world size 或任意第三方训练状态支持。远程 Linux 工作流已配置，但尚无该工作流的远端执行记录。
