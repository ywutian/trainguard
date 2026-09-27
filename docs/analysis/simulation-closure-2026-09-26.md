# 训练恢复的本机模拟闭环与升级边界

日期：2026-09-26。基线：0.2.2；本轮实现：0.3.0。目标是让固定拓扑训练恢复在本机形成可执行、可复核的闭环，并把远程存储、节点接管与断电语义先变成会失败的协议测试。**本机模拟通过只证明其所执行的代码路径和模型假设，不能替代对应基础设施的实测。**

## 同类系统对照

| 系统 | 官方资料中的恢复边界 | 本项目采用的验收要求 |
| --- | --- | --- |
| [PyTorch DCP](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)、[异步保存](https://docs.pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html) | 多 rank 协同保存；异步暂存与上传是不同完成阶段，应用负责等待与管理在途请求 | 同一真实 DCP 路径测每 rank 完成、Future 取消／超时、元数据／载荷验证、提交前后的硬退出 |
| [TorchSnapshot](https://meta-pytorch.org/torchsnapshot/main/getting_started.html) | 应用状态包含模型、优化器、进度与 RNG；异步快照和有限的拓扑重分片有明确条件 | 完整状态与数据游标必须由参考运行、恢复运行和故意遗漏状态的负控共同证明；当前固定拓扑不推断弹性恢复 |
| [Ray Train 故障恢复](https://docs.ray.io/en/latest/train/user-guides/fault-tolerance.html)、[检查点](https://docs.ray.io/en/latest/train/user-guides/checkpoints.html) | 区分 worker、node 与 driver 恢复；多节点依赖共享持久存储；已上传提交和[用户定义的模型验证](https://docs.ray.io/en/latest/train/user-guides/asynchronous-validation.html)是不同状态 | 分别测 worker 与控制者退出；将“已发布”“文件完整”“训练结果正确”设为三个独立判据，不能用一个状态代替另一个 |
| [DeepSpeed](https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html) | 每个 rank 都必须参加保存，恢复由 tag／latest 指向检查点 | 验收各 rank 在同一步保存并加载，不能只信最新目录名称；该资料没有给出可直接沿用的目录级原子性保证 |
| [Megatron Core](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/core/dist_checkpointing.html)、[NeMo 韧性](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/resiliency.html) | 分片、异步、跨并行配置及本地／全局检查点各有独立约束 | 保留完整哈希和损坏降级；把本地版本选择与远程副本／节点故障分开验收 |
| [TorchFT](https://meta-pytorch.org/torchft/manager.html) | step 级 quorum 与重组需要管理器自身状态 | 作为更广的故障模型参考，不把固定 world size 恢复误称为同等能力 |

[S3 官方一致性说明](https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html)保证单键原子更新及成功写后的强读一致性，但没有跨键原子提交。[条件写](https://docs.aws.amazon.com/AmazonS3/latest/userguide/conditional-writes.html)可以处理同键竞争，仍需应对 412、409、multipart 与结果未知的重试。[Linux `fsync` 手册](https://man7.org/linux/man-pages/man2/fsync.2.html)说明文件同步不自动保证目录项同步。这里的推论是：远程 generation 必须先逐件验真，再通过单个条件提交点发布；本地 `COMMITTED` 之前，样本日志与包含它们的目录也必须完成同步。

## 闭环判定方式

每个正面案例都要留下故障注入点、旧／新候选、控制者选择、每个 rank 的实际加载、有效样本和批次序列、最终模型／优化器／调度器状态、退出原因及原始日志。独立无故障运行是数值与样本的参照。负控故意遗漏 RNG、优化器或数据游标：它必须完成恢复并被精确比较检出差异，不能以训练报错冒充负控通过。

本机矩阵有三层，证据级别分别记录：

1. **真实执行路径**：双 rank CPU DDP、DCP 保存／载入、控制器退出续跑与十案例参考／故障／负控；同机两个独立启动器重启实际 Gloo 进程组。这里运行的是训练器本身。
2. **真实进程故障切点**：子进程在 payload、rank sidecar、manifest、`COMMITTED` 和索引更新之间以 `os._exit` 硬退出，父进程用正式候选选择器及 DCP 载入判断。该层不模拟断电丢失的内核缓存。
3. **确定性协议模型及训练桥接**：条件写对象存储、不可变 generation、完整载荷清单、丢响应读回、候选损坏降级、隔离确认后的单调 epoch 接管。桥接测试将真实 DCP 检查点的全部字节上传到该模型，损坏较新版本后回退、下载，并经正式本地校验器与 DCP 载入验证；另一项双 rank 测试在本地检查点全部丢失后，从模型下载旧版并交给正式控制器续跑，对照完整训练结果。模型仍假定单键条件写的线性化、强一致读／列举及独立隔离事实；尚未成为生产训练器的远程后端，也未连接真实存储或调度器。

| 故障或不变量 | 执行入口 | 独立判据 |
| --- | --- | --- |
| worker 退出、保存中断、损坏、挂起和故意遗漏状态 | `acceptance` 十案例、`tests/test_recovery_integration.py` | 指定故障事件一次、恢复一次、两次 attempt 状态；正确案例最终状态及每 rank 有效样本完全相同，负控只出现预期差异 |
| 多启动器组退出和重建 | `tests/test_local_topology_simulation.py` | 两个独立本机 `torchrun` 启动器真实组成双 rank Gloo 组，恢复后比较最终摘要和逐 rank 样本顺序 |
| 提交切点硬退出 | `tests/test_checkpoint_crash_matrix.py` | 子进程真实退出；正式选点器和 DCP loader 判定旧版或新版，SQLite 仍可滞后 |
| 自洽哈希但 DCP 不可载入 | `tests/test_checkpoint.py`、`tests/test_simulated_failure_boundaries.py` | 坏最新版被排除，旧版保留并被双 rank 恢复，最终状态与参考一致 |
| async Future 取消、超时和证据发布顺序 | `tests/test_async_upgrade.py`、`tests/test_simulated_failure_boundaries.py` | 每 rank 协同退出或先同步事件再提交，不能以单 rank 成功作总体成功 |
| 条件发布、丢响应、epoch 接管和本地检查点丢失 | `tests/test_remote_protocol.py`、`tests/test_remote_dcp_roundtrip.py`、`tests/test_remote_training_restore.py` | 条件头不回退、旧 epoch 被拒绝；真实 DCP 字节下载后由正式控制器完成双 rank 续跑和精确比较 |

## 本轮发现并修复的实际失效路径

- **初始化窗口**：`run.json` 已落盘但 SQLite `runs` 行未插入时，原续跑触发外键错误。现在只在没有任何训练／检查点／控制事件的初始状态，按配置与运行身份重建行；训练开始后的索引丢失会拒绝续跑。
- **候选可加载性**：原先自洽的 manifest 哈希仍可指向坏 DCP 元数据或错误 shard。现在恢复选点与保留删除共同检查元数据结构、引用范围及每个 payload 片段的实际解码；坏最新版不能反复占用重试，也不能挤掉较旧的可加载回退。正常保存的提交阶段保持较轻的哈希路径；恢复时的解码会增加读取时间与内存峰值，需在真实规模另测。
- **异步协调**：取消的 Future 曾在集体通信前抛出，使另一 rank 超时。现在取消／异常统一汇总后所有 rank 同因退出；已完成但回调未记录时间且超过 deadline 的状态保守判为超时。
- **证据耐久顺序**：原先检查点可持久发布，相关步骤／批次日志却未 `fsync`，模拟日志尾部丢失后健康恢复会被审计拒绝。现在每个 rank 在发布前同步日志文件和目录，协同确认后才由 rank 0 提交；完成事件在训练函数返回前也同步。真实掉电行为仍需指定文件系统和主机实测。
- **故障归因**：十案例验收原先只要求至少一次恢复。现在每案须有恰好一次目标注入、一次恢复、两次预期状态的 attempt，再满足精确比较或负控差异。

## 尚未获得的实测能力

本机没有两张真实 CUDA 卡、两台机器、对象服务或受控断电环境。当前训练控制器仍使用本地 `flock`、`ps`、SQLite 和 `--nnodes=1`；对象协议与 fencing 是可执行设计模型，并非训练器已切换到远程运行。要把这些能力列为“已支持”，须接入真实存储适配器和独立隔离权威，冻结拓扑与身份规则，再在目标环境重跑相同参考／故障／负控／回退矩阵。跨版本迁移、任意第三方训练状态和弹性 world size 也不属于 0.3.0 的固定拓扑契约。

对象模型的修订标识必须在对象内容先改变、再变回旧内容后仍有不同值；不能直接将可能由内容决定的 ETag 当作这种修订标识。接入真实对象服务时还需验证条件写的竞争结果、结果未知时的读回、列举一致性、多段上传和旧节点隔离。桥接测试证明了真实 DCP 字节能穿过该协议模型、双 rank 训练能从下载版本恢复，不证明这些外部服务条件已经成立。

执行结果与原始路径见[本轮验收报告](../experiments/simulation-closure-2026-09-26.md)。
