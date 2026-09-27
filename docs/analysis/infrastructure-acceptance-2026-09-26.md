# 后续基础设施闭环契约

## CUDA 实卡

需要两张可见 CUDA GPU、同型号／驱动／CUDA／PyTorch 环境、可用 NCCL 与本地存储空间。执行 `configs/cuda_ddp.yaml`、`configs/cuda_fsdp2.yaml` 的 acceptance，再执行 `tests/test_gpu_acceptance.py`（DDP FP32/BF16/FP16、FSDP2 FP32）。保留全部失败与负面控制，比较同环境精确状态，不跨 CPU/GPU 比较哈希。FP16 还需实际非有限梯度／scaler 与恢复矩阵，记录各 rank 显存与资源退出。

## 多节点

选择一个调度器及明确节点代理后实施。单一控制服务拥有重试决策；每次 attempt 获取单调 fencing epoch。所有节点确认旧组退出／被隔离后才启动新 epoch。检查点写入和提交携带 epoch；存储提交服务拒绝旧 epoch。调度器负责控制服务崩溃恢复和节点失联隔离。SQLite 保留为单服务本地索引，不能放到共享网络目录作为分布式锁。

验收：两个固定节点；任一 worker/node 退出、网络分区、控制者重启、旧控制者复活、保存中断；确认不存在重叠有效 attempt，样本／状态与参考一致。拓扑变化另立数据/RNG/批量契约。

## 对象存储

选择实际服务和认证方式后实施。每次保存使用不可变 generation 键；payload 与 rank 状态全部上传并校验后生成 manifest；通过服务提供的条件写入提交 generation。读取只认已提交 manifest，并验证完整对象集合和哈希。索引写入失败不改变对象 transaction 的有效性。Retention 根据已提交版本与加载 lease 删除，支持删除中断重试。不能把 POSIX rename 当作多对象事务。

验收：单机远程保存/加载先通过；再组合多节点。覆盖上传中断、条件写冲突、旧 epoch 提交、对象缺失／损坏、部分删除、凭据过期、限流、网络失败。需要实际服务证据后才能关闭此门槛。

## Linux 耐久性与性能

明确本地文件系统、挂载选项与存储介质；验证文件/metadata/manifest/marker/目录的持久化顺序，覆盖 ENOSPC/EIO/只读和每个发布边界退出。独立 crash/reboot rig 测试主机断电恢复；文件 fsync 调用测试不替代它。

性能使用专用稳定主机，两次独立批次，每模式初始 12 次测量与预热，六排列平衡。记录载荷、I/O、RSS、GPU 峰值、回滚与 RTO，保留原始测量。根据预先固定的有意义差异和波动再决定样本量；不按满意结果提前停止。
