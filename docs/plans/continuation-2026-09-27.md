# TrainGuard 候选交接与后续验收计划

日期：2026-09-27。范围：0.3.6 的 Linux CPU/Gloo 受限评价候选。**生产发布仍为 `BLOCKED`。**

## 1. 当前交接状态

- 工作分支：`feature/product-readiness`；继续使用[现有 PR #1](https://github.com/ywutian/trainguard/pull/1)，不要为同一仓库的这批工作另开个人 PR。
- 最新代码修复提交：`f2cfefd606f2310277d3ff0ed2e38b9e477c051f`。本文件只作交接记录，不改变执行输入；开始新运行时仍以 `git rev-parse HEAD` 冻结执行提交。
- 当前源码 SHA-256：`dabe10fa40fd194a9be069e1569f3f8d2fae82aa9976845903c3d559b1044ee6`；执行输入 SHA-256：`61fece3ba23a5f512c98fea0ebde9826b183c8af0390a8669707260d48bd2078`。后续改动任一执行输入都须重新计算并完整复测。
- 已通过的定向验证：交付包、版本门禁、运行身份和模拟门禁相关 72 项测试；`ruff check .` 通过。原生 Ubuntu 24.04 的 Python 3.11/3.12 启动身份、运行身份测试及合成训练冒烟均通过，[诊断运行](https://github.com/ywutian/trainguard/actions/runs/36340001577)只覆盖启动链接修复的代码树，不能替代本候选完整托管门禁。
- [r2 本机证据](../commercial/evidence/local-0.3.6-r2/local-validation.json)属于旧执行提交 `8a37796967dffc1b1bad0bdd3ded94fa01fae77a`，其 512 项测试、10 项 CPU 故障案例和八项门禁不能转用于当前候选。[旧托管运行](https://github.com/ywutian/trainguard/actions/runs/36337374496)失败；Python 3.12 遇到 Ubuntu 标准启动链接误拒，Python 3.11 达到当时 30 分钟整套测试期限。这些失败记录保留。
- `docs/commercial/release-gates.json` 目前绑定 r2；对当前源码应判为过期并失败关闭。最终八项本机门禁和当前双版本托管门禁**尚未完成**。中断的本机试跑目录不可当作验收证据。

## 2. 下一轮按顺序执行

1. 在现有分支确认工作树干净，检查 PR 状态并遵守个人 PR 工作约定，执行 `uv sync --locked --dev`。若需修改源码、测试、配置、脚本、工作流或随包文档，先完成修改与定向验证，再冻结执行输入；不要拼接不同提交的通过记录。
2. 从仓库根目录运行完整本机门禁：

   ```bash
   uv run python scripts/run_simulation_closure.py --output-root runs/commercial-readiness-036-final
   ```

   记录输出的唯一 `simulation-*` 目录。核对 `result.json` 为 `SUCCEEDED`、八项门槛退出码全零、完整测试与固定 10 案例成立，且 `execution_commit`、`source_sha256`、`execution_inputs_sha256` 与本次冻结身份一致。任一步失败都保留原目录、修复并从头运行。
3. 成功后归档本机安全摘要：

   ```bash
   uv run python scripts/archive_local_evidence.py <成功的 simulation 目录> --label 0.3.6-r<N>
   ```

   脚本会自动加 `local-` 前缀，输出目录为 `docs/commercial/evidence/local-0.3.6-r<N>`。以归档命令输出的两个收据路径和 SHA-256 更新 `docs/commercial/release-gates.json` 的 `local_package`、`local_cpu` 及当前候选源码/执行输入摘要。只提交证据和放行清单；若改变执行输入，返回第 1 步重跑。保留原始本机目录，不把客户敏感原文上传。
4. 推送同一分支，并选用该提交的 **push** 工作流运行 ID；不要用独立诊断运行或旧 PR 重复运行冒充验收。等待 Ubuntu 24.04、Python 3.11 和 3.12 两条完整 lane 均成功，确认各自原始摘要、供应链记录和 wheel/sdist 已按工作流规则留存。若任一失败，定位原因并从第 1 步重新冻结候选。
5. 使用第 2 步产出的 wheel、sdist 与第 4 步运行 ID 执行：

   ```bash
   uv run python scripts/check_release_readiness.py --wheel <wheel> --sdist <sdist> --hosted-run-id <push 运行 ID> --report <报告路径>
   ```

   只在 `local_experiment_allowed=true` 且 `linux_customer_evaluation_allowed=true`、候选身份与所有文件摘要一致时，构建 `EVALUATION_ONLY` 包：

   ```bash
   uv run python scripts/build_delivery_bundle.py --wheel <wheel> --sdist <sdist> --readiness-report <报告路径> --output-dir <全新交付目录>
   python3 -I -S scripts/verify_delivery_bundle.py <交付目录> --expected-manifest-sha256 <构建命令输出的清单摘要>
   ```

   核查包内 `docs/commercial/` 文档、README、SECURITY、本地链接、脱敏证据、锁依赖与清单。`EVALUATION_ONLY` 只准用于客户隔离环境技术评价，不能表述为生产授权。
6. 更新同一 PR 的正文：列出最终执行提交、证据提交、两版 Linux 工作流 ID、实际测试数量、CPU 矩阵、依赖扫描时点、交付摘要、仍阻断的门槛；保持草稿状态，复核后再交给相应负责人处理。

## 3. 尚需外部资源才能结束的门槛

真实客户 DDP 作业及完整状态/数据声明；至少两台主机与目标 GPU、驱动、NCCL；选定对象服务及独立强一致 HEAD/epoch 权威；受控断电、分区、旧身份拒写和升级回滚演练；客户权限、安全运营、支持值班、合同责任、报价、付费试点与双方签收；持续运行的 SLO/事故数据。具体证据与停止条件见[产品放行计划](product-closure-2026-09-26.md)、[运营手册](../commercial/operations-runbook.md)及[试点模板](../commercial/customer-pilot-template.md)。这些门槛没有真实环境和签收前，生产状态必须保持 `BLOCKED`。
