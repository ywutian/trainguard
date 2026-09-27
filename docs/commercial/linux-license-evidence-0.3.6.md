# 历史 Linux GPU 依赖许可证据：CUDA 元包与 cuSPARSELt

记录日期：2026-09-27。范围为首次 0.3.6 Linux 候选锁文件中的 `cuda-toolkit==13.0.3.0` 和 `nvidia-cusparselt-cu13==0.8.1`，历史提交为 `9b2d7e24eca3780a4e200eba6f2db5a3540e4aa1`。本记录核对了指定版本的发布件、元数据和厂商条款来源；它不是许可兼容性、再分发权利或生产放行结论。当前 Linux CPU/Gloo 候选锁定 `torch==2.14.0+cpu`，已移除这些 GPU 依赖。下述发布件摘要只追溯首次候选，不代表当前锁文件。

## 当前门槛

首次候选的托管 Linux Python 3.11 和 3.12 两条供应链扫描均为 `BLOCKED`。这两个依赖在由 CycloneDX 已声明元数据生成的许可证清单中没有可核验的许可项；`scripts/supply_chain.py` 对第三方空许可保持拒绝。当前 CPU 候选移除 GPU 依赖不更改历史失败扫描、原始收据或发布门槛。只有在同一最终 CPU 候选 wheel、锁文件和目标 Linux 平台上重新生成并核验两条安装及供应链扫描后，才能重新评估受限评价门槛。GPU 交付仍须独立完成许可审阅和实卡验收。

## 指定版本的原始证据

| 锁定组件与发布件 | 包内及元数据所见 | 厂商条款来源与适用边界 |
| --- | --- | --- |
| `cuda-toolkit==13.0.3.0`，通用 wheel `cuda_toolkit-13.0.3.0-py2.py3-none-any.whl`，SHA-256 `d693caaa261214ddd7dbb60d68e71cbed884e68c2be7509778f3051da0b91c3f` | 已读取完整 2,512 字节 wheel 并独立重算上述摘要，与[首次候选锁文件](https://github.com/ywutian/trainguard/blob/9b2d7e24eca3780a4e200eba6f2db5a3540e4aa1/uv.lock)及 [PyPI 该版本文件记录](https://pypi.org/project/cuda-toolkit/13.0.3/) 一致。wheel 仅含 `METADATA`、`WHEEL`、`RECORD`；`METADATA` 没有 `License`、`License-Expression` 或 `License-File`，包内也没有许可证文本。PyPI 将其描述为仅安装其他包的元包。空许可证元数据是真实缺项，不应填成 MIT 或从下游组件推断许可。 | [NVIDIA CUDA Toolkit 13.0.3 归档 EULA](https://docs.nvidia.com/cuda/archive/13.0.3/eula/index.html) 是厂商提供的对应工具包条款。[归档 PDF](https://docs.nvidia.com/cuda/archive/13.0.3/pdf/EULA.pdf) 为 228,376 字节，读取时 SHA-256 为 `c365e3bfe9ed70d7fb7dbacc4fdd4b4a15fb951be5440f0f6bd95e3c935e93ef`。它是外部条款证据，不会补写元包的 wheel 元数据，也不自动授予分发所有 CUDA 组件的权利。 |
| `nvidia-cusparselt-cu13==0.8.1`，Linux x86_64 wheel `nvidia_cusparselt_cu13-0.8.1-py3-none-manylinux2014_x86_64.whl`，SHA-256 `786ce87568c303fadb5afcc7102d454cd3040d75f6f8626f5db460d1871f4dd0` | 已从首次候选锁文件指定的 [PyPI 发布件](https://pypi.org/project/nvidia-cusparselt-cu13/0.8.1/)下载完整 170,148,586 字节 wheel，独立重算整包 SHA-256，与旧锁文件和 PyPI 发布记录一致。包内 `METADATA` 的旧式 `License` 字段是 `NVIDIA Proprietary Software`，没有 `License-Expression` 或 `License-File` 字段；`nvidia/cusparselt/LICENSE.txt` 为 17,948 字节，SHA-256 为 `e8d158885a681b95ec7a6fc06dd8d4a52989f374cb1380c8a4c8fb27fd3d5d5e`。 | [NVIDIA cuSPARSELt 0.8.1 再分发清单](https://developer.download.nvidia.com/compute/cusparselt/redist/redistrib_0.8.1.json) 指定 `libcusparse_lt/LICENSE.txt`；其中 `release_label` 为 `0.8.1`，库版本为 `0.8.1.1`。[厂商 LICENSE 原文](https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/LICENSE.txt) 与 wheel 内文本逐字节一致。条款含 cuSPARSELt 专项补充及分发限制，应作为专有许可证文本审阅，不应伪写为 SPDX 开源许可标识。 |

PyPI 版本页、厂商归档和发布件分别证明不同事实：PyPI 给出发布件身份，wheel 元数据与包内文件给出该包实际携带的声明，厂商文档给出可审阅的条款。厂商网页或 PDF 可更新；上面的内容摘要用于发现证据变化，变化后必须重新审阅。

本机完整发布件复核进一步核对了 cuSPARSELt wheel 的 `nvidia_cusparselt_cu13-0.8.1.dist-info/METADATA`（SHA-256 `353669bafb80cbf22f1324121ac05049d36a6b0fb44b1fed3fab00b513231171`，元数据版本 `2.4`）、包内 `LICENSE.txt` 及两者在 wheel `RECORD` 中记录的 SHA-256 与字节数。包内许可文本与上述厂商原文同为 17,948 字节，SHA-256 相同且逐字节比较相等。完整 wheel 摘要已核验，但既有两条托管 Linux 扫描属于旧候选；本机复核不替代最终候选在目标环境中实际选中的 wheel、安装环境和供应链门槛复测。

## 当前 CPU 候选与未来 GPU 交付

当前 CPU 候选须在 Linux Python 3.11、3.12 各自从最终锁文件同步到干净环境，证明安装的 PyTorch 为 `2.14.0+cpu`、`torch.version.cuda` 为空，且安装清单、SBOM、许可证清单均无 CUDA、NVIDIA、Triton 组件。再对实际安装版本执行 OSV 已知漏洞扫描，保留原始输出、缺声明与漏扫阻断规则及两条独立收据。CPU 锁文件变化使首次候选扫描和本记录的 GPU 发布件摘要都不能替代新收据；客户环境与商业门槛仍需独立验收。

若未来制作 GPU 交付候选，仍需完成以下原有许可工作：

1. 在未来 GPU 候选的 Linux Python 3.11、3.12 干净环境分别按该候选锁文件安装，保留已安装包清单、SBOM、许可证清单、原始漏洞扫描输出和工具版本。记录实际选中的平台 wheel 与完整 SHA-256；确认 cuSPARSELt 的目标环境发布件与上面已完整核验的 x86_64 wheel 身份一致，并核对包内 `METADATA`、`LICENSE.txt` 与摘要。其他架构须单独核对其实际 wheel，不沿用 x86_64 的文件摘要。
2. 对空许可条目使用单独的人工审定证据记录，精确绑定包名、版本、平台、wheel SHA-256、厂商 URL、条款内容 SHA-256、审阅者、审阅时间和适用范围。记录应明确区分“包内声明”“包内许可文件”和“外部厂商条款”；不能改写原始 CycloneDX 或漏洞扫描输出。缺任一绑定或出现新的空许可组件时继续 `BLOCKED`。
3. 在客户安装或制作离线 wheelhouse 前，审阅实际交付形态、目标平台、NVIDIA 各组件随附通知、客户取得与接受条款的流程，以及厂商条款中的使用和分发限制。当前交付包不含依赖 wheelhouse；若改为打包、镜像或直接再分发 NVIDIA 二进制，须针对该形态单独取得法律和采购审阅结果。许可证据齐全只表示来源与条款可追溯，不表示任何商业分发已获准。
4. 若依赖版本、wheel 摘要、目标平台、厂商条款或最终候选身份变化，旧审阅结果不迁移。保留原阻断记录，对新身份重跑相应扫描和门槛；生产发布仍需独立的客户环境、安全运营和商业签收证据。
