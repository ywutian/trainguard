# Linux 候选依赖许可证据：CUDA 元包与 cuSPARSELt

记录日期：2026-09-27。范围仅为 `uv.lock` 中的 `cuda-toolkit==13.0.3.0` 和 `nvidia-cusparselt-cu13==0.8.1`。本记录核对了指定版本的发布件、元数据和厂商条款来源；它不是许可兼容性、再分发权利或生产放行结论。

## 当前门槛

候选版本的托管 Linux Python 3.11 和 3.12 两条供应链扫描均保持 `BLOCKED`。这两个依赖在由 CycloneDX 已声明元数据生成的许可证清单中没有可核验的许可项；`scripts/supply_chain.py` 对第三方空许可保持拒绝。本记录不修改扫描结果、原始收据、发布门槛或交付资格。只有在同一最终候选 wheel、锁文件和目标 Linux 平台上重新生成并核验两条扫描及许可证据后，才能重新评估受限评价门槛。

## 指定版本的原始证据

| 锁定组件与发布件 | 包内及元数据所见 | 厂商条款来源与适用边界 |
| --- | --- | --- |
| `cuda-toolkit==13.0.3.0`，通用 wheel `cuda_toolkit-13.0.3.0-py2.py3-none-any.whl`，SHA-256 `d693caaa261214ddd7dbb60d68e71cbed884e68c2be7509778f3051da0b91c3f` | 已读取完整 2,512 字节 wheel 并独立重算上述摘要，与 [`uv.lock`](../../uv.lock) 及 [PyPI 该版本文件记录](https://pypi.org/project/cuda-toolkit/13.0.3/) 一致。wheel 仅含 `METADATA`、`WHEEL`、`RECORD`；`METADATA` 没有 `License`、`License-Expression` 或 `License-File`，包内也没有许可证文本。PyPI 将其描述为仅安装其他包的元包。空许可证元数据是真实缺项，不应填成 MIT 或从下游组件推断许可。 | [NVIDIA CUDA Toolkit 13.0.3 归档 EULA](https://docs.nvidia.com/cuda/archive/13.0.3/eula/index.html) 是厂商提供的对应工具包条款。[归档 PDF](https://docs.nvidia.com/cuda/archive/13.0.3/pdf/EULA.pdf) 为 228,376 字节，读取时 SHA-256 为 `c365e3bfe9ed70d7fb7dbacc4fdd4b4a15fb951be5440f0f6bd95e3c935e93ef`。它是外部条款证据，不会补写元包的 wheel 元数据，也不自动授予分发所有 CUDA 组件的权利。 |
| `nvidia-cusparselt-cu13==0.8.1`，Linux x86_64 wheel `nvidia_cusparselt_cu13-0.8.1-py3-none-manylinux2014_x86_64.whl`，[PyPI 发布 SHA-256](https://pypi.org/project/nvidia-cusparselt-cu13/0.8.1/) `786ce87568c303fadb5afcc7102d454cd3040d75f6f8626f5db460d1871f4dd0` | 该版本 `METADATA` 的 `License` 字段是 `NVIDIA Proprietary Software`，但没有 `License-Expression` 或 `License-File` 字段。通过 PyPI 发布 wheel 的 HTTP 字节范围读取 ZIP 目录及对应成员，核对 CRC 后得到 `nvidia/cusparselt/LICENSE.txt`，17,948 字节、SHA-256 `e8d158885a681b95ec7a6fc06dd8d4a52989f374cb1380c8a4c8fb27fd3d5d5e`。本次未下载完整约 170 MB wheel，故未独立重算整包 SHA-256；整包值来自 PyPI 发布记录与 `uv.lock`。 | [NVIDIA cuSPARSELt 0.8.1 再分发清单](https://developer.download.nvidia.com/compute/cusparselt/redist/redistrib_0.8.1.json) 指定 `libcusparse_lt/LICENSE.txt`；其中 `release_label` 为 `0.8.1`，库版本为 `0.8.1.1`。[厂商 LICENSE 原文](https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/LICENSE.txt) 与 wheel 内文本的 SHA-256 完全一致。条款含 cuSPARSELt 专项补充及分发限制，应作为专有许可证文本审阅，不应伪写为 SPDX 开源许可标识。 |

PyPI 版本页、厂商归档和发布件分别证明不同事实：PyPI 给出发布件身份，wheel 元数据与包内文件给出该包实际携带的声明，厂商文档给出可审阅的条款。厂商网页或 PDF 可更新；上面的内容摘要用于发现证据变化，变化后必须重新审阅。

## 关闭缺口所需的核验与审阅

1. 在最终候选 Linux Python 3.11、3.12 的干净环境分别按锁文件安装，保留已安装包清单、SBOM、许可证清单、原始漏洞扫描输出和工具版本。记录实际选中的平台 wheel 与完整 SHA-256；对 cuSPARSELt 必须读取**完整发布件**并重算 SHA-256、核对 `METADATA`、内置 `LICENSE.txt` 和该文件摘要。其他架构须单独核对其实际 wheel，不沿用 x86_64 的文件摘要。
2. 对空许可条目使用单独的人工审定证据记录，精确绑定包名、版本、平台、wheel SHA-256、厂商 URL、条款内容 SHA-256、审阅者、审阅时间和适用范围。记录应明确区分“包内声明”“包内许可文件”和“外部厂商条款”；不能改写原始 CycloneDX 或漏洞扫描输出。缺任一绑定或出现新的空许可组件时继续 `BLOCKED`。
3. 在客户安装或制作离线 wheelhouse 前，审阅实际交付形态、目标平台、NVIDIA 各组件随附通知、客户取得与接受条款的流程，以及厂商条款中的使用和分发限制。当前交付包不含依赖 wheelhouse；若改为打包、镜像或直接再分发 NVIDIA 二进制，须针对该形态单独取得法律和采购审阅结果。许可证据齐全只表示来源与条款可追溯，不表示任何商业分发已获准。
4. 若依赖版本、wheel 摘要、目标平台、厂商条款或最终候选身份变化，旧审阅结果不迁移。保留原阻断记录，对新身份重跑相应扫描和门槛；生产发布仍需独立的客户环境、安全运营和商业签收证据。
