# 恢复验证产品的市场与收费证据

核查日期：2026-09-26；2026-09-27 复核恢复能力资料。本页把官方产品页面可证实的事实与本项目待验证的商业假设分开；动态价格和条款在报价当天重查。

## 竞争边界

| 官方来源 | 已证实的包装或能力 | 对本项目的实际含义 |
| --- | --- | --- |
| [Anyscale 定价](https://www.anyscale.com/pricing) | 托管和客户自带云并列，按使用量与承诺合约收费，企业支持有单独层级。 | 客户环境部署、支持和计费可分层，但不能据此推断本项目标价或利润。 |
| [AWS SageMaker AI 定价](https://aws.amazon.com/sagemaker/ai/pricing/)与[HyperPod 用量报告](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-usage-reporting.html) | 资源按实际使用计费，相关存储/集群费用另计，已有 GPU 用量报告。 | 报价和 ROI 取客户自身账单；基础遥测不是独有卖点。 |
| [HPE ML Development Environment QuickSpecs](https://www.hpe.com/us/en/collaterals/collateral.a50006978enw.html) | 企业训练软件按 GPU 和多年订阅期限包装。 | 只说明企业采购可以订阅，公开材料不提供本项目可直接使用的单功能价格。 |
| [NVIDIA AI Enterprise 价格与许可](https://docs.nvidia.com/ai-enterprise/planning-resource/licensing-guide/latest/pricing.html) | 面向完整企业软件栈及支持的 GPU 许可/订阅。 | 不能把其整套产品标价拿来作为单一恢复验证组件的价格锚。 |
| [NVIDIA Resiliency Extension](https://nvidia.github.io/nvidia-resiliency-ext/) | 官方 0.7.0 文档列出挂起检测、作业内重启、异步及本地检查点、慢 rank 检测和故障归因。 | 单卖“自动重启”或“故障检测”缺乏可证实差异；需证明客户接入、负控拒绝、恢复正确性和可审计证据的额外价值。 |
| [Ray Train 故障恢复](https://docs.ray.io/en/latest/train/user-guides/fault-tolerance.html)与[持久存储](https://docs.ray.io/en/latest/train/user-guides/persistent-storage.html) | 官方文档区分工作进程、节点和驱动进程故障；恢复依赖训练代码保存并加载检查点，多节点要求外部持久共享存储。 | 现有训练平台已经覆盖多层重试与持久化；本项目必须在相同作业和故障条件下证明正确性审计的独立价值。 |
| [AWS HyperPod checkpointless 真实训练验证](https://aws.amazon.com/blogs/machine-learning/checkpointless-training-on-amazon-sagemaker-hyperpod-production-scale-training-with-faster-fault-recovery/) | 官方已公开真实 Llama-3 70B 训练的逐步 loss 位级对照、checksum 和多规模恢复时间。 | 不能宣称逐步等价是业界首创；必须同客户现有方案在同环境对照。 |

首个可验证的定位是假设：**为客户现有 PyTorch 长训练提供独立的恢复正确性审计和可复跑的故障验收，并在通过后提供受限环境中的执行与支持**。这不是当前版本已经证明的优势。买方假设是掌管跨节点训练与 GPU 预算的 ML 平台负责人；触发事件可能是恢复事故、训练平台迁移或使用可中断资源。询问近 90–180 天真实故障、作业长度、人工处置、成本与隔离演练许可，以决定是否进入试点。访谈记录与签约结果才是需求证据。

## 单一受控报价与价值核算

第一笔交易可先为**有限付费接入研究**：按一个模型/训练代码库、一个固定拓扑、一个调度器和对象服务、一个隔离测试集群的固定范围服务费报价，明确当前尚未实现的适配/存储/接管能力和补救、退出边界。将客户承担的 GPU、网络与存储账单单列。完成标准是实际交付的接入研究、演练与证据；结果可能 `PASS`、`FAIL` 或 `BLOCKED`，阻断须归因，供应商未实现约定能力不能默示通过或自动触发付款。发现可复现缺陷有价值，但不等于训练已达生产条件。额外适配和演练预算须走书面变更单。客户环境技术保护范围仅在冻结矩阵真实通过后确认；广泛商用另经完整放行。

报价底线由人员集成、演练、报告、支持、维护与商业成本构成；价格上限须由客户在明确观察期间、真实故障频率及实际费率下保守测得的净价值和采购意愿校准。配对故障注入账本只是测试样本的条件成本差额，不能直接充当年度净收益或报价上限。若成本底线高于该客户可证明的净收益，不能靠编造故障率或竞品价格完成报价。现有 MIT 代码权利不因服务费而变窄；年度收入假设来自持续复测、客户环境适配和有人员承接的支持。纯席位、全群 GPU 总量或未验证的受保护时长现在不适合作为计费依据。

一个试点应同时报告：同故障的客户原方案与受测方案的有效重跑 GPU 小时、闲置恢复时长、工程工时，以及新增保存计算、对象存储、演练和支持费用。相同时间段只计一次。零自然故障期间只报告观察到的开销和故障注入条件效果；历史故障率乘以条件改善仅是情景预测。使用 [配对账本模板](pilot-ledger-template.json) 和 [试点订单与验收附件](customer-pilot-template.md) 保留费率来源、原始事件、冻结矩阵和双方签收。

季度复核按客户真实事故正确率、RPO/RTO、升级后的复测覆盖、负面控制拒绝、支持工时及总成本进行。客户若已有 Ray、HyperPod、NVIDIA 或其他恢复方案，应将**现有方案**作为基线，不以完全从零恢复作为对照。未经客户授权，不公开名称、报价、案例或节省比例。

客观性能与经济宣称需要可复核证据；[美国 FTC 的广告证据政策](https://www.ftc.gov/legal-library/browse/ftc-policy-statement-regarding-advertising-substantiation) 可作为美国市场宣传审查参考。适用的法律、合同与行业规则须按实际交易另行审阅。
