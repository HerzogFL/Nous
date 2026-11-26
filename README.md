# Nous

Nous 是一个以苏格拉底式追问为主题的研究项目，探索大语言模型如何通过提问帮助用户把模糊的指令说清楚。

## 论文背景

用户往往难以在一条指令里说清复杂任务的所有要求。论文提出的 Nous 会主动追问尚未明确的细节，并用每轮对话对任务要求的**不确定性降低了多少**（香农熵的下降量）衡量问题的信息增益。作者用模拟对话生成训练数据，在科学示意图生成任务中评估该方法，报告了更充分的需求收集以及更高的图示质量。

Jianwen Sun 等，*Closing the Expression Gap in LLM Instructions via Socratic Questioning*，ICML 2026。[论文原文](https://proceedings.mlr.press/v306/sun26i.html)

## 目录

- `docs/` — 研究笔记与设计文档
- `src/` — 实现代码
- `experiments/` — 实验配置与结果

项目处于初始化阶段。各目录目前仅含用于 Git 跟踪的空占位文件。
