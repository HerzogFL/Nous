# Nous

Nous 是基于 ICML 2026 论文实现的研究原型，探索大语言模型如何通过苏格拉底式追问，把用户的初步指令细化成可执行的规格。

论文将科学示意图拆成离散属性，用对话前后香农熵的差衡量信息增益。系统让模拟用户回答同一轮的多个候选问题，再用获得的信息量训练追问策略。[论文原文](https://proceedings.mlr.press/v306/sun26i.html)

## 快速开始

基础流程只需要 Python 3.11+。内置合成数据，整个流程可以在 CPU 上运行。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .

nous prepare runs/demo --cases 80 --seed 42
nous train runs/demo --method ofgrpo
nous train runs/demo --method sft
nous train runs/demo --method dpo
```

训练目录包含策略权重和训练指标。给评估命令传入要比较的策略权重文件。

```bash
nous evaluate runs/demo --policies \
  runs/demo/models/ofgrpo-entropy.json \
  runs/demo/models/sft-entropy.json \
  runs/demo/models/dpo-entropy.json
```

评估输出为 `runs/demo/evaluation.json`，包含累计信息增益、澄清轮数、规格证据的精确率和召回率，以及按模拟用户类型和起始条件拆分的结果。

交互演示支持输入一个候选值或若干组 `key=value`，用 `/done` 结束并查看汇总规格。

```bash
nous ask "I need a scientific diagram of the water cycle." \
  --work-dir runs/demo --policy runs/demo/models/ofgrpo-entropy.json
```

## 语言模型训练

可选的 `lm` 依赖支持加载本地 checkpoint 或 Hugging Face 上的因果语言模型。先用适合本机内存的小模型跑通训练和保存，再扩大训练规模。

```bash
python -m pip install -e '.[lm]'
nous train-lm runs/demo --model /path/to/causal-lm \
  --output runs/demo/lm --max-groups 16 --max-length 512 --device cpu
nous evaluate runs/demo --lm-model runs/demo/lm --lm-max-length 512 \
  --profiles template vague --starts draft
```

## 代码与验证

| 路径 | 内容 |
| --- | --- |
| `src/nous/` | 规格空间、信念更新、对话采样、离线训练、评估和 CLI |
| `tests/` | 信息增益、训练目标梯度、回应掩码和端到端测试 |
| `docs/method.md` | 论文公式对应的代码、实现取舍和自定义数据格式 |
| `experiments/` | 后续实验占位目录 |

```bash
python -m unittest discover -s tests -v
```

当前实现基于合成语料和模板候选问题，输出可供绘图系统使用的文字规格。论文的原始语料、模型权重和生成图像的评估留待后续复现。
