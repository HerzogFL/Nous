# 论文方法与实现

参考 Jianwen Sun 等的 *Closing the Expression Gap in LLM Instructions via Socratic Questioning*，ICML 2026。论文第 3 节定义信息增益和离线策略优化，第 4 节与附录给出评估场景和样本结构。[论文原文](https://proceedings.mlr.press/v306/sun26i.html) · [可检索全文](https://arxiv.org/html/2510.27410)

## 从用户请求到训练样本

1. `schema.py` 为科学图定义有限离散属性。内置示例涵盖用途、受众、版式、色彩、组件和关系。
2. `corpus.py` 生成可复现的合成任务，或读取用户提供的数据。划分训练和留出任务后，由训练任务的完整规格估计每个属性的先验。
3. `questions.py` 为尚待明确的属性生成开放式、选项式和组合问题。每轮固定同一个 `Context`。
4. `oracle.py` 按目标规格回答问题。不同模拟用户可省略字段或加入无关文字。候选答案由任务、轮次和问题标识共同确定，调整候选顺序不会改变单个答案。
5. `evidence.py` 将回答解析成离散字段。`belief.py` 应用硬约束，计算该候选带来的熵降。
6. `preferences.py` 保存同轮候选及其信息增益。每轮只选择一个候选的后验来推进对话，其他候选贡献训练标签。

草稿起点会先应用初始证据。累计信息增益从该起点的剩余不确定性开始计算。

## 信息增益

按论文式 (1)–(3)，给定历史后，各属性使用独立边缘分布。联合熵等于各属性的香农熵之和，以比特计量。先验支持无平滑的极大似然计数，也支持少量加性平滑。CLI 默认平滑系数为 0.1。

按论文式 (4)–(7)，当前回答明确的属性变为点质量分布，奖励是更新前后的熵差。

```text
H(P) = sum_i H(P_i)
reward = H(P_before) - H(P_after)
       = sum_{i newly resolved} H(P_before_i)
```

重复证据的奖励为零。与已确认事实冲突的字段会在对话结果中标记为待复核。单轮熵降和单轮后验相对先验的 KL 在非均匀先验下通常数值不同。按回答分布取期望后，两者均等于论文式 (5) 的互信息。

## 策略优化

按论文式 (8)–(9)，对同一问题组的奖励做组内标准化，再计算当前策略与冻结参考策略的概率比。优势全相同时，组内策略梯度项为零。优化 PPO 风格的裁剪目标，加入策略相对于参考策略的 KL 惩罚。

- **CPU 策略**使用对话当下可观测的特征，在有限候选集合上做 softmax。训练使用解析梯度和 Adam 更新，KL 是候选分布间的精确值。
- **因果语言模型**将候选问题及 EOS 作为回应序列。概率比由回应 token 的对数概率之和计算，只有回应位置参与损失。序列 KL 用离线回应前缀上的条件 token KL 近似。推理按候选问题的平均回应对数概率排序。
- **对照实验**提供 SFT 最高奖励候选、DPO 最佳对最差候选，以及用问题涉及的字段数替代信息增益的奖励消融。评估还包含随机、贪心熵和单问题策略。

内置评估报告按留出任务聚类进行配对 bootstrap，并拆分模板、含糊、带噪、初学者和专家式回答。报告测量结构化规格的质量。完整规格准确率对未确认字段使用先验众数填补，证据精确率和召回率只计已确认字段。

## 自定义语料

Schema 文件是一个带 `attributes` 数组的 JSON 对象。每项包含 `key`、`label`、`values` 和 `section`。下面是一个最小示例。

```json
{
  "attributes": [
    {
      "key": "diagram.layout",
      "label": "overall layout",
      "values": ["horizontal", "vertical"],
      "section": "Overview"
    }
  ]
}
```

任务文件每行一个 JSON 对象，包含唯一 `id`、文本 `brief`、完整 `target` 和可选 `draft`。草稿中的字段必须与目标一致。至少需要两条任务样本才能划分训练集和留出集。

```jsonl
{"id":"t01","brief":"Illustrate the phases of the water cycle.","target":{"diagram.layout":"horizontal"},"draft":{}}
{"id":"t02","brief":"Explain how telescope lenses focus light.","target":{"diagram.layout":"vertical"},"draft":{}}
```

```bash
nous prepare runs/custom --schema schema.json --tasks tasks.jsonl
nous train runs/custom --method ofgrpo
nous evaluate runs/custom --policies runs/custom/models/ofgrpo-entropy.json
```

`preferences.jsonl` 每行保存一个任务轮次的共享提示和候选数组。每个候选包含问题 ID、文本、目标属性、模拟回答、熵降和 CPU 特征。可通过 `dataset.json` 查看生成参数和样本数量。用 `--include-cases` 输出逐案例评估数据。
