# Agent Baseline Evaluation

该目录包含独立的、数据驱动的 Baseline 评测框架。评测器运行当前生产 Agent，但使用确定性规则评分，不使用另一个 LLM 作为裁判，也不会自动修改 Prompt、阈值或任何生产逻辑。

运行：

```powershell
python -m evaluation.run
```

默认读取 `evaluation/datasets/baseline_v1.json`，并写入 `evaluation/results/latest.json`。完整运行会使用 Agent 已有的 SiliconFlow/Qwen、Embedding 和 Reranker 调用方式，可能产生 API 请求；评测代码本身不保存任何凭据。

对已有结果执行纯离线重算（不初始化 Agent，不调用任何 API）：

```powershell
python -m evaluation.rescore
```

默认生成 `rescored.json`、`failure_analysis.json` 和
`relevance_analysis.json`。可通过 `--input`、`--dataset`、`--output`、
`--failure-output`、`--relevance-output` 覆盖对应路径。
