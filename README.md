# 医学影像智能问答 Agent

本项目用于构建面向医学影像场景的智能问答 Agent，后续将逐步接入 FastAPI、LangGraph、RAG 与大语言模型能力。

## 当前阶段

当前完成两阶段 RAG Demo：从 `data/knowledge/` 加载 Markdown 文档、按段落优先切分、通过 SiliconFlow 的 `BAAI/bge-m3` 创建向量并以余弦相似度召回 Top-10，再用 `BAAI/bge-reranker-v2-m3` 重排序并保留 Top-3。最终片段会作为上下文发送给已配置的 Qwen 模型。
模型的配置在.env文件里进行修改。

## 运行

```powershell
python main.py
```

输入医学影像问题；输入 `exit` 或 `quit` 结束程序。
