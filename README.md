# 医学影像智能问答 Agent

本项目用于构建面向医学影像场景的智能问答 Agent，后续将逐步接入 FastAPI、LangGraph、RAG 与大语言模型能力。

## 当前阶段

当前完成最小 RAG Demo：从 `data/knowledge/` 加载 Markdown 文档、按段落优先切分、通过 SiliconFlow 的 `BAAI/bge-m3` 创建向量，并以余弦相似度进行本地 Top-K 检索。检索结果会作为上下文发送给已配置的 Qwen 模型。

## 运行

```powershell
python main.py
```

输入医学影像问题；输入 `exit` 或 `quit` 结束程序。
