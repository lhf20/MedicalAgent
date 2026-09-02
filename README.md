# 医学影像智能问答 Agent

本项目用于构建面向医学影像场景的智能问答 Agent，后续将逐步接入 FastAPI、LangGraph、RAG 与大语言模型能力。

## 当前阶段

当前完成最小 LangGraph Agent：它先识别用户意图，再通过条件路由进入医学影像 RAG、普通聊天或不支持问题的安全兜底。医学影像路径复用既有的 BGE-M3 向量召回 Top-10、BGE Rerank 和最终 Top-3 流程。
模型的配置在.env文件里进行修改。

## 运行

```powershell
python main.py
```

输入医学影像问题或简单问候；输入 `exit` 或 `quit` 结束程序。

测试不访问外部 API：

```powershell
python -m unittest discover -s tests -v
```
