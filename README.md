# 医学影像智能问答 Agent

本项目用于构建面向医学影像场景的智能问答 Agent，后续将逐步接入 FastAPI、LangGraph、RAG 与大语言模型能力。

## 当前阶段

当前完成最小 LangGraph Agent：它先识别用户意图，再通过条件路由进入医学影像 RAG、普通聊天或不支持问题的安全兜底。医学影像路径复用既有的 BGE-M3 向量召回 Top-10、BGE Rerank 和最终 Top-3 流程。

另包含 `query_petct_result` Tool，用于查询 `data/petct_demo_results.json` 中虚构、脱敏的 PET-CT 结构化演示结果。涉及具体检查事实的问题会优先调用 Tool；同时请求医学解释时，Agent 还会执行 RAG 检索，并将 Tool Result（具体事实）与 RAG Context（一般医学知识）分区传给 Qwen。Tool 查询失败时不会由 LLM 猜测数据。

Agent 在单个进程会话中保留最近 6 轮对话和最近一次成功 PET-CT 查询的结构化实体，可补全“这个病灶”“这个指标”“刚才那个检查”等指代。缺少可靠上下文时会要求用户澄清；可调用 `reset_conversation()` 清空当前会话。

Intent Router V2 采用规则优先、上下文补全、LLM fallback：明确医学问题、PET-CT 查询、问候和明显越界任务走快速规则；只有规则无法可靠判断时才让 Qwen 结合最近对话，以固定 JSON 格式分类。

模型的配置在.env文件里进行修改。
代码主体在app/下面，数据放在同级目录data/下

## 运行

```powershell
python main.py
```

输入医学影像问题或简单问候；输入 `exit` 或 `quit` 结束程序。

测试不访问外部 API：

```powershell
python -m unittest discover -s tests -v
```
