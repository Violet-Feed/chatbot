# Chatbot Orchestrator

基于 IMService 的群聊机器人编排服务，支持窗口聚合、LLM 决策、延迟发送、多 Agent、记忆与搜索。

## 核心能力

- 窗口聚合：5s 基础窗口，热度可延长到 10s
- LangGraph 流程：频控 -> LLM 决策 -> 搜索/黑话处理 -> 生成 -> 记忆更新
- 0~2 bot 回复：主/次回复严格约束，bot↔bot 限制
- 记忆系统：滚动摘要、事件记忆、语言风格/热词/术语表
- 搜索与黑话：Jina 搜索 + 上下文推断 + 询问群里
- 发送调度：Redis ZSET 延迟发送，幂等去重，失败重试

## 运行架构

### 组件

- Consumer：订阅 RocketMQ 消息并转换为 MessageEvent
- Planner：窗口管理、LLM 决策、生成与调度、记忆更新
- Scheduler：从 Redis 取出到期任务并调用 IMService.SendMessage
- Memory：MySQL 存储摘要/事件/风格/术语表

### LangGraph 流程

1) load_context：加载 agent 列表、冷却与链深
2) rate_limit：群级 token bucket
3) llm_decide：决定是否回复与选中 agent
4) resolve_search：
   - LLM 决策是否搜索
   - 黑话检测：联网搜索 -> 上下文推断 -> 不懂则追问
   - 结果写入 glossary
5) schedule_primary/secondary：生成与入队
6) finalize：链深更新与 token 退还
7) memory_update：摘要/事件/风格/术语表更新

## 数据协议（im.proto）

核心字段已切换为 sender 抽象：

```proto
message MessageBody{
  int64 sender_id = 1;
  int32 sender_type = 2; // 1=User, 2=Conv, 3=AI
  ...
}

message SendMessageRequest{
  int64 sender_id = 1;
  int32 sender_type = 2;
  ...
}
```

## 记忆结构

存储在 `conversation_style_fingerprint.features` 的 JSON：

```json
{
  "style_rules": ["..."],
  "hotwords": ["..."],
  "glossary": [
    {"term": "词条", "meaning": "解释", "source": "search|context"}
  ],
  "updated_at": 1710000000000
}
```

## 配置

配置通过 `.env` 加载（见 `.env.example`）：

```
# LLM
DASHSCOPE_API_KEY=...
QWEN_BASE_URL=...
QWEN_MODEL=...

# Jina Search
JINA_API_KEY=...
JINA_SEARCH_BASE_URL=https://s.jina.ai/http://www.google.com/search?q=
JINA_SEARCH_TIMEOUT_SEC=20
```

## 主要入口

- `chatbot-bot`: `src/chatbot/main.py`
- `SendScheduler`: `src/chatbot/scheduler/send_scheduler.py`

## 运行说明

1) 配置 `.env`
2) 启动 RocketMQ / Redis / MySQL / IMService
3) 启动服务：

```bash
uv run chatbot-bot
```

## 目录结构

- `src/chatbot/consumer` MQ 消费与 MessageEvent 解析
- `src/chatbot/planner` LangGraph 编排、LLM 决策与生成
- `src/chatbot/scheduler` 发送调度
- `src/chatbot/dal` Redis/MySQL/IMService 接口
- `src/chatbot/memory` 记忆处理逻辑
