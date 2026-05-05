# Chatbot Agent Architecture

## Overview

This repository implements a multi-agent group-chat bot that consumes IM messages asynchronously via RocketMQ, buffers
them in Redis using a time-window aggregation pattern, reasons with a LangGraph-based LLM pipeline, persists memory in
MySQL, and sends replies through a gRPC IM service.

## Core Components

The system is wired together by an IoC container `Container` (`src/chatbot/bootstrap.py`) that holds all singleton
components. `build_container()` creates all instances: Redis, MySQL, gRPC clients (IM/Aigc/Action), `LLMClient`,
`ConversationGraph`, `WindowManager`, `MessageConsumer`, `SendScheduler`.

### 1. Agent (`src/chatbot/agent/`)

The brain of the chatbot. Incoming messages flow through two layers:

- **`WindowManager`** (`window.py`): Entry point for every message. Implements a time-window aggregation pattern —
  messages are appended to a per-conversation Redis buffer (`grp:{id}:win:buf`) and processed as a batch when the window
  closes (default 10 seconds). Uses a Redis lock (`grp:{id}:win:close_scheduled`, `SET NX EX 300`) to prevent concurrent
  window-close tasks. Resolves `sender_name` via gRPC (users via `ActionClient`, agents via `AigcClient`). After window
  close, calls `ConversationGraph.run()`.

- **`ConversationGraph`** (`graph.py`): A `langgraph.StateGraph` with 5 nodes and conditional/parallel edges:

  ```
  load_context ──(agents empty?)──→ END
           │
      (agents exist)
           ├─→ decide ──(should reply?)──→ generate → send → END
           │                          └──→ END
           └─→ update_memories → END   (runs in parallel with decide)
  ```

  - `load_context` — fetches conversation info via `IMClient.get_conversation_info()`, loads agent personas (groups via
    gRPC, private chats by parsing `con_id`), loads memory summaries and glossary from MySQL. Returns `{}` (routing to
    END) if no agents found, otherwise fans out to `decide` + `update_memories` in parallel.
  - `decide` — calls `LLMClient.decide_reply()` to determine whether any agent should respond and which one. Private AI
    chats (`con_type == 4`) always reply without LLM invocation.
  - `update_memories` — background memory update node (runs concurrently with `decide`). Handles short-term/long-term
    summary updates and glossary term extraction/inference.
  - `generate` — calls `LLMClient.generate_reply()` which runs a LangGraph React agent (LLM + tools) to produce a reply.
  - `send` — enqueues the reply to Redis sorted set `sched:send` for the `SendScheduler` to dispatch.

- **`LLMClient`** (`llm.py`): Wraps `langchain_openai.ChatOpenAI` pointed at DashScope (Qwen model). Creates the React
  agent (`create_react_agent`) at init time with `web_search` + `fetch_more_context` tools. Exposes:
  - `decide_reply(inputs)` — intent routing, returns whether to reply and which agent.
  - `generate_reply(inputs)` — calls the React agent with `AGENT_SYSTEM`/`AGENT_USER` prompts, extracts the final
    `AIMessage` from the message chain. Uses `recursion_limit = max_tool_calls * 2 + 3` and passes `con_short_id`,
    `min_con_index`, `sender_id`, `sender_type` via `config["configurable"]` for tool access.
  - `update_short_memory(inputs)` — rolling short-term summary update.
  - `update_long_memory(inputs)` — rolling long-term summary update.
  - `extract_unknown_terms(inputs)` — glossary term extraction from messages.
  - `infer_glossary_meanings(inputs)` — glossary meaning inference.

  All methods use typed dataclass inputs (`DecisionInputs`, `GenerateInputs`, `ShortMemoryInputs`, `LongMemoryInputs`,
  `GlossaryExtractInputs`, `GlossaryInferInputs`).

- **`prompts.py`**: All LLM prompt templates — `AGENT_SYSTEM`/`AGENT_USER` (reply generation), `DECISION_SYSTEM`/
  `DECISION_USER` (intent routing), short/long memory prompts, glossary extraction/inference prompts.

- **`state.py`**: `Agent` dataclass (`agent_id`, `name`, `personality`) and `ConversationState` TypedDict (LangGraph
  pipeline state with keys: `con_short_id`, `con_id`, `con_type`, `items`, `history_items`, `con_name`,
  `con_description`, `agents`, `memory_state`, `glossary`, `should_reply`, `primary_agent`, `reply_reason`,
  `reply_text`).

- **`tools.py`**: Two LangChain tool factories:
  - `make_web_search_tool(...)` — creates a `web_search` tool that calls the Bocha AI Web Search API via `urllib.request`
    (wrapped in `asyncio.to_thread` for async compatibility). Configurable: `base_url`, `api_key`, `timeout_sec`,
    `max_chars`, `count`, `freshness`, `summary`.
  - `make_fetch_context_tool(im_client)` — creates a `fetch_more_context` tool that calls
    `IMClient.get_message_by_conversation()` for additional conversation history. Runtime context (`con_short_id`,
    `min_con_index`, `sender_id`, `sender_type`) is injected via `RunnableConfig["configurable"]` at invoke time.

### 2. Consumer (`src/chatbot/consumer/`)

- **`MessageConsumer`** (`message_consumer.py`): Wraps `rocketmq.client.PushConsumer` (C++ extension). Subscribes to a
  RocketMQ topic, parses JSON payloads into `im_pb2.MessageEvent` protobuf objects, and forwards them to
  `WindowManager.on_message()`. The C++ callback bridges to asyncio via `run_coroutine_threadsafe()`.

### 3. Scheduler (`src/chatbot/scheduler/`)

- **`SendScheduler`** (`send_scheduler.py`): Polls the Redis sorted set `sched:send` for due messages (score ≤ current
  timestamp). Atomically claims tasks via `ZREM`, then sends them via `IMClient.send_message()`. Retries up to 2
  additional times (3 total attempts) with `retry_count * 1000ms` delays. Runs as a background asyncio task with 200ms
  poll interval and 100 batch size.

### 4. Memory Management

Memory is managed through the `update_memories` graph node in `ConversationGraph`:

- **MySQL persistence** via `MemoryService` (`dal/mysql/memory_service.py`):
  - `chatbot_memory_summary` table: per-conversation short-term and long-term rolling summaries with version tracking
    and timestamps (`short_updated_at`, `long_updated_at`).
  - `chatbot_memory_glossary` table: crowd-sourced slang/colloquial term extraction — terms are collected from
    conversations, meanings are inferred by LLM once `count >= 5` (configurable via `min_count`), and re-inferred every
    10 additional occurrences.

- **Memory update logic** in `ConversationGraph._update_memories()` (`agent/graph.py`):
  - Short-term summary: updated every 25 merge counts or 30 minutes timeout.
  - Long-term summary: updated when `all_buf` length ≥ 200 or 24 hours timeout.
  - Glossary: terms extracted from current window items; meanings inferred when a term reaches `min_count` occurrences.

- **Redis buffering** via `RedisService` (`dal/redis/service.py`):
  - Window buffer (`grp:{id}:win:buf`) — current window messages.
  - All buffer (`grp:{id}:win:all_buf`) — accumulated history for long-term memory.
  - Merge count (`grp:{id}:win:merge_count`) — counter for short-memory trigger threshold.
  - Window deadline (`grp:{id}:win:deadline_ts`) — window close timestamp.
  - Close lock (`grp:{id}:win:close_scheduled`) — distributed lock to prevent concurrent closes.
  - Send queue (`sched:send` sorted set) — scheduled send tasks.

### 5. Data Access Layer (`src/chatbot/dal/`)

- **MySQL** (`mysql/`):
  - `client.py`: Creates SQLAlchemy `AsyncEngine` + `async_sessionmaker` (asyncmy driver).
  - `memory_service.py`: `MemoryService` — CRUD for summary and glossary tables. `SummaryState` dataclass holds both
    summaries with version and timestamp tracking.

- **Redis** (`redis/`):
  - `client.py`: Creates `redis.asyncio.Redis` client.
  - `keys.py`: Redis key naming functions — `win_deadline`, `win_buf`, `winclose_scheduled`, `win_all_buf`,
    `win_merge_count`.
  - `service.py`: `RedisService` — window buffer operations (upsert deadline, append message, drain, merge to all_buf,
    trim all_buf), merge count management, distributed lock for close scheduling, and `enqueue_send_task()`
    (`ZADD sched:send`).

- **RPC** (`rpc/`):
  - `im.py`: `IMClient` — async gRPC client wrapping `im_pb2_grpc.IMServiceStub`. Provides `send_message()`,
    `get_message_by_conversation()`, `get_conversation_info()`, `get_conversation_agents()`. All methods validate
    `BaseResp.StatusCode == Success(1000)`.
  - `aigc.py`: `AigcClient` — async gRPC client wrapping `aigc_pb2_grpc.AigcServiceStub`. Provides
    `get_agents_by_ids()` returning `AgentInfo` protos with `agent_id`, `agent_name`, `personality`.
  - `action.py`: `ActionClient` — async gRPC client wrapping `action_pb2_grpc.ActionServiceStub`. Provides
    `get_user_infos()` returning `{user_id: UserInfo}` dict for sender name resolution.

### 6. Proto (`src/chatbot/proto/`) & Generated Code (`src/chatbot/proto_gen/`)

- `common.proto`: `BaseResp`, `StatusCode` enum, `SpecialUser` enum.
- `im.proto`: `IMService` with RPCs for messaging and conversation management.
- `aigc.proto`: `AigcService` with `GetAgentsByIds`, `GetAgentsByUser`, plus material/creation CRUD RPCs.
  `AgentInfo` message carries agent persona data.
- `action.proto`: `ActionService` with `GetUserInfos`, `Login`, `Register`, etc. `UserInfo` message carries user profile.
- Generated Python code produced by `gen_proto.sh` using `grpc_tools.protoc`.

### 7. Utils (`src/chatbot/utils/`)

- `json.py`: `json_dumps()` (orjson-based), `safe_obj_loads()`.
- `time.py`: `now_ms()` — current UTC timestamp in milliseconds.

## Execution Flow

1. **Bootstrap**: `main.py` loads settings from `.env` via `pydantic-settings`, builds the `Container` (wiring all
   components), and sets up signal handlers.

2. **Background tasks**: Two asyncio tasks run concurrently:
   - `rocketmq-consumer`: `MessageConsumer.run_forever()` — blocks on a stop event, processes messages via C++
     callback.
   - `send-scheduler`: `SendScheduler.run_forever()` — polls Redis ZSET for due messages and dispatches them.

3. **Message processing pipeline**:
   ```
   RocketMQ → MessageConsumer (JSON → protobuf)
     → WindowManager.on_message()
       → filter (text only, group/AI-chat only)
       → resolve sender_name (gRPC)
       → append to Redis buffer (grp:{con_short_id}:win:buf)
       → set window deadline (grp:{con_short_id}:win:deadline_ts)
       → spawn wait_window task (with Redis lock)
         → sleep until deadline
         → close_window()
           → drain buffer, merge to all_buf, incr merge_count
           → ConversationGraph.run(state)
             → load_context (get conversation info, load agents, load memory/glossary)
               ├─ (no agents) → END
               └─ (has agents) → fan-out:
                   ├─ decide (LLM: should reply? which agent?)
                   │    └─ (should reply) → generate (React agent) → send → END
                   └─ update_memories (short/long/glossary) → END
   ```

4. **React agent tool context injection**:
   `LLMClient.generate_reply()` passes runtime context through `config["configurable"]`:
   ```python
   config = {
       "recursion_limit": max_tool_calls * 2 + 3,
       "configurable": {
           "con_short_id": ...,
           "min_con_index": ...,
           "sender_id": ...,
           "sender_type": 2,
       },
   }
   ```
   The `fetch_more_context` tool reads these from its `RunnableConfig` to make gRPC calls.

5. **Send pipeline**:
   ```
   SendScheduler.run_forever()
     → ZRANGEBYSCORE sched:send (score ≤ now)
     → ZREM (claim task atomically)
     → IMClient.send_message() via gRPC
     → retry on failure (up to MAX_RETRY=2, delay = retry_count * 1000ms)
   ```

6. **Graceful shutdown**: On SIGINT/SIGTERM or worker exception → stop consumer & scheduler → cancel tasks →
   `await container.aclose()` (dispose gRPC channels, MySQL engine, Redis connection).

## Configuration (Settings)

All config via environment variables or `.env` file (loaded by `pydantic-settings`):

| Setting | Default | Description |
|---------|---------|-------------|
| `DASHSCOPE_API_KEY` | `""` | DashScope API key for Qwen LLM |
| `QWEN_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | Qwen API base URL |
| `QWEN_MODEL` | `qwen-plus` | Qwen model name |
| `QWEN_TEMPERATURE` | `0.7` | LLM temperature |
| `QWEN_TIMEOUT_SEC` | `30` | LLM request timeout |
| `BOCHA_API_KEY` | `""` | Bocha AI search API key |
| `WEB_SEARCH_BASE_URL` | `https://api.bochaai.com/v1/web-search` | Bocha search endpoint |
| `WEB_SEARCH_TIMEOUT_SEC` | `20` | Search request timeout |
| `AGENT_MAX_TOOL_CALLS` | `2` | Max tool calls for React agent |
| `ROCKETMQ_NAME_SERVER` | `127.0.0.1:9876` | RocketMQ nameserver |
| `ROCKETMQ_TOPIC` | `im_conv` | RocketMQ topic |
| `REDIS_URL` | `redis://127.0.0.1:6379/0` | Redis connection |
| `MYSQL_DSN` | `mysql+asyncmy://...` | MySQL connection string |
| `IM_GRPC_TARGET` | `127.0.0.1:3004` | IM gRPC endpoint |
| `AIGC_GRPC_TARGET` | `127.0.0.1:3005` | Aigc gRPC endpoint |
| `ACTION_GRPC_TARGET` | `127.0.0.1:3003` | Action gRPC endpoint |
| `WINDOW_SEC` | `10` | Time-window aggregation seconds |

## Tooling & Infrastructure

- **Python**: 3.11+ (managed via `uv`)
- **Runtime**: `chatbot-bot` entrypoint (`main.py`), single-process async (consumer + scheduler co-located)
- **LangChain / LangGraph**: StateGraph pipeline with conditional fan-out edges, React agent pattern for tool-using
  generation
- **Protobuf**: `gen_proto.sh` via `grpc_tools.protoc`
- **LibRocketMQ**: `install_librocketmq.sh` provides the C++ RocketMQ client library