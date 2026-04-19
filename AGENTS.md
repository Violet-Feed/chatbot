# 🤖 Agent Architecture

## Overview
This repository implements a scalable chatbot agent architecture designed to handle asynchronous messages using RocketMQ, Redis for rate-limiting/caching, MySQL for persistence, and a Planner pattern for LLM-based reasoning and decision making.

## Core Components

The system is constructed using an IoC (Inversion of Control) style container `Container` (defined in `src/chatbot/bootstrap.py`) that wires together the following components:

### 1. Consumers (`src/chatbot/consumer/`)
- **`MessageConsumer`**: Subscribes to a RocketMQ topic to consume incoming user messages asynchronously.
- Forwards parsed messages to the `Planner` for processing.

### 2. Planner (`src/chatbot/planner/`)
The brain of the chatbot.
- **`Planner`**: Orchestrates the core logic when a message is received. Checks memory, makes decisions (via LLM), and schedules or sends responses.
- **`LLMClient`**: Interfaces with Language Models (using OpenAI-compatible APIs).
- **`decision.py`, `prompts.py`, `triggers.py`, `quality.py`**: Handle intent routing, prompt assembly, and response quality assessment.

### 3. Memory (`src/chatbot/memory/`)
Manages the agent's short-term and long-term memory.
- **`episodic.py`**: Handles discrete conversational events.
- **`summary.py`**: Maintains rolling summaries of conversations.
- **`style.py`**: Manages user or bot persona/style parameters.

### 4. Scheduler (`src/chatbot/scheduler/`)
- **`SendScheduler`**: A background task that processes delayed or rate-limited outgoing messages. Polling Redis or other queues to send messages via gRPC at the right time.

### 5. Data Access Layer (DAL) (`src/chatbot/dal/`)
- **MySQL (`mysql/`)**: Stores agent configurations and memory state via SQLAlchemy async engine.
  - `AgentService`: Fetches and updates agent personas/configs.
  - `MemoryService`: Persists conversation histories.
- **Redis (`redis/`)**: Used for distributed locking, rate limiting (via Lua scripts like `bucket_refund.lua` and `bucket_take.lua`), and temporary data storage.
- **RPC (`rpc/`)**:
  - `IMClient`: A gRPC client communicating with the upstream Instant Messaging service (protobuf definitions in `src/chatbot/proto/`).

## Execution Flow

1. **Bootstrap**: `main.py` initializes the environment, sets up logging, and builds the `Container`.
2. **Tasks**: Two main `asyncio` background tasks are launched:
   - `rocketmq-consumer`: Listens for incoming IM messages.
   - `send-scheduler`: Flushes outgoing messages.
3. **Processing**:
   - Message received $ightarrow$ `RocketMQConsumer` $ightarrow$ `Planner.on_message`
   - `Planner` retrieves context from `MemoryService` and `RedisService`.
   - `Planner` constructs prompts and calls `LLMClient`.
   - `Planner` decides whether to reply immediately or schedule a reply.
   - Response sent via `IMClient` (gRPC).

## Tooling & Infrastructure
- **Python**: 3.10+ (managed via `uv`)
- **Protobuf**: Code generation script `gen_proto.sh`
- **LibRocketMQ**: Provided installation script `install_librocketmq.sh`
