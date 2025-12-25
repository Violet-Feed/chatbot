# src/chatbot/settings.py
from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

def _project_root() -> Path:
    # settings.py 在 src/chatbot/ 下：向上两级是项目根（chatbot/）
    return Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    """
    统一从环境变量 / .env 读取配置（推荐用 uv + pyproject 的方式管理依赖）。
    """

    model_config = SettingsConfigDict(
        env_file=str(_project_root() / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- App ---
    APP_ENV: str = "dev"
    LOG_LEVEL: str = "INFO"

    # --- RocketMQ ---
    ROCKETMQ_NAME_SERVER: str = "127.0.0.1:9876"
    ROCKETMQ_TOPIC: str = "im_conv"
    ROCKETMQ_CONSUMER_GROUP: str = "chatbot_conv_consumer"
    ROCKETMQ_SUBSCRIPTION_EXPRESSION: str = "*"  # 通常 "*"

    # --- Redis / MySQL ---
    REDIS_URL: str = "redis://127.0.0.1:6379/0"
    MYSQL_DSN: str = "mysql+asyncmy://root:password@127.0.0.1:3306/violet?charset=utf8mb4"

    # --- IMService gRPC ---
    IM_GRPC_TARGET: str = "127.0.0.1:50051"

    # --- Window policy (5s + 10s) ---
    WINDOW_BASE_MS: int = 5000
    WINDOW_MAX_MS: int = 10000

    # --- Two-bot gap policy ---
    SECOND_GAP_MIN_MS: int = 3000
    SECOND_GAP_MAX_MS: int = 8000

    # --- Group rate limit (token bucket, simplified) ---
    GROUP_BUCKET_CAP: int = 3
    GROUP_BUCKET_PERIOD_MS: int = 300_000  # 5 min

    # --- bot↔bot strict token ---
    BOT2BOT_TOKEN_TTL_SEC: int = 600  # 10 min

    # --- Scheduler worker ---
    SCHEDULER_POLL_INTERVAL_MS: int = 200  # 轮询 sched:zset 间隔
    SCHEDULER_BATCH_SIZE: int = 64

    DASHSCOPE_API_KEY: str = ""
    QWEN_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    QWEN_MODEL: str = "qwen-plus"
    QWEN_TEMPERATURE: float = 0.7
    QWEN_TIMEOUT_SEC: int = 30
