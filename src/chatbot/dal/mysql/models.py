# src/chatbot/dal/mysql/models.py
from __future__ import annotations

from sqlalchemy import BigInteger, Float, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class AgentInfo(Base):
    __tablename__ = "agent_info"

    agent_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    agent_name: Mapped[str] = mapped_column(String(64), nullable=False)
    avatar_uri: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    description: Mapped[str] = mapped_column(String(512), nullable=False, default="")
    personality: Mapped[str] = mapped_column(Text, nullable=False, default="")

    owner_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)

    create_time: Mapped[int] = mapped_column(BigInteger, nullable=False)
    modify_time: Mapped[int] = mapped_column(BigInteger, nullable=False)

    status: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    extra: Mapped[str] = mapped_column(Text, nullable=False, default="")


class ConversationAgentInfo(Base):
    __tablename__ = "conversation_agent_info"
    __table_args__ = (
        UniqueConstraint("con_short_id", "agent_id", name="uq_con_agent"),
        Index("idx_con_agent_con", "con_short_id"),
        Index("idx_con_agent_agent", "agent_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    con_short_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    agent_id: Mapped[int] = mapped_column(BigInteger, nullable=False)

    create_time: Mapped[int] = mapped_column(BigInteger, nullable=False)
    modify_time: Mapped[int] = mapped_column(BigInteger, nullable=False)

    status: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    extra: Mapped[str] = mapped_column(Text, nullable=False, default="")


class ConversationRollingSummary(Base):
    __tablename__ = "conversation_rolling_summary"
    __table_args__ = (Index("idx_summary_updated", "updated_at"),)

    con_short_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    last_con_index: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    updated_at: Mapped[int] = mapped_column(BigInteger, nullable=False)


class ConversationEpisodicMemory(Base):
    __tablename__ = "conversation_episodic_memory"
    __table_args__ = (
        Index("idx_mem_con_expire", "con_short_id", "expire_at"),
        Index("idx_mem_con_imp", "con_short_id", "importance"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    con_short_id: Mapped[int] = mapped_column(BigInteger, nullable=False)

    event_type: Mapped[str] = mapped_column(String(32), nullable=False, default="generic")
    content: Mapped[str] = mapped_column(Text, nullable=False)

    importance: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    expire_at: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)

    created_at: Mapped[int] = mapped_column(BigInteger, nullable=False)


class ConversationStyleFingerprint(Base):
    __tablename__ = "conversation_style_fingerprint"

    con_short_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    features: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    updated_at: Mapped[int] = mapped_column(BigInteger, nullable=False)
