# src/chatbot/dal/mysql/client.py
from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine


def new_mysql_client(mysql_dsn: str) -> AsyncEngine:
    """
    创建 MySQL 异步引擎（SQLAlchemy AsyncEngine）。
    - pool_pre_ping: 避免空闲断连导致首次查询失败
    - pool_recycle: 避免连接被服务端回收
    """
    return create_async_engine(
        mysql_dsn,
        pool_pre_ping=True,
        pool_recycle=3600,
        echo=False,
        future=True,
    )


def new_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """
    创建 AsyncSession 工厂。
    expire_on_commit=False：提交后对象不自动过期，避免后续访问触发隐式查询。
    """
    return async_sessionmaker(
        bind=engine,
        expire_on_commit=False,
        autoflush=False,
        autocommit=False,
    )
