"""异步数据库引擎与会话管理。

* 主库为 **MySQL + aiomysql**（生产配置）。
* 若 MySQL 不可用（未安装、密码错误、网络不通），自动回退到 **SQLite + aiosqlite**，
  保证项目「拉下来就能跑」——回退状态会记录在 ``get_backend_info()`` 中，
  ``/health`` 接口会如实返回，不会假装在用 MySQL。
* 使用 ``AsyncSession`` + 连接池，配合 asyncio 支撑高并发。
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from sqlalchemy import text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from config import get_settings

from .models import Base

logger = logging.getLogger("ai_resume_helper.db")

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None
_backend_info: dict[str, Any] = {
    "backend": "unknown",
    "url": "",
    "fallback": False,
    "detail": "未初始化",
}
_init_lock = asyncio.Lock()


# ---------------------------------------------------------------------- #
# 引擎构建
# ---------------------------------------------------------------------- #
def _engine_kwargs(url_str: str) -> dict[str, Any]:
    cfg = get_settings()
    kwargs: dict[str, Any] = {"echo": cfg.db_echo, "future": True}
    if url_str.startswith("sqlite"):
        kwargs["pool_pre_ping"] = True
    else:
        kwargs.update(
            pool_pre_ping=True,
            pool_size=cfg.db_pool_size,
            max_overflow=cfg.db_max_overflow,
            pool_recycle=cfg.db_pool_recycle,
            pool_timeout=30,
        )
    return kwargs


async def _ensure_mysql_database(url_str: str) -> None:
    """MySQL 库不存在时自动创建（否则连接直接报 Unknown database）。"""
    cfg = get_settings()
    if not cfg.mysql_auto_create_database:
        return

    url = make_url(url_str)
    database = url.database
    if not database:
        return

    # 注意：SQLAlchemy 的 URL.set(database=None) 不会真的清空库名，
    # 必须用 URL.create 重新拼一个「不带 database」的服务端连接串。
    server_url = URL.create(
        drivername=url.drivername,
        username=url.username,
        password=url.password,
        host=url.host,
        port=url.port,
        query=dict(url.query),
    )
    server_engine = create_async_engine(
        server_url, isolation_level="AUTOCOMMIT", poolclass=None, echo=False
    )
    try:
        async with server_engine.connect() as conn:
            await conn.execute(
                text(
                    f"CREATE DATABASE IF NOT EXISTS `{database}` "
                    "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                )
            )
        logger.info("MySQL 数据库 `%s` 已就绪", database)
    finally:
        await server_engine.dispose()


async def _try_engine(url_str: str) -> AsyncEngine:
    engine = create_async_engine(url_str, **_engine_kwargs(url_str))
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
    return engine


# ---------------------------------------------------------------------- #
# 初始化
# ---------------------------------------------------------------------- #
async def init_database(create_tables: bool = True) -> dict[str, Any]:
    """初始化引擎、建表，返回后端信息（含是否发生回退）。"""
    global _engine, _session_factory, _backend_info

    async with _init_lock:
        if _engine is not None:
            return _backend_info

        cfg = get_settings()
        primary = cfg.database_url
        errors: list[str] = []

        if primary.startswith("mysql"):
            try:
                await _ensure_mysql_database(primary)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"自动建库失败：{type(exc).__name__}: {exc}")

        engine: AsyncEngine | None = None
        used_url = primary
        fallback = False

        try:
            engine = await _try_engine(primary)
            logger.info("数据库连接成功：%s", make_url(primary).render_as_string(hide_password=True))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"主库连接失败：{type(exc).__name__}: {exc}")
            logger.warning("主库不可用，回退到 SQLite：%s", exc)
            engine = await _try_engine(cfg.sqlite_fallback_url)
            used_url = cfg.sqlite_fallback_url
            fallback = True

        if create_tables:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            await _ensure_columns(engine)

        _engine = engine
        _session_factory = async_sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
        )
        _backend_info = {
            "backend": "mysql" if used_url.startswith("mysql") else "sqlite",
            "url": make_url(used_url).render_as_string(hide_password=True),
            "fallback": fallback,
            "errors": errors,
            "detail": "SQLite 回退（MySQL 不可用）" if fallback else "正常",
        }
        return _backend_info


async def _ensure_columns(engine: AsyncEngine) -> list[str]:
    """轻量自动迁移：给已存在的表补上模型里新增的列。

    ``create_all`` 只会建新表，不会给老表加字段。项目会持续迭代，
    为了不强制用户手动执行 DDL，这里对比模型与实际表结构，
    对缺失的列执行 ``ALTER TABLE ... ADD COLUMN``（MySQL / SQLite 语法一致）。
    """
    from sqlalchemy import inspect
    from sqlalchemy.schema import CreateColumn
    from sqlalchemy.dialects import mysql, sqlite

    dialect = mysql.dialect() if engine.dialect.name == "mysql" else sqlite.dialect()
    added: list[str] = []

    def _collect(sync_conn) -> list[tuple[str, str, str]]:
        inspector = inspect(sync_conn)
        existing_tables = set(inspector.get_table_names())
        statements: list[tuple[str, str, str]] = []
        for table in Base.metadata.sorted_tables:
            if table.name not in existing_tables:
                continue
            present = {col["name"] for col in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in present:
                    continue
                ddl = str(CreateColumn(column).compile(dialect=dialect)).strip()
                statements.append((table.name, column.name, ddl))
        return statements

    async with engine.begin() as conn:
        pending = await conn.run_sync(_collect)
        for table_name, column_name, ddl in pending:
            await conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {ddl}"))
            added.append(f"{table_name}.{column_name}")

    if added:
        logger.info("数据库结构已自动补齐新增列：%s", added)
    return added


async def dispose_database() -> None:
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None


def get_engine() -> AsyncEngine:
    if _engine is None:
        raise RuntimeError("数据库尚未初始化，请先调用 init_database()")
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    if _session_factory is None:
        raise RuntimeError("数据库尚未初始化，请先调用 init_database()")
    return _session_factory


def get_backend_info() -> dict[str, Any]:
    return dict(_backend_info)


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """事务作用域：正常提交，异常回滚。"""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
