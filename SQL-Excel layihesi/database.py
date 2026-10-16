"""Database connection and query execution helpers."""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import quote_plus

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

from utils import enforce_select_limit, sanitize_identifier, validate_sql_query


load_dotenv()


class DatabaseConfigurationError(RuntimeError):
    """Raised when database environment variables are missing or invalid."""


class QueryTimeoutError(RuntimeError):
    """Raised when PostgreSQL cancels a query because it ran too long."""


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    """Read a bounded integer setting from the environment."""
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        value = int(raw_value)
    except ValueError:
        return default
    return max(minimum, min(value, maximum))


def get_database_url() -> str:
    """Build a SQLAlchemy PostgreSQL URL from environment variables."""
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT", "5432")
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    database = os.getenv("DB_NAME")

    missing = [
        name
        for name, value in {
            "DB_HOST": host,
            "DB_USER": user,
            "DB_PASSWORD": password,
            "DB_NAME": database,
        }.items()
        if not value
    ]
    if missing:
        raise DatabaseConfigurationError(f"Missing database environment variables: {', '.join(missing)}")

    safe_user = quote_plus(user or "")
    safe_password = quote_plus(password or "")
    return f"postgresql+psycopg2://{safe_user}:{safe_password}@{host}:{port}/{database}"


def get_engine() -> Engine:
    """Create a SQLAlchemy engine using the current .env settings."""
    return create_engine(get_database_url(), pool_pre_ping=True, future=True)


def test_connection() -> tuple[bool, str]:
    """Return database connection status and a short message for the UI."""
    engine = get_engine()
    try:
        with engine.connect() as connection:
            version = connection.execute(text("SELECT version()")).scalar_one()
        return True, str(version).split(",")[0]
    except Exception as exc:
        return False, str(exc)


def upload_dataframe_to_postgres(df: pd.DataFrame, table_name: str) -> tuple[str, int]:
    """Create or replace a PostgreSQL table and insert all DataFrame rows."""
    if df.empty:
        raise ValueError("Cannot upload an empty DataFrame.")

    safe_table_name = sanitize_identifier(table_name)
    engine = get_engine()

    # pandas.to_sql uses SQLAlchemy metadata to create a compatible PostgreSQL table.
    df.to_sql(
        safe_table_name,
        engine,
        schema="public",
        if_exists="replace",
        index=False,
        method="multi",
        chunksize=1000,
    )
    return safe_table_name, len(df)


def list_tables() -> list[str]:
    """List user tables in the public schema."""
    engine = get_engine()
    inspector = inspect(engine)
    return sorted(inspector.get_table_names(schema="public"))


def get_table_schema(table_name: str, sample_rows: int = 5) -> dict[str, Any]:
    """Return table schema and a small sample for AI context."""
    safe_table_name = sanitize_identifier(table_name)
    engine = get_engine()
    inspector = inspect(engine)

    columns = inspector.get_columns(safe_table_name, schema="public")
    if not columns:
        raise ValueError(f"Table '{safe_table_name}' was not found or has no columns.")

    column_info = [
        {"name": column["name"], "type": str(column["type"]), "nullable": bool(column.get("nullable", True))}
        for column in columns
    ]

    sample_sql = f'SELECT * FROM public."{safe_table_name}" LIMIT :sample_rows'
    with engine.connect() as connection:
        sample_df = pd.read_sql_query(text(sample_sql), connection, params={"sample_rows": sample_rows})

    return {
        "schema": "public",
        "table": safe_table_name,
        "columns": column_info,
        "sample_rows": sample_df.to_dict(orient="records"),
    }


def execute_select_query(sql: str, allowed_table: str, row_limit: int = 1000) -> pd.DataFrame:
    """Validate and execute a SELECT statement inside a read-only transaction."""
    validate_sql_query(sql, allowed_table=allowed_table)
    limited_sql = enforce_select_limit(sql, row_limit=row_limit)
    engine = get_engine()
    timeout_seconds = _env_int("SQL_QUERY_TIMEOUT_SECONDS", 60, minimum=5, maximum=600)
    timeout_ms = timeout_seconds * 1000

    with engine.connect() as connection:
        transaction = connection.begin()
        try:
            # Defense in depth: PostgreSQL itself rejects writes in this transaction.
            connection.execute(text("SET TRANSACTION READ ONLY"))
            connection.execute(text(f"SET LOCAL statement_timeout = '{timeout_ms}ms'"))
            result = pd.read_sql_query(text(limited_sql), connection)
            transaction.commit()
            return result
        except Exception as exc:
            transaction.rollback()
            message = str(exc).lower()
            if "statement timeout" in message or "query_canceled" in message or "canceling statement" in message:
                raise QueryTimeoutError(
                    f"Database query timed out after {timeout_seconds} seconds. "
                    "Try fewer rows, ask for a grouped summary, or increase SQL_QUERY_TIMEOUT_SECONDS in .env."
                ) from exc
            raise
