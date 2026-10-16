"""Shared data cleaning, SQL safety, and charting utilities."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import sqlparse
except ImportError:  # pragma: no cover - exercised only before optional dependency install
    sqlparse = None


DISALLOWED_SQL_KEYWORDS = {
    "ALTER",
    "ANALYZE",
    "CALL",
    "COMMENT",
    "COPY",
    "CREATE",
    "DELETE",
    "DROP",
    "EXEC",
    "EXECUTE",
    "GRANT",
    "INSERT",
    "INTO",
    "MERGE",
    "REINDEX",
    "RESET",
    "REVOKE",
    "SET",
    "TRUNCATE",
    "UPDATE",
    "VACUUM",
}


def sanitize_identifier(value: str, fallback: str = "uploaded_data") -> str:
    """Convert user-provided table or column names into safe PostgreSQL identifiers."""
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", str(value).strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = fallback
    if cleaned[0].isdigit():
        cleaned = f"{fallback}_{cleaned}"
    return cleaned[:63]


def generate_table_name(filename: str) -> str:
    """Create a unique, readable table name from an uploaded filename."""
    stem = Path(filename).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return sanitize_identifier(f"{stem}_{timestamp}")


def clean_dataframe_for_database(df: pd.DataFrame) -> pd.DataFrame:
    """Drop empty rows/columns and normalize missing values before database upload."""
    cleaned = df.copy()
    cleaned = cleaned.dropna(axis=0, how="all").dropna(axis=1, how="all")
    cleaned = cleaned.where(pd.notnull(cleaned), None)
    return cleaned


def normalize_dataframe_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Rename DataFrame columns to unique safe SQL identifiers."""
    renamed = df.copy()
    used: dict[str, int] = {}
    column_map: dict[str, str] = {}
    new_columns: list[str] = []

    for original in renamed.columns:
        base = sanitize_identifier(str(original), fallback="column")
        count = used.get(base, 0)
        used[base] = count + 1
        safe_name = base if count == 0 else sanitize_identifier(f"{base}_{count + 1}", fallback="column")
        new_columns.append(safe_name)
        column_map[str(original)] = safe_name

    renamed.columns = new_columns
    return renamed, column_map


def _strip_sql_comments(sql: str) -> str:
    """Remove SQL comments before validation checks."""
    if sqlparse is not None:
        return sqlparse.format(sql, strip_comments=True).strip()
    without_block_comments = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    without_line_comments = re.sub(r"--.*?$", "", without_block_comments, flags=re.MULTILINE)
    return without_line_comments.strip()


def _normalized_sql_for_table_match(sql: str) -> str:
    """Normalize punctuation so table references can be matched reliably."""
    return re.sub(r'["`\[\]]', "", sql.lower())


def validate_sql_query(sql: str, allowed_table: str | None = None) -> None:
    """Validate that SQL is a single SELECT statement and targets the active table."""
    if not sql or not sql.strip():
        raise ValueError("SQL query is empty.")

    stripped = _strip_sql_comments(sql)
    if sqlparse is not None:
        statements = [statement for statement in sqlparse.parse(stripped) if str(statement).strip()]
        if len(statements) != 1:
            raise ValueError("Only one SQL statement is allowed.")
        statement = statements[0]
        first_token = next((token for token in statement.tokens if not token.is_whitespace), None)
        first_keyword = first_token.normalized.upper() if first_token is not None else ""
    else:
        first_match = re.match(r"^\s*([a-zA-Z]+)\b", stripped)
        first_keyword = first_match.group(1).upper() if first_match else ""

    if first_keyword != "SELECT":
        raise ValueError("Only SELECT queries are allowed.")

    upper_sql = stripped.upper()
    for keyword in DISALLOWED_SQL_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", upper_sql):
            raise ValueError(f"Disallowed SQL keyword detected: {keyword}")

    if ";" in stripped.rstrip(";"):
        raise ValueError("Semicolons are only allowed as the final character.")

    if allowed_table:
        normalized_sql = _normalized_sql_for_table_match(stripped)
        safe_table = sanitize_identifier(allowed_table)
        if safe_table not in normalized_sql:
            raise ValueError(f"Query must reference the active table '{safe_table}'.")


def enforce_select_limit(sql: str, row_limit: int = 1000) -> str:
    """Add a LIMIT clause to SELECT statements that do not already have one."""
    clean_sql = _strip_sql_comments(sql).rstrip(";")
    safe_limit = max(1, min(int(row_limit), 5000))
    if re.search(r"\blimit\b", clean_sql, flags=re.IGNORECASE):
        return clean_sql
    return f"{clean_sql}\nLIMIT {safe_limit}"


def format_schema_for_display(schema: dict[str, Any]) -> str:
    """Render table schema as small HTML chips for Streamlit."""
    chips = []
    for column in schema.get("columns", []):
        chips.append(
            f'<span class="schema-chip">{column["name"]} <small>{column["type"]}</small></span>'
        )
    return "".join(chips)


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return list(df.select_dtypes(include="number").columns)


def _datetime_columns(df: pd.DataFrame) -> list[str]:
    columns: list[str] = list(df.select_dtypes(include=["datetime", "datetimetz"]).columns)
    for column in df.columns:
        if column in columns:
            continue
        if any(word in column.lower() for word in ["date", "month", "time"]):
            converted = pd.to_datetime(df[column], errors="coerce")
            if converted.notna().mean() >= 0.75:
                df[column] = converted
                columns.append(column)
    return columns


def _categorical_columns(df: pd.DataFrame) -> list[str]:
    excluded = set(_numeric_columns(df)) | set(_datetime_columns(df))
    return [column for column in df.columns if column not in excluded]


def build_auto_chart(df: pd.DataFrame, chart_type: str = "Auto", question: str = "") -> Any | None:
    """Create a Plotly chart when the result shape is suitable."""
    import plotly.express as px

    if df.empty or len(df.columns) < 1:
        return None

    chart_df = df.copy()
    numeric_cols = _numeric_columns(chart_df)
    datetime_cols = _datetime_columns(chart_df)
    categorical_cols = _categorical_columns(chart_df)

    if not numeric_cols:
        return None

    selected_type = chart_type
    question_lower = question.lower()
    if selected_type == "Auto":
        wants_share = any(word in question_lower for word in ["share", "percent", "percentage", "distribution"])
        if datetime_cols:
            selected_type = "Line"
        elif wants_share and categorical_cols and 2 <= len(chart_df) <= 12:
            selected_type = "Pie"
        elif categorical_cols and 2 <= len(chart_df) <= 8 and chart_df[numeric_cols[0]].ge(0).all():
            selected_type = "Pie"
        else:
            selected_type = "Bar"

    title = "Query result visualization"

    if selected_type == "Pie":
        if not categorical_cols:
            return None
        names = categorical_cols[0]
        values = numeric_cols[0]
        grouped = chart_df.groupby(names, dropna=False, as_index=False)[values].sum().head(12)
        return px.pie(grouped, names=names, values=values, title=title, hole=0.35)

    if selected_type == "Line":
        x_col = datetime_cols[0] if datetime_cols else (categorical_cols[0] if categorical_cols else chart_df.index)
        y_col = numeric_cols[0]
        return px.line(chart_df, x=x_col, y=y_col, markers=True, title=title)

    if selected_type == "Bar":
        x_col = categorical_cols[0] if categorical_cols else chart_df.index
        y_col = numeric_cols[0]
        return px.bar(chart_df, x=x_col, y=y_col, title=title)

    return None
