"""AI-powered natural-language to SQL generation.

The app supports two providers:
- Ollama for free local SQL generation.
- OpenAI for hosted API-based SQL generation.
"""

from __future__ import annotations

import json
import os
import re
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

from utils import validate_sql_query


load_dotenv()


class OpenAIConfigurationError(RuntimeError):
    """Raised when the configured AI provider cannot be used."""


class AIProviderTimeoutError(OpenAIConfigurationError):
    """Raised when the configured AI provider does not answer in time."""


@dataclass(frozen=True)
class SQLGenerationResult:
    """Structured AI SQL generation output."""

    sql: str
    explanation: str


def _get_openai_client():
    """Create an OpenAI client from OPENAI_API_KEY."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise OpenAIConfigurationError(
            "The openai package is not installed. Run 'pip install -r requirements.txt' first."
        ) from exc

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise OpenAIConfigurationError("OPENAI_API_KEY is missing. Add it to your .env file.")
    return OpenAI(api_key=api_key, timeout=_env_int("AI_TIMEOUT_SECONDS", 240, minimum=30, maximum=900))


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


def _schema_to_prompt(schema: dict[str, Any]) -> str:
    """Serialize schema context compactly for the model."""
    return json.dumps(schema, default=str, indent=2)


def _quote_identifier(identifier: str) -> str:
    """Quote a PostgreSQL identifier."""
    return f'"{str(identifier).replace(chr(34), chr(34) * 2)}"'


def _table_ref(table_name: str) -> str:
    """Build a fully qualified public table reference."""
    return f"public.{_quote_identifier(table_name)}"


_AZ_TRANSLATION = str.maketrans(
    {
        "\u0259": "e",
        "\u018f": "e",
        "\u0131": "i",
        "\u0130": "i",
        "\u00f6": "o",
        "\u00d6": "o",
        "\u00fc": "u",
        "\u00dc": "u",
        "\u011f": "g",
        "\u011e": "g",
        "\u00e7": "c",
        "\u00c7": "c",
        "\u015f": "s",
        "\u015e": "s",
    }
)


def _normalize_for_match(value: str) -> str:
    """Normalize free text for simple rule-based matching."""
    folded = str(value).translate(_AZ_TRANSLATION).lower()
    return re.sub(r"[^a-z0-9]+", " ", folded).strip()


def _expanded_tokens(text: str) -> set[str]:
    """Return simple token variants for plural/singular matching."""
    tokens = set(text.split())
    for token in list(tokens):
        if len(token) > 4 and token.endswith("ies"):
            tokens.add(f"{token[:-3]}y")
        if len(token) > 3 and token.endswith("s") and token not in {"sales"}:
            tokens.add(token[:-1])
    return tokens


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    """Return True when any normalized term is present as text."""
    padded = f" {text} "
    return any((term in padded if " " in term else f" {term} " in padded) for term in terms)


def _is_numeric_type(type_name: str) -> bool:
    normalized = type_name.lower()
    numeric_markers = (
        "int",
        "numeric",
        "decimal",
        "double",
        "real",
        "float",
        "money",
    )
    return any(marker in normalized for marker in numeric_markers)


def _is_datetime_type(type_name: str) -> bool:
    normalized = type_name.lower()
    return any(marker in normalized for marker in ("date", "time", "timestamp"))


def _column_names(schema: dict[str, Any]) -> list[str]:
    return [str(column["name"]) for column in schema.get("columns", []) if column.get("name")]


def _columns_by_kind(schema: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    numeric: list[str] = []
    datetime_columns: list[str] = []
    categorical: list[str] = []

    for column in schema.get("columns", []):
        name = str(column.get("name", "")).strip()
        if not name:
            continue
        type_name = str(column.get("type", ""))
        if _is_numeric_type(type_name):
            numeric.append(name)
        elif _is_datetime_type(type_name):
            datetime_columns.append(name)
        else:
            categorical.append(name)

    return numeric, datetime_columns, categorical


def _extract_limit(question: str, default: int = 20) -> int:
    """Extract a small row limit from user text."""
    normalized = _normalize_for_match(question)
    match = re.search(r"\b(?:top|first|last|limit|ilk|son)\s+(\d{1,4})\b", normalized)
    if match:
        return max(1, min(int(match.group(1)), 5000))
    match = re.search(r"\b(\d{1,4})\s+(?:rows|records|setir|netice)\b", normalized)
    if match:
        return max(1, min(int(match.group(1)), 5000))
    return default


def _find_column_by_question(
    question: str,
    columns: list[str],
    preferred_terms: tuple[str, ...] = (),
    *,
    default_to_first: bool = True,
) -> str | None:
    """Pick the best matching column for the question."""
    if not columns:
        return None
    normalized_question = _normalize_for_match(question)
    question_tokens = _expanded_tokens(normalized_question)

    best_column: str | None = None
    best_score = 0
    for column in columns:
        normalized_column = _normalize_for_match(column)
        column_tokens = _expanded_tokens(normalized_column)
        score = 0
        if normalized_column and normalized_column in normalized_question:
            score += 4
        score += len(question_tokens & column_tokens) * 3
        for index, term in enumerate(preferred_terms):
            if not term:
                continue
            term_matches_column = term in column_tokens or term in normalized_column
            if not term_matches_column:
                continue
            score += 3 if index < 3 else 2 if index < 6 else 1
            if term in question_tokens or term in normalized_question:
                score += 5
        if score > best_score:
            best_score = score
            best_column = column

    if best_score > 0:
        return best_column
    return columns[0] if default_to_first else None


def _find_group_column(question: str, categorical: list[str], datetime_columns: list[str]) -> tuple[str, str] | None:
    """Find a grouping expression and alias."""
    normalized_question = _normalize_for_match(question)
    if datetime_columns and _contains_any(normalized_question, ("month", "ay")):
        date_column = _find_column_by_question(question, datetime_columns)
        if date_column:
            expression = f"DATE_TRUNC('month', {_quote_identifier(date_column)})"
            return expression, "month"
    if datetime_columns and _contains_any(normalized_question, ("year", "il")):
        date_column = _find_column_by_question(question, datetime_columns)
        if date_column:
            expression = f"DATE_TRUNC('year', {_quote_identifier(date_column)})"
            return expression, "year"
    if datetime_columns and _contains_any(normalized_question, ("day", "gun", "date", "tarix")):
        date_column = _find_column_by_question(question, datetime_columns)
        if date_column:
            expression = f"DATE_TRUNC('day', {_quote_identifier(date_column)})"
            return expression, "day"

    has_group_hint = _contains_any(
        normalized_question,
        ("by", "per", "each", "group", "which", "breakdown", "uzre", "gore", "her"),
    )
    group_column = _find_column_by_question(
        question,
        categorical,
        preferred_terms=(
            ("product", "category", "customer", "region", "city", "rep", "name", "mehsul", "kateqoriya")
            if has_group_hint
            else ()
        ),
        default_to_first=has_group_hint,
    )
    if group_column:
        return _quote_identifier(group_column), group_column
    return None


def _metric_alias(function_name: str, column_name: str | None = None) -> str:
    if not column_name:
        return "row_count"
    safe_column = _normalize_for_match(column_name).replace(" ", "_") or "value"
    return f"{function_name.lower()}_{safe_column}"[:63]


def _measure_preferred_terms(normalized_question: str) -> tuple[str, ...]:
    """Return metric preference terms tuned to the user's wording."""
    if _contains_any(normalized_question, ("sold", "unit", "units", "quantity", "qty")):
        return ("sold", "unit", "units", "quantity", "qty", "revenue", "sales", "amount", "total")
    if _contains_any(normalized_question, ("salary", "salaries", "maas")):
        return ("salary", "maas", "revenue", "sales", "amount", "total")
    if _contains_any(normalized_question, ("price", "qiymet")):
        return ("price", "unit", "revenue", "sales", "amount", "total")
    return ("revenue", "sales", "amount", "total", "profit", "quantity", "sold", "units", "price", "salary")


def _build_group_query(
    *,
    table_name: str,
    group_expression: str,
    group_alias: str,
    metric_expression: str,
    metric_alias: str,
    limit: int,
    sort_direction: str = "DESC",
) -> str:
    return "\n".join(
        [
            f"SELECT {group_expression} AS {_quote_identifier(group_alias)},",
            f"       {metric_expression} AS {_quote_identifier(metric_alias)}",
            f"FROM {_table_ref(table_name)}",
            f"GROUP BY {group_expression}",
            f"ORDER BY {_quote_identifier(metric_alias)} {sort_direction}",
            f"LIMIT {limit}",
        ]
    )


def _generate_rule_based_sql(question: str, schema: dict[str, Any]) -> SQLGenerationResult | None:
    """Generate SQL for common analysis questions without waiting for an AI provider."""
    table_name = str(schema.get("table", "")).strip()
    if not table_name:
        return None

    all_columns = _column_names(schema)
    if not all_columns:
        return None

    numeric_columns, datetime_columns, categorical_columns = _columns_by_kind(schema)
    normalized_question = _normalize_for_match(question)
    limit = _extract_limit(question)

    asks_average = _contains_any(normalized_question, ("avg", "average", "mean", "orta"))
    asks_count = _contains_any(normalized_question, ("count", "number", "how many", "say", "nece"))
    asks_total = _contains_any(
        normalized_question,
        ("sum", "total", "revenue", "sales", "sold", "amount", "quantity", "cem", "toplam", "satis"),
    )
    asks_lowest = _contains_any(normalized_question, ("lowest", "least", "minimum", "min", "smallest", "az", "asagi"))
    asks_top = _contains_any(normalized_question, ("top", "highest", "most", "maximum", "max", "largest", "cox", "yuksek"))
    asks_list = _contains_any(normalized_question, ("show", "list", "display", "rows", "records", "goster", "siyahi"))

    measure_column = _find_column_by_question(
        question,
        numeric_columns,
        preferred_terms=_measure_preferred_terms(normalized_question),
    )
    group_match = _find_group_column(question, categorical_columns, datetime_columns)

    if asks_count:
        if group_match:
            group_expression, group_alias = group_match
            sql = _build_group_query(
                table_name=table_name,
                group_expression=group_expression,
                group_alias=group_alias,
                metric_expression="COUNT(*)",
                metric_alias="row_count",
                limit=limit,
            )
        else:
            sql = f"SELECT COUNT(*) AS {_quote_identifier('row_count')}\nFROM {_table_ref(table_name)}"
        return SQLGenerationResult(sql=sql, explanation="Generated by the built-in SQL planner.")

    if asks_average and measure_column:
        metric_alias = _metric_alias("avg", measure_column)
        if group_match:
            group_expression, group_alias = group_match
            sql = _build_group_query(
                table_name=table_name,
                group_expression=group_expression,
                group_alias=group_alias,
                metric_expression=f"AVG({_quote_identifier(measure_column)})",
                metric_alias=metric_alias,
                limit=limit,
            )
        else:
            sql = (
                f"SELECT AVG({_quote_identifier(measure_column)}) AS {_quote_identifier(metric_alias)}\n"
                f"FROM {_table_ref(table_name)}"
            )
        return SQLGenerationResult(sql=sql, explanation="Generated by the built-in SQL planner.")

    if (asks_total or (asks_top and group_match)) and group_match:
        group_expression, group_alias = group_match
        if measure_column:
            metric_expression = f"SUM({_quote_identifier(measure_column)})"
            metric_alias = _metric_alias("sum", measure_column)
        else:
            metric_expression = "COUNT(*)"
            metric_alias = "row_count"
        sql = _build_group_query(
            table_name=table_name,
            group_expression=group_expression,
            group_alias=group_alias,
            metric_expression=metric_expression,
            metric_alias=metric_alias,
            limit=limit,
            sort_direction="ASC" if asks_lowest else "DESC",
        )
        return SQLGenerationResult(sql=sql, explanation="Generated by the built-in SQL planner.")

    if (asks_top or asks_lowest) and measure_column:
        sort_direction = "ASC" if asks_lowest else "DESC"
        sql = "\n".join(
            [
                "SELECT *",
                f"FROM {_table_ref(table_name)}",
                f"ORDER BY {_quote_identifier(measure_column)} {sort_direction}",
                f"LIMIT {limit}",
            ]
        )
        return SQLGenerationResult(sql=sql, explanation="Generated by the built-in SQL planner.")

    if asks_list:
        sql = f"SELECT *\nFROM {_table_ref(table_name)}\nLIMIT {limit}"
        return SQLGenerationResult(sql=sql, explanation="Generated by the built-in SQL planner.")

    return None


def _parse_json_response(text: str) -> SQLGenerationResult:
    """Parse the model's JSON response and return a SQLGenerationResult."""
    text = text.strip()
    if not text.startswith("{"):
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            text = match.group(0)

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"AI response was not valid JSON: {text}") from exc

    sql = str(payload.get("sql", "")).strip()
    explanation = str(payload.get("explanation", "")).strip()
    if not sql:
        raise ValueError("AI did not return a SQL query.")
    return SQLGenerationResult(sql=sql, explanation=explanation)


def _build_prompts(question: str, schema: dict[str, Any]) -> tuple[str, str]:
    """Build provider-neutral prompts for SQL generation."""
    table_name = schema["table"]
    system_prompt = f"""
You are a careful PostgreSQL data analyst.
Return only JSON matching this structure:
{{"sql": "SELECT ...", "explanation": "short explanation"}}
Generate exactly one SQL query for the user's question.
Security rules:
- Only generate SELECT statements.
- Never generate DROP, DELETE, UPDATE, INSERT, ALTER, CREATE, TRUNCATE, MERGE, COPY, CALL, EXECUTE, GRANT, REVOKE, VACUUM, ANALYZE, SET, RESET, or SELECT INTO.
- Query only the provided table: public."{table_name}".
- Use double quotes around table and column names.
- Prefer simple aggregations and clear aliases.
- Add a LIMIT when returning detail rows.
""".strip()

    user_prompt = f"""
Question:
{question}

Database schema and sample rows:
{_schema_to_prompt(schema)}
""".strip()
    return system_prompt, user_prompt


def _generate_with_ollama(system_prompt: str, user_prompt: str) -> str:
    """Generate SQL using a free local Ollama model."""
    model = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    timeout_seconds = _env_int("AI_TIMEOUT_SECONDS", 240, minimum=30, maximum=900)
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {"temperature": 0},
    }

    request = urllib.request.Request(
        f"{base_url}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            details = exc.read().decode("utf-8")
        except Exception:
            details = str(exc)
        raise OpenAIConfigurationError(f"Ollama returned an error: {details}") from exc
    except (TimeoutError, socket.timeout) as exc:
        raise AIProviderTimeoutError(
            f"Ollama did not respond within {timeout_seconds} seconds. "
            "Try a smaller OLLAMA_MODEL, increase AI_TIMEOUT_SECONDS, or set AI_PROVIDER=openai."
        ) from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", "")
        if isinstance(reason, (TimeoutError, socket.timeout)) or "timed out" in str(reason).lower():
            raise AIProviderTimeoutError(
                f"Ollama did not respond within {timeout_seconds} seconds. "
                "Try a smaller OLLAMA_MODEL, increase AI_TIMEOUT_SECONDS, or set AI_PROVIDER=openai."
            ) from exc
        raise OpenAIConfigurationError(
            "Ollama is not running. Install Ollama, run 'ollama pull qwen2.5-coder:7b', "
            "then start it with 'ollama serve'."
        ) from exc

    return str(data.get("message", {}).get("content", ""))


def _generate_with_responses_api(client: Any, model: str, system_prompt: str, user_prompt: str) -> str:
    """Generate SQL using the OpenAI Responses API."""
    response = client.responses.create(
        model=model,
        temperature=0,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "sql_generation",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "sql": {"type": "string"},
                        "explanation": {"type": "string"},
                    },
                    "required": ["sql", "explanation"],
                },
                "strict": True,
            }
        },
    )
    return response.output_text


def _generate_with_chat_completions(client: Any, model: str, system_prompt: str, user_prompt: str) -> str:
    """Fallback for environments using older OpenAI SDK/API settings."""
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content or "{}"


def generate_sql_from_question(question: str, schema: dict[str, Any]) -> SQLGenerationResult:
    """Convert a natural-language question into validated PostgreSQL SELECT SQL."""
    table_name = schema["table"]
    provider = os.getenv("AI_PROVIDER", "ollama").strip().lower()
    system_prompt, user_prompt = _build_prompts(question, schema)

    use_rule_based_first = os.getenv("AI_RULE_BASED_FIRST", "false").strip().lower() in {"1", "true", "yes"}
    if use_rule_based_first:
        rule_based_result = _generate_rule_based_sql(question, schema)
        if rule_based_result is not None:
            validate_sql_query(rule_based_result.sql, allowed_table=table_name)
            return rule_based_result

    try:
        if provider == "openai":
            model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
            client = _get_openai_client()
            try:
                raw_text = _generate_with_responses_api(client, model, system_prompt, user_prompt)
            except Exception:
                raw_text = _generate_with_chat_completions(client, model, system_prompt, user_prompt)
        elif provider == "ollama":
            raw_text = _generate_with_ollama(system_prompt, user_prompt)
        else:
            raise OpenAIConfigurationError("AI_PROVIDER must be either 'ollama' or 'openai'.")

        result = _parse_json_response(raw_text)
    except Exception as exc:
        fallback_enabled = os.getenv("AI_FALLBACK_ON_ERROR", "true").strip().lower() not in {"0", "false", "no"}
        rule_based_result = _generate_rule_based_sql(question, schema) if fallback_enabled else None
        if rule_based_result is None:
            raise
        result = SQLGenerationResult(
            sql=rule_based_result.sql,
            explanation=f"{rule_based_result.explanation} AI provider failed first: {exc}",
        )

    validate_sql_query(result.sql, allowed_table=table_name)
    return result
