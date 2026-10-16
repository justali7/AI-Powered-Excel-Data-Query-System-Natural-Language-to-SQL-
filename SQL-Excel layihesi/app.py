"""Streamlit frontend for the AI-powered Excel-to-PostgreSQL analyst app."""

from __future__ import annotations

import os
from datetime import datetime
from io import BytesIO

import pandas as pd
import streamlit as st

from ai_engine import AIProviderTimeoutError, OpenAIConfigurationError, generate_sql_from_question
from database import (
    DatabaseConfigurationError,
    QueryTimeoutError,
    execute_select_query,
    get_table_schema,
    list_tables,
    test_connection,
    upload_dataframe_to_postgres,
)
from utils import (
    build_auto_chart,
    clean_dataframe_for_database,
    format_schema_for_display,
    generate_table_name,
    normalize_dataframe_columns,
    validate_sql_query,
)


st.set_page_config(
    page_title="AI Data Analyst",
    layout="wide",
    initial_sidebar_state="expanded",
)


def initialize_session_state() -> None:
    """Create Streamlit session keys used across pages."""
    defaults = {
        "active_table": None,
        "query_history": [],
        "last_result": None,
        "last_sql": None,
        "last_question": None,
        "last_explanation": None,
        "theme": "Light",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def inject_dashboard_css(theme: str) -> None:
    """Apply a lightweight dashboard style on top of Streamlit components."""
    is_dark = theme == "Dark"
    colors = {
        "bg": "#0E1117" if is_dark else "#F6F8FB",
        "panel": "#161B22" if is_dark else "#FFFFFF",
        "surface": "#1D2430" if is_dark else "#FFFFFF",
        "input": "#0F1623" if is_dark else "#FFFFFF",
        "hover": "#202A38" if is_dark else "#EEF3FA",
        "code": "#0B1220" if is_dark else "#F1F5F9",
        "muted": "#9BA7B4" if is_dark else "#5D6675",
        "text": "#F3F7FB" if is_dark else "#162033",
        "border": "#2A3441" if is_dark else "#E3E8EF",
        "accent": "#3B82F6",
        "accent_soft": "rgba(59, 130, 246, 0.16)",
        "success": "#10B981",
        "danger": "#EF4444",
    }
    st.markdown(
        f"""
        <style>
            :root {{
                color-scheme: {"dark" if is_dark else "light"};
            }}
            html, body, [data-testid="stAppViewContainer"] {{
                background: {colors["bg"]};
            }}
            .stApp {{
                background: {colors["bg"]};
                color: {colors["text"]};
            }}
            .stApp p,
            .stApp li,
            .stApp label,
            .stApp span,
            .stMarkdown,
            [data-testid="stMarkdownContainer"] {{
                color: {colors["text"]};
            }}
            [data-testid="stHeader"] {{
                background: transparent;
            }}
            [data-testid="stSidebar"],
            [data-testid="stSidebar"] > div {{
                background: {colors["panel"]};
                border-right: 1px solid {colors["border"]};
            }}
            [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] span,
            [data-testid="stSidebar"] .stMarkdown {{
                color: {colors["text"]};
            }}
            .main .block-container {{
                padding-top: 1.4rem;
                padding-bottom: 3rem;
                max-width: 1440px;
            }}
            h1, h2, h3 {{
                letter-spacing: 0;
                color: {colors["text"]};
            }}
            .dashboard-card {{
                background: {colors["panel"]};
                border: 1px solid {colors["border"]};
                border-radius: 8px;
                padding: 1rem 1.1rem;
                box-shadow: 0 14px 36px rgba(15, 23, 42, 0.08);
            }}
            .metric-label {{
                color: {colors["muted"]};
                font-size: 0.82rem;
                margin-bottom: 0.2rem;
            }}
            .metric-value {{
                color: {colors["text"]};
                font-size: 1.45rem;
                font-weight: 700;
            }}
            .status-ok {{
                color: {colors["success"]};
                font-weight: 700;
            }}
            .status-bad {{
                color: {colors["danger"]};
                font-weight: 700;
            }}
            .schema-chip {{
                display: inline-block;
                border: 1px solid {colors["border"]};
                background: {colors["accent_soft"]};
                border-radius: 6px;
                padding: 0.2rem 0.45rem;
                margin: 0.15rem;
                font-size: 0.8rem;
            }}
            div[data-testid="stDataFrame"] {{
                border: 1px solid {colors["border"]};
                border-radius: 8px;
                overflow: hidden;
                background: {colors["surface"]};
            }}
            div[data-testid="stTextInput"] input,
            div[data-testid="stTextArea"] textarea,
            div[data-baseweb="select"] > div,
            div[data-baseweb="base-input"] {{
                background-color: {colors["input"]};
                color: {colors["text"]};
                border-color: {colors["border"]};
            }}
            div[data-testid="stTextArea"] textarea::placeholder,
            div[data-testid="stTextInput"] input::placeholder {{
                color: {colors["muted"]};
                opacity: 1;
            }}
            div[data-baseweb="select"] span,
            div[data-baseweb="popover"] li,
            div[data-baseweb="popover"] div {{
                color: {colors["text"]};
            }}
            div[data-baseweb="popover"] {{
                background: {colors["surface"]};
            }}
            div[role="radiogroup"] label,
            [data-testid="stWidgetLabel"] {{
                color: {colors["text"]};
            }}
            div[data-testid="stAlert"] {{
                background: {colors["surface"]};
                color: {colors["text"]};
                border: 1px solid {colors["border"]};
            }}
            div[data-testid="stCodeBlock"] pre,
            code {{
                background: {colors["code"]};
                color: {colors["text"]};
            }}
            .stButton > button {{
                border-radius: 6px;
                border: 1px solid {colors["accent"]};
                background: {colors["accent"]};
                color: #FFFFFF;
                font-weight: 650;
            }}
            .stButton > button:hover {{
                border-color: {colors["accent"]};
                background: {colors["accent"]};
                color: #FFFFFF;
                filter: brightness(0.96);
            }}
            .stButton > button span {{
                color: #FFFFFF;
            }}
            .stButton > button:disabled {{
                background: {colors["hover"]};
                border-color: {colors["border"]};
                color: {colors["muted"]};
            }}
            .stButton > button:disabled span {{
                color: {colors["muted"]};
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def ai_provider_label() -> str:
    """Return the configured AI provider label for the UI."""
    provider = os.getenv("AI_PROVIDER", "ollama").strip().lower()
    if provider == "ollama":
        return "Ollama"
    if provider == "openai":
        return "OpenAI"
    return provider or "AI"


def apply_chart_theme(chart, theme: str):
    """Match Plotly figures to the selected app theme."""
    is_dark = theme == "Dark"
    chart.update_layout(
        template="plotly_dark" if is_dark else "plotly_white",
        paper_bgcolor="#161B22" if is_dark else "#FFFFFF",
        plot_bgcolor="#161B22" if is_dark else "#FFFFFF",
        font={"color": "#F3F7FB" if is_dark else "#162033"},
        margin={"l": 20, "r": 20, "t": 56, "b": 32},
    )
    return chart


def render_metric(label: str, value: str) -> None:
    """Render a compact metric card with custom CSS."""
    st.markdown(
        f"""
        <div class="dashboard-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> tuple[str, bool, str | None]:
    """Render sidebar navigation and database status."""
    st.sidebar.title("AI Data Analyst")
    selected_page = st.sidebar.radio(
        "Navigation",
        ["Upload Data", "Ask AI", "Query History", "Setup Help"],
    )
    st.session_state.theme = st.sidebar.radio("Theme", ["Light", "Dark"], horizontal=True)

    st.sidebar.divider()
    st.sidebar.subheader("Database")
    try:
        db_ok, db_message = test_connection()
    except DatabaseConfigurationError as exc:
        db_ok, db_message = False, str(exc)

    if db_ok:
        st.sidebar.markdown('<span class="status-ok">Connected</span>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<span class="status-bad">Not connected</span>', unsafe_allow_html=True)
    st.sidebar.caption(db_message)

    active_table = st.session_state.get("active_table")
    if active_table:
        st.sidebar.success(f"Active table: {active_table}")
    else:
        st.sidebar.info("Upload or select a table before asking questions.")

    return selected_page, db_ok, db_message


def read_uploaded_excel(uploaded_file: BytesIO, sheet_name: str) -> pd.DataFrame:
    """Read an uploaded Excel sheet into a DataFrame."""
    uploaded_file.seek(0)
    return pd.read_excel(uploaded_file, sheet_name=sheet_name)


def upload_page(db_ok: bool) -> None:
    """Page for uploading Excel files and loading them into PostgreSQL."""
    st.title("Upload Excel Data")
    st.caption("Upload an .xlsx file, preview it, then create a PostgreSQL table automatically.")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        render_metric("Upload format", ".xlsx")
    with col_b:
        render_metric("Database target", "PostgreSQL")
    with col_c:
        render_metric("Processing engine", "Pandas")

    st.write("")
    uploaded_file = st.file_uploader("Choose an Excel workbook", type=["xlsx"])

    if not uploaded_file:
        st.info("Use the sample workbook in sample_data/example_sales.xlsx if you want to test quickly.")
        return

    try:
        workbook = pd.ExcelFile(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read the Excel workbook: {exc}")
        return

    sheet_name = st.selectbox("Sheet", workbook.sheet_names)
    try:
        raw_df = read_uploaded_excel(uploaded_file, sheet_name)
        cleaned_df = clean_dataframe_for_database(raw_df)
        normalized_df, column_map = normalize_dataframe_columns(cleaned_df)
    except Exception as exc:
        st.error(f"Could not process the selected sheet: {exc}")
        return

    if normalized_df.empty:
        st.warning("The selected sheet has no usable rows.")
        return

    default_table_name = generate_table_name(uploaded_file.name)
    table_name = st.text_input("PostgreSQL table name", value=default_table_name)

    with st.expander("Column names that will be stored in PostgreSQL", expanded=False):
        st.dataframe(
            pd.DataFrame(
                [{"Excel column": source, "Database column": target} for source, target in column_map.items()]
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Preview")
    st.dataframe(normalized_df.head(100), use_container_width=True)

    left, right = st.columns([1, 2])
    with left:
        upload_clicked = st.button("Create table and upload data", type="primary", disabled=not db_ok)
    with right:
        if not db_ok:
            st.warning("Fix the database connection in .env before uploading.")

    if upload_clicked:
        try:
            with st.spinner("Creating table and inserting rows into PostgreSQL..."):
                saved_table, row_count = upload_dataframe_to_postgres(normalized_df, table_name)
            st.session_state.active_table = saved_table
            st.success(f"Uploaded {row_count:,} rows to table '{saved_table}'.")
        except Exception as exc:
            st.error(f"Upload failed: {exc}")


def choose_active_table(db_ok: bool) -> str | None:
    """Let the user choose an existing table when no active upload exists."""
    if not db_ok:
        return None

    try:
        tables = list_tables()
    except Exception as exc:
        st.warning(f"Could not list database tables: {exc}")
        return st.session_state.get("active_table")

    if not tables:
        st.info("No database tables were found. Upload an Excel file first.")
        return None

    current = st.session_state.get("active_table")
    default_index = tables.index(current) if current in tables else 0
    selected = st.selectbox("Active table", tables, index=default_index)
    st.session_state.active_table = selected
    return selected


def ai_page(db_ok: bool) -> None:
    """Page for natural-language questions, SQL generation, results, and charts."""
    st.title("Ask AI")
    st.caption("Ask questions in plain English. The app generates safe SELECT SQL and returns a table.")

    table_name = choose_active_table(db_ok)
    if not table_name:
        return

    try:
        schema = get_table_schema(table_name)
    except Exception as exc:
        st.error(f"Could not read schema for '{table_name}': {exc}")
        return

    with st.expander("Table schema", expanded=False):
        st.markdown(format_schema_for_display(schema), unsafe_allow_html=True)

    examples = [
        "Show top 10 highest salaries",
        "Which products sold the most?",
        "Average sales by month",
    ]
    st.caption("Example prompts: " + " | ".join(examples))

    question = st.text_area(
        "Question",
        placeholder="Example: Show total revenue by product category",
        height=100,
    )
    max_rows = st.slider("Maximum rows to return", min_value=10, max_value=5000, value=1000, step=10)

    run_clicked = st.button("Generate and run SQL", type="primary", disabled=not db_ok or not question.strip())

    if run_clicked:
        try:
            with st.spinner(f"Generating SQL with {ai_provider_label()}..."):
                sql_result = generate_sql_from_question(question=question, schema=schema)
                sql = sql_result.sql

            validate_sql_query(sql, allowed_table=table_name)
            st.session_state.last_sql = sql
            st.session_state.last_question = question
            st.session_state.last_explanation = sql_result.explanation

            with st.spinner("Running safe read-only query..."):
                result_df = execute_select_query(sql, allowed_table=table_name, row_limit=max_rows)

            st.session_state.last_result = result_df
            st.session_state.query_history.insert(
                0,
                {
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "table": table_name,
                    "question": question,
                    "sql": sql,
                    "rows": len(result_df),
                },
            )
            st.success(f"Returned {len(result_df):,} rows.")
        except AIProviderTimeoutError as exc:
            st.error(str(exc))
        except OpenAIConfigurationError as exc:
            st.error(str(exc))
        except QueryTimeoutError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Query failed: {exc}")

    if st.session_state.last_sql:
        with st.expander("Generated SQL", expanded=True):
            st.code(st.session_state.last_sql, language="sql")
            if st.session_state.last_explanation:
                st.caption(st.session_state.last_explanation)

    result_df = st.session_state.last_result
    if isinstance(result_df, pd.DataFrame):
        st.subheader("Results")
        st.dataframe(result_df, use_container_width=True, hide_index=True)

        st.subheader("Visualization")
        chart_type = st.radio("Chart type", ["Auto", "Bar", "Line", "Pie"], horizontal=True)
        chart = build_auto_chart(result_df, chart_type=chart_type, question=st.session_state.last_question or "")
        if chart is None:
            st.info("No chart was generated because the result needs at least one useful numeric or categorical column.")
        else:
            st.plotly_chart(apply_chart_theme(chart, st.session_state.theme), use_container_width=True)


def history_page() -> None:
    """Page for recent generated SQL queries."""
    st.title("Query History")
    history = st.session_state.query_history
    if not history:
        st.info("No queries yet. Ask a question after uploading data.")
        return

    history_df = pd.DataFrame(history)
    st.dataframe(history_df, use_container_width=True, hide_index=True)

    selected = st.selectbox(
        "Inspect query",
        options=list(range(len(history))),
        format_func=lambda i: f"{history[i]['time']} - {history[i]['question'][:80]}",
    )
    st.code(history[selected]["sql"], language="sql")


def setup_help_page() -> None:
    """Page with beginner-friendly setup hints inside the app."""
    st.title("Setup Help")
    st.markdown(
        """
        1. Install dependencies with `pip install -r requirements.txt`.
        2. Copy `.env.example` to `.env`.
        3. For free local AI, install Ollama and run `ollama pull qwen2.5-coder:7b`.
        4. Fill in the PostgreSQL settings.
        5. Start PostgreSQL and create the database listed in `DB_NAME`.
        6. Run the app with `streamlit run app.py`.

        The full setup guide is in `README.md`.
        """
    )


def main() -> None:
    """Application entry point."""
    initialize_session_state()
    selected_page, db_ok, _ = render_sidebar()
    inject_dashboard_css(st.session_state.theme)

    if selected_page == "Upload Data":
        upload_page(db_ok)
    elif selected_page == "Ask AI":
        ai_page(db_ok)
    elif selected_page == "Query History":
        history_page()
    else:
        setup_help_page()


if __name__ == "__main__":
    main()
