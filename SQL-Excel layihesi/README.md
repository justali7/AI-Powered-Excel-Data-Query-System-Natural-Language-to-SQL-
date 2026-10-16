# AI-Powered Excel Data Analyst

A beginner-friendly Streamlit app that uploads Excel files, stores the data in PostgreSQL, converts natural-language questions into safe SQL with free local Ollama AI or optional OpenAI API, and visualizes results with Plotly.

## Features

- Upload `.xlsx` files and preview the data before import.
- Automatically creates a PostgreSQL table from the uploaded sheet.
- Uses Pandas for Excel processing and SQLAlchemy for database access.
- Converts prompts like `Show top 10 highest salaries` into PostgreSQL SELECT queries.
- Works for free with a local Ollama model.
- Blocks destructive SQL such as `DROP`, `DELETE`, `UPDATE`, `INSERT`, and `ALTER`.
- Runs generated queries inside a read-only PostgreSQL transaction.
- Shows result tables, automatic Plotly charts, and query history.
- Includes light and dark dashboard styling.

## Project Structure

```text
.
├── app.py
├── ai_engine.py
├── database.py
├── utils.py
├── requirements.txt
├── .env.example
├── README.md
└── sample_data/
    └── example_sales.xlsx
```

## 1. Install Python Dependencies

Use Python 3.10 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

## 2. Set Up PostgreSQL

Install PostgreSQL locally, then create a database and user.

```sql
CREATE DATABASE ai_excel;
CREATE USER ai_excel_user WITH PASSWORD 'replace_with_a_strong_password';
GRANT ALL PRIVILEGES ON DATABASE ai_excel TO ai_excel_user;
```

Connect to the new database and grant schema permissions:

```sql
\c ai_excel
GRANT USAGE, CREATE ON SCHEMA public TO ai_excel_user;
```

Docker alternative:

```bash
docker run --name ai-excel-postgres -e POSTGRES_DB=ai_excel -e POSTGRES_USER=ai_excel_user -e POSTGRES_PASSWORD=replace_with_a_strong_password -p 5432:5432 -d postgres:16
```

## 3. Configure Environment Variables

Copy the example file:

```bash
copy .env.example .env
```

On macOS or Linux:

```bash
cp .env.example .env
```

Edit `.env`:

```env
AI_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5-coder:7b
OLLAMA_BASE_URL=http://localhost:11434
AI_TIMEOUT_SECONDS=240
AI_FALLBACK_ON_ERROR=true

# Optional paid provider. Only needed if AI_PROVIDER=openai.
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4.1-mini

DB_HOST=localhost
DB_PORT=5432
DB_USER=ai_excel_user
DB_PASSWORD=replace_with_a_strong_password
DB_NAME=ai_excel
SQL_QUERY_TIMEOUT_SECONDS=60
```

## 4. Free Local AI Setup

Install Ollama from:

```text
https://ollama.com/download
```

Then download the local SQL/code-capable model:

```bash
ollama pull qwen2.5-coder:7b
```

Start Ollama if it is not already running:

```bash
ollama serve
```

With this setup, the app uses your own computer for AI generation and does not need an OpenAI API key.

## 5. Optional OpenAI Setup

OpenAI is not required. If you prefer to use OpenAI instead of Ollama, install the optional package:

```bash
pip install openai
```

Then update `.env`:

```env
AI_PROVIDER=openai
OPENAI_API_KEY=sk-your-openai-api-key
OPENAI_MODEL=gpt-4.1-mini
```

## 6. Run the App

```bash
streamlit run app.py
```

Open the local URL shown by Streamlit, usually:

```text
http://localhost:8501
```

## 7. Try the Sample Dataset

Upload:

```text
sample_data/example_sales.xlsx
```

Example prompts:

- `Which products sold the most?`
- `Show revenue by month`
- `Top 10 sales reps by revenue`
- `Average unit price by category`

## Security Notes

The app uses multiple safeguards:

- The OpenAI prompt requests only one PostgreSQL SELECT query.
- `utils.validate_sql_query` rejects non-SELECT statements and dangerous keywords.
- The query must reference the active uploaded table.
- `database.execute_select_query` runs inside a read-only PostgreSQL transaction.
- A row limit is enforced when the AI does not include one.

For production, use separate database roles: one role for uploads and a second read-only role for AI-generated queries.

## Troubleshooting

If the sidebar says the database is not connected:

- Confirm PostgreSQL is running.
- Check `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, and `DB_NAME` in `.env`.
- Confirm the user has permission to create tables in the `public` schema.

If AI queries fail with Ollama:

- Confirm Ollama is installed.
- Run `ollama pull qwen2.5-coder:7b`.
- Run `ollama serve`.
- Confirm `.env` contains `AI_PROVIDER=ollama`.
- If your computer is slow loading the local model, increase `AI_TIMEOUT_SECONDS` or use a smaller Ollama model.
- For common summary questions, the app falls back to a built-in SQL planner when the AI provider times out.

If database queries time out:

- Ask for a grouped summary instead of all detail rows.
- Lower the "Maximum rows to return" slider.
- Increase `SQL_QUERY_TIMEOUT_SECONDS` in `.env` if the database can safely run longer queries.

If AI queries fail with OpenAI:

- Confirm `openai` is installed with `pip install openai`.
- Confirm `OPENAI_API_KEY` is set.
- Try a smaller prompt that mentions a visible column name.
- Check the generated SQL in the app expander.

If Excel upload fails:

- Confirm the file is `.xlsx`, not `.xls` or `.csv`.
- Remove completely blank header rows.
- Make sure the sheet has at least one non-empty row and column.
