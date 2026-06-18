# Data Cleaning & Analysis Agent

An AI-powered data pipeline built with **LangGraph**, **Groq LLaMA-4**, and **Streamlit**. Upload any CSV file and two autonomous agents — a **Cleaning Agent** and an **Analysis Agent** — work together to clean your data and answer business questions about it in plain English.

---

## How It Works

Most LLM-based data tools either hand raw data to the model and ask it to guess what's wrong, or generate Python code and blindly execute it. Both approaches are unreliable.

This app takes a different approach:

1. **Detect issues programmatically first** — before the LLM is ever called, a `pre_analyze()` function scans the dataset and produces a concrete list of issues: bad column names, datetime strings, missing values, outliers, duplicates.
2. **Give the LLM a specific task, not a vague one** — the agent receives the detected issue list and is told to fix it, not to figure out what's wrong on its own.
3. **The LLM calls pre-built tools** — the agent never generates or executes Python code. It calls named, typed tools (`rename_columns`, `fill_missing`, `handle_outliers`, etc.) that apply safe, tested transformations.
4. **Store cleaned data in SQLite** — the cleaned DataFrame is written to a per-session SQLite database, which the Analysis Agent queries via SQL.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI                             │
│  Upload CSV → Cleaning Page → Analysis Page                     │
└──────────────────────┬──────────────────┬───────────────────────┘
                       │                  │
          ┌────────────▼──────┐  ┌────────▼────────────┐
          │  Cleaning Agent   │  │  Analysis Agent     │
          │  (LangGraph ReAct)│  │  (LangGraph ReAct)  │
          └────────────┬──────┘  └────────┬────────────┘
                       │                  │
          ┌────────────▼──────┐  ┌────────▼────────────┐
          │  Cleaning Tools   │  │  Analysis Tools     │
          │  (pure Python)    │  │  (SQLite queries)   │
          └────────────┬──────┘  └────────┬────────────┘
                       │                  │
          ┌────────────▼──────────────────▼────────────┐
          │           SQLite Database                   │
          │       (per-session, ephemeral)              │
          └─────────────────────────────────────────────┘
```

### Cleaning Agent

| Component | Detail |
|---|---|
| Framework | `langgraph.prebuilt.create_react_agent` |
| Model | Groq — `meta-llama/llama-4-scout-17b-16e-instruct` |
| State | Mutable `self.df` updated in-place by each tool call |
| Pre-analysis | `pre_analyze(df)` detects issues before the agent runs |
| Tools | `compute_statistics`, `rename_columns`, `fill_missing`, `handle_outliers`, `standardize_values`, `standardize_text`, `convert_dtype`, `strip_whitespace`, `remove_duplicates` |

**Flow:**
```
pre_analyze(df) → issue list → Agent prompt
    → compute_statistics (LLM reads real stats)
    → rename_columns (fixes all column names)
    → convert_dtype (strings → datetime, etc.)
    → fill_missing / handle_outliers / standardize_values
    → remove_duplicates
    → compute_statistics (verify)
    → done
```

### Analysis Agent

| Component | Detail |
|---|---|
| Framework | `langgraph.prebuilt.create_react_agent` |
| Model | Same Groq LLaMA-4 |
| Tools | `get_rich_schema`, `execute_sql`, `get_distinct_values` |

**Flow:**
```
User question (plain English)
    → get_rich_schema (tables, columns, types, sample values, FK hints)
    → get_distinct_values (exact filter values before writing WHERE)
    → execute_sql (run query, retry on error)
    → present result as markdown table + explanation
```

---

## Features

- **Any CSV** — not hardcoded for any specific dataset; column renaming rules apply generically
- **Live streaming UI** — watch every tool call and result as the agent works; collapses into a "Working log" dropdown when done
- **Real EDA statistics** — agent receives actual null counts, value distributions, outlier counts across all rows (not sample rows)
- **Type-safe tools** — every tool has a Pydantic `args_schema`; the LLM cannot pass malformed arguments
- **Safe transformations** — no `exec()` or generated code; all transformations are pre-built Python functions
- **Bool/nullable-int aware** — handles pandas `bool` dtype and nullable `Int64` correctly (no numpy boolean subtract errors)
- **Per-session database** — each session gets its own `databases/database_<timestamp>.db`; no data leaks between users

---

## Project Structure

```
├── app.py                    # Streamlit UI (upload, cleaning, analysis pages)
├── db_manager.py             # SQLite helper (per-session DB creation)
├── cleaning_agent/
│   ├── agent.py              # CleaningAgent — LangGraph ReAct graph + streaming
│   └── tools.py              # Pure Python cleaning functions + pre_analyze()
├── analysis_agent/
│   ├── agent.py              # AnalysisAgent — LangGraph ReAct graph + streaming
│   └── tools.py              # Schema introspection + SQL execution
├── requirements.txt
└── .gitignore
```

---

## Tech Stack

| Layer | Library |
|---|---|
| UI | Streamlit 1.58 |
| Agent framework | LangGraph 1.2.5 (`create_react_agent`) |
| LLM | Groq API — LLaMA-4-Scout 17B |
| LLM client | LangChain-Groq 1.1.3 |
| Data | Pandas 3.0, NumPy 2.4 |
| Database | SQLite (stdlib) |
| Validation | Pydantic 2.13 |

---

## Local Setup

**1. Clone the repository**
```bash
git clone https://github.com/Abhilokesh/data-cleaning-analysis-agent.git
cd data-cleaning-analysis-agent
```

**2. Create a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add your Groq API key**

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get a free key at [console.groq.com](https://console.groq.com).

**5. Run the app**
```bash
streamlit run app.py
```

---

## Deployment (Streamlit Community Cloud)

1. Fork this repo to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **Create app** → select this repo, branch `main`, file `app.py`
4. Open **Advanced settings → Secrets** and add:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   ```
5. Click **Deploy**

---

## Usage

### Cleaning
1. Upload any CSV file on the **Upload** page
2. Optionally add extra instructions (e.g. "fill missing prices with median")
3. Click **Run Cleaning Agent** — watch tool calls stream in real time
4. Review the before/after data preview
5. Download the cleaned CSV or **Add to Database**

### Analysis
1. After adding cleaned data to the database, click **Go to Analysis**
2. Type any question in plain English:
   - *"What is the average sale amount by category?"*
   - *"Which product had the most returns last month?"*
   - *"Show the top 10 customers by total spend."*
3. The agent explores the schema, runs SQL, and returns results as a table

---

## Key Design Decisions

**Why not generate Python code?**
Generated code is unpredictable and can silently corrupt data. Pre-built tools are deterministic, testable, and auditable.

**Why pre-analyze before calling the LLM?**
An LLM reading 30-column statistics to decide what's wrong is slow and unreliable. Detecting obvious issues programmatically (regex, type checks, IQR outliers) and handing the LLM a specific fix list is faster and more accurate.

**Why LangGraph `create_react_agent` instead of LangChain AgentExecutor?**
`AgentExecutor` was removed in LangChain 1.x. LangGraph's `create_react_agent` is the current recommended approach, supports streaming via `graph.stream(..., stream_mode="updates")`, and has a cleaner tool-calling loop.
