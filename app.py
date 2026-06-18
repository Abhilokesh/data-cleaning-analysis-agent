import streamlit as st
import pandas as pd
import os
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from cleaning_agent.agent import CleaningAgent
from cleaning_agent.tools import pre_analyze
from analysis_agent.agent import AnalysisAgent
from db_manager import DatabaseManager

load_dotenv()

st.set_page_config(page_title="Data Cleaning & Analysis Agent", layout="wide")

# ── LLM ───────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_llm():
    api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
    return ChatGroq(
        api_key=api_key,
        model="meta-llama/llama-4-scout-17b-16e-instruct",
    )

llm = get_llm()

# ── Helpers ───────────────────────────────────────────────────────────────────

def slugify(filename: str) -> str:
    name = os.path.splitext(filename)[0].lower()
    return re.sub(r"[^a-z0-9]+", "_", name).strip("_")

def render_log(log: list):
    """Render a stored log list inside an expander or status block."""
    for kind, text in log:
        if kind == "write":
            st.markdown(text)
        elif kind == "caption":
            st.caption(text)

# ── Session state init ────────────────────────────────────────────────────────

DEFAULTS = {
    "page": "upload",
    "raw_df": None,
    "cleaned_df": None,
    "agent_summary": None,
    "table_slug": None,
    "db_manager": None,
    "analysis_result": None,
    "cleaning_log": [],   # persists the cleaning agent's tool-call trace
    "analysis_log": [],   # persists the analysis agent's tool-call trace
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.db_manager is None:
    st.session_state.db_manager = DatabaseManager()

db: DatabaseManager = st.session_state.db_manager

# ── Pages ─────────────────────────────────────────────────────────────────────

def upload_page():
    st.header("Upload Data")
    st.write("Upload a CSV file to get started. The Cleaning Agent will analyse and fix it automatically.")

    uploaded = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.session_state.raw_df = df
        st.session_state.table_slug = slugify(uploaded.name)
        st.session_state.cleaned_df = None
        st.session_state.agent_summary = None
        st.session_state.cleaning_log = []
        st.session_state.analysis_log = []
        st.session_state.page = "cleaning"
        st.rerun()


def cleaning_page():
    st.header("Cleaning Agent")

    raw_df: pd.DataFrame = st.session_state.raw_df

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Raw Data")
        st.dataframe(raw_df.head(10), width='stretch')
        st.caption(f"{raw_df.shape[0]:,} rows × {raw_df.shape[1]} columns")

    if st.session_state.cleaned_df is not None:
        with col_right:
            st.subheader("Cleaned Data")
            st.dataframe(st.session_state.cleaned_df.head(10), width='stretch')
            st.caption(
                f"{st.session_state.cleaned_df.shape[0]:,} rows × "
                f"{st.session_state.cleaned_df.shape[1]} columns"
            )

    instructions = st.text_area(
        "Optional: additional cleaning instructions",
        placeholder="e.g. standardize the status column, fill missing fares with median",
        height=80,
    )

    if st.button("Run Cleaning Agent", type="primary"):
        # Reset log for this fresh run
        st.session_state.cleaning_log = []

        detected_issues = pre_analyze(raw_df)
        prompt = (
            f"Clean this dataset. The following issues were detected automatically:\n"
            f"{detected_issues}\n\n"
            f"Fix every issue listed above. After fixing, call compute_statistics to verify."
        )
        if instructions.strip():
            prompt += f"\n\nAdditional user instructions: {instructions.strip()}"

        # Live status block — open while running, collapses when done
        with st.status("Cleaning agent is working…", expanded=True) as status:
            header = f"**Detected issues:**\n{detected_issues}"
            st.caption(header)
            st.session_state.cleaning_log.append(("caption", header))

            agent = CleaningAgent(raw_df, llm)
            for event in agent.stream_run(prompt):
                if event[0] == "tool_call":
                    _, name, args = event
                    args_preview = (
                        ", ".join(f"{k}={repr(v)[:60]}" for k, v in args.items())
                        if args else ""
                    )
                    text = f"**→ `{name}`**" + (f" ({args_preview})" if args_preview else "")
                    st.session_state.cleaning_log.append(("write", text))
                    st.write(text)
                elif event[0] == "tool_result":
                    _, name, result = event
                    preview = result[:250] + ("…" if len(result) > 250 else "")
                    text = f"↳ {preview}"
                    st.session_state.cleaning_log.append(("caption", text))
                    st.caption(text)
                elif event[0] == "done":
                    _, cleaned_df, summary = event
                    st.session_state.cleaned_df = cleaned_df
                    st.session_state.agent_summary = summary

            status.update(label="Cleaning complete ✓", state="complete")

        st.rerun()

    # ── Persistent collapsed log (shown on every render after a run) ──────────
    if st.session_state.cleaning_log:
        with st.expander("Working log — click to see what the agent did", expanded=False):
            render_log(st.session_state.cleaning_log)

    # ── Post-cleaning results & actions ───────────────────────────────────────
    if st.session_state.cleaned_df is not None:
        cleaned_df = st.session_state.cleaned_df

        if st.session_state.agent_summary:
            with st.expander("Agent summary", expanded=False):
                st.write(st.session_state.agent_summary)

        st.divider()
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Add to Database"):
                table_name = st.session_state.table_slug + "_cleaned"
                if db.add_table(cleaned_df, table_name):
                    st.success(f"Added as table: **{table_name}**")
                else:
                    st.error("Failed to add to database.")
        with c2:
            st.download_button(
                "Download Cleaned CSV",
                data=cleaned_df.to_csv(index=False),
                file_name=f"{st.session_state.table_slug}_cleaned.csv",
                mime="text/csv",
            )
        with c3:
            if st.button("Go to Analysis →"):
                if db.is_empty():
                    st.warning("Add the cleaned data to the database first.")
                else:
                    st.session_state.page = "analysis"
                    st.rerun()


def analysis_page():
    st.header("Analysis Agent")

    nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 4])
    with nav_col1:
        if st.button("← Back"):
            st.session_state.page = "upload"
            st.session_state.analysis_result = None
            st.session_state.analysis_log = []
            st.rerun()
    with nav_col2:
        if st.button("Clear DB"):
            db.clear()
            st.success("Database cleared.")
            st.rerun()

    with st.expander("Add a cleaned CSV directly to the database"):
        direct = st.file_uploader("Upload cleaned CSV", type="csv", key="direct_upload")
        if direct:
            df = pd.read_csv(direct)
            table_name = slugify(direct.name) + "_cleaned"
            st.write(f"Preview ({df.shape[0]:,} rows × {df.shape[1]} columns)")
            st.dataframe(df.head(5))
            if st.button("Add to Database", key="direct_add"):
                if db.add_table(df, table_name):
                    st.success(f"Added as: **{table_name}**")
                    st.rerun()
                else:
                    st.error("Failed to add.")

    tables = db.get_tables()
    if not tables:
        st.warning("No tables in the database. Go back and add cleaned data first.")
        return

    st.subheader("Available Tables")
    st.write(", ".join(f"`{t}`" for t in tables))
    st.divider()

    question = st.text_area(
        "Ask a question about your data in plain English:",
        placeholder="e.g. Which airline had the most cancellations? What is the average fare by class?",
        height=80,
    )

    if st.button("Run Analysis", type="primary"):
        if not question.strip():
            st.error("Please enter a question.")
            return

        # Reset log and result for this fresh run
        st.session_state.analysis_log = []
        st.session_state.analysis_result = None

        with st.status("Analysis agent is working…", expanded=True) as status:
            agent = AnalysisAgent(db.db_path, llm)
            for event in agent.stream_run(question.strip()):
                if event[0] == "tool_call":
                    _, name, args = event
                    args_preview = (
                        ", ".join(f"{k}={repr(v)[:80]}" for k, v in args.items())
                        if args else ""
                    )
                    text = f"**→ `{name}`**" + (f" ({args_preview})" if args_preview else "")
                    st.session_state.analysis_log.append(("write", text))
                    st.write(text)
                elif event[0] == "tool_result":
                    _, name, result = event
                    preview = result[:300] + ("…" if len(result) > 300 else "")
                    text = f"↳ {preview}"
                    st.session_state.analysis_log.append(("caption", text))
                    st.caption(text)
                elif event[0] == "done":
                    _, answer = event
                    st.session_state.analysis_result = answer

            status.update(label="Analysis complete ✓", state="complete")

        st.rerun()

    # ── Persistent collapsed log ───────────────────────────────────────────────
    if st.session_state.analysis_log:
        with st.expander("Working log — click to see what the agent did", expanded=False):
            render_log(st.session_state.analysis_log)

    # ── Result ────────────────────────────────────────────────────────────────
    if st.session_state.analysis_result:
        st.subheader("Result")
        st.markdown(st.session_state.analysis_result)


# ── Router ────────────────────────────────────────────────────────────────────

st.title("Data Cleaning & Analysis Agent")

page = st.session_state.page
if page == "upload":
    upload_page()
elif page == "cleaning":
    cleaning_page()
elif page == "analysis":
    analysis_page()
