from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from langchain_core.messages import AIMessage
from langgraph.prebuilt import create_react_agent
from . import tools as T

SYSTEM_PROMPT = """You are a SQL data analysis agent connected to a SQLite database. Answer business questions by exploring the schema and running SQL queries.

Use get_rich_schema first to understand what tables and columns exist, their types, row counts, and sample values. Use get_distinct_values when you need the exact string values in a column before writing a WHERE clause. Use execute_sql to run queries — if it returns a SQL_ERROR, read the error and retry with a corrected query.

SQL rules: write only SQLite-compatible SQL (no ILIKE, no DATE_FORMAT; use strftime for dates). Wrap column names in [square brackets]. Use actual values from sample_values or get_distinct_values in WHERE clauses — never guess. Check _inferred_relationships in the schema to find JOIN columns between tables. Present the final answer clearly with the result table included."""


class NoInput(BaseModel):
    pass


class SQLInput(BaseModel):
    query: str = Field(description="A valid SQLite SQL query to execute")


class DistinctValuesInput(BaseModel):
    table: str = Field(description="Table name")
    column: str = Field(description="Column name to get distinct values for")


class AnalysisAgent:
    def __init__(self, db_path: str, llm):
        self.db_path = db_path
        self.llm = llm
        self.graph = self._build_graph()

    def _build_graph(self):
        agent_self = self

        tools = [
            StructuredTool.from_function(
                name="get_rich_schema",
                description=(
                    "Get the full database schema: all tables, column names, data types, "
                    "row counts, sample values per column, and inferred JOIN relationships. "
                    "Always call this first before writing any SQL."
                ),
                func=lambda: T.get_rich_schema(agent_self.db_path),
                args_schema=NoInput,
            ),
            StructuredTool.from_function(
                name="execute_sql",
                description=(
                    "Execute a SQLite SQL query. Returns results as a markdown table. "
                    "If the result starts with SQL_ERROR, fix the query and try again."
                ),
                func=lambda query: T.execute_sql(agent_self.db_path, query),
                args_schema=SQLInput,
            ),
            StructuredTool.from_function(
                name="get_distinct_values",
                description=(
                    "Get all distinct values and their counts for a specific column in a table. "
                    "Use this before writing WHERE clauses to know the exact values to filter on."
                ),
                func=lambda table, column: T.get_distinct_values(agent_self.db_path, table, column),
                args_schema=DistinctValuesInput,
            ),
        ]

        return create_react_agent(self.llm, tools, prompt=SYSTEM_PROMPT)

    @staticmethod
    def _extract_text(content) -> str:
        if isinstance(content, list):
            parts = [
                b["text"] for b in content
                if isinstance(b, dict) and b.get("type") == "text" and b.get("text")
            ]
            return " ".join(parts).strip()
        return str(content).strip() if content else ""

    def stream_run(self, question: str):
        """Yield events as the agent works, then yield ('done', answer) at the end.

        Event types:
          ('tool_call',  tool_name: str, args: dict)
          ('tool_result', tool_name: str, result: str)
          ('done',        answer: str)
        """
        from langchain_core.messages import ToolMessage as TM

        answer = "Agent completed without a final text response."
        for chunk in self.graph.stream(
            {"messages": [("human", question)]},
            config={"recursion_limit": 25},
            stream_mode="updates",
        ):
            for node_output in chunk.values():
                for msg in node_output.get("messages", []):
                    if isinstance(msg, AIMessage):
                        for tc in msg.tool_calls:
                            yield ("tool_call", tc["name"], tc.get("args", {}))
                        if not msg.tool_calls:
                            text = self._extract_text(msg.content)
                            if text:
                                answer = text
                    elif isinstance(msg, TM):
                        yield ("tool_result", getattr(msg, "name", None) or "tool", str(msg.content))

        yield ("done", answer)

    def run(self, question: str) -> str:
        answer = ""
        for event in self.stream_run(question):
            if event[0] == "done":
                answer = event[1]
        return answer
