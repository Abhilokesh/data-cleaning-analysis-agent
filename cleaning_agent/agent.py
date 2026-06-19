import pandas as pd
from typing import Optional, Tuple
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from langchain_core.messages import AIMessage
from langgraph.prebuilt import create_react_agent
from . import tools as T

SYSTEM_PROMPT = """You are a data cleaning agent. Use the available tools to analyze and clean a pandas DataFrame thoroughly.

Start by calling compute_statistics to get real statistics on the entire dataset — actual null counts, value distributions for every column, and outlier counts for numeric columns.

Once you have the statistics, clean the data in this order:

Column names — rename any column that is not clean lowercase_underscore. This is always required. Apply these rules to every column name: (1) special characters must be spelled out — "order#" → "order_number", "pct%" → "percent"; (2) abbreviations must be expanded — "qty" → "quantity", "amt" → "amount", "cd" → "code", "nm" → "name", "dt" → "date", "tm" → "time", "ent" → "entertainment", "dept" → "department", "cust" → "customer"; (3) misspellings must be corrected — fix any word with missing vowels or wrong letters; (4) CamelCase and UPPERCASE must become snake_case — "CustomerID" → "customer_id", "TOTAL_AMT" → "total_amount"; (5) single-letter or cryptic suffixes must be expanded — "_id" is fine if it means identifier, but "_cd", "_nm", "_dt", "_tm" must be expanded. After renaming, use only the new names for all subsequent tool calls.

Data types — any column whose sample values look like "2023-06-26 00:11:20" is a datetime stored as a string; convert it with dtype="datetime". Integer-like float columns should be converted to int.

Categorical consistency — read value_counts for every string column. If you see the same value written differently (different casing, abbreviations, inconsistent prefixes), standardize them.

Missing values — for any column where null_count > 0, fill with an appropriate strategy.

Outliers — for any numeric column where outlier_count > 0, handle them with clip or fill_median.

Duplicates — always call remove_duplicates.

Verify — call compute_statistics again at the end to confirm column names are clean, dtypes are correct, and no issues remain.

When all cleaning steps are done, you MUST call finish_task with a brief summary of what was done. Never return a plain text response — always end by calling finish_task.

Extra rules: never guess column names; if a tool returns an error, fix the parameters and retry; a dataset is not clean just because it has no nulls."""


class NoInput(BaseModel):
    pass


class FinishInput(BaseModel):
    summary: str = Field(description="Brief summary of all cleaning steps performed and the final dataset state")


class RenameInput(BaseModel):
    mapping: dict = Field(description="Dict mapping old column names to new column names, e.g. {'Old Name': 'new_name'}")


class FillMissingInput(BaseModel):
    column: str = Field(description="Exact column name from compute_statistics output")
    strategy: str = Field(description="One of: mean, median, mode, constant, forward_fill")
    fill_value: Optional[str] = Field(default=None, description="Required when strategy='constant'")


class StandardizeValuesInput(BaseModel):
    column: str = Field(description="Exact column name")
    mapping: dict = Field(description="Dict mapping inconsistent values to correct ones, e.g. {'CANC': 'Cancelled'}")


class ConvertDtypeInput(BaseModel):
    column: str = Field(description="Exact column name")
    dtype: str = Field(description="One of: datetime, int, float, str")


class StandardizeTextInput(BaseModel):
    column: str = Field(description="Exact column name")
    case: str = Field(description="One of: upper, lower, title, strip")


class OutlierInput(BaseModel):
    column: str = Field(description="Exact column name (must be numeric)")
    method: str = Field(description="One of: clip (cap at bounds), fill_median (replace with median)")
    lower: Optional[float] = Field(default=None, description="Lower bound. If omitted, IQR lower bound is used.")
    upper: Optional[float] = Field(default=None, description="Upper bound. If omitted, IQR upper bound is used.")


class ColumnInput(BaseModel):
    column: str = Field(description="Exact column name")


class CleaningAgent:
    def __init__(self, df: pd.DataFrame, llm):
        self.df = df.copy()
        self.llm = llm
        self._final_summary = ""
        self.graph = self._build_graph()

    def _apply(self, fn, **kwargs) -> str:
        updated_df, message = fn(self.df, **kwargs)
        self.df = updated_df
        return message

    def _set_summary(self, summary: str) -> str:
        self._final_summary = summary
        return "Task complete."

    def _build_graph(self):
        agent_self = self

        tools = [
            StructuredTool.from_function(
                name="compute_statistics",
                description=(
                    "Compute comprehensive statistics on the FULL dataframe: null counts, "
                    "data types, value distributions for categoricals, outlier counts for numerics. "
                    "Call this first, and again after cleaning to verify."
                ),
                func=lambda: T.compute_statistics(agent_self.df),
                args_schema=NoInput,
            ),
            StructuredTool.from_function(
                name="rename_columns",
                description="Rename columns. Provide a mapping of {old_name: new_name} for every column you want to rename.",
                func=lambda mapping: agent_self._apply(T.rename_columns, mapping=mapping),
                args_schema=RenameInput,
            ),
            StructuredTool.from_function(
                name="fill_missing",
                description="Fill null/missing values in a column using a strategy.",
                func=lambda column, strategy, fill_value=None: agent_self._apply(
                    T.fill_missing, column=column, strategy=strategy, fill_value=fill_value
                ),
                args_schema=FillMissingInput,
            ),
            StructuredTool.from_function(
                name="standardize_values",
                description=(
                    "Map specific values in a column to standardized versions. "
                    "Use this to fix inconsistent categorical values, e.g. {'CANC': 'Cancelled'}."
                ),
                func=lambda column, mapping: agent_self._apply(
                    T.standardize_values, column=column, mapping=mapping
                ),
                args_schema=StandardizeValuesInput,
            ),
            StructuredTool.from_function(
                name="remove_duplicates",
                description="Remove exact duplicate rows from the dataframe.",
                func=lambda: agent_self._apply(T.remove_duplicates),
                args_schema=NoInput,
            ),
            StructuredTool.from_function(
                name="convert_dtype",
                description="Convert a column to a different data type (datetime, int, float, str).",
                func=lambda column, dtype: agent_self._apply(
                    T.convert_dtype, column=column, dtype=dtype
                ),
                args_schema=ConvertDtypeInput,
            ),
            StructuredTool.from_function(
                name="standardize_text",
                description="Standardize text case in a string column (upper, lower, title, strip).",
                func=lambda column, case: agent_self._apply(
                    T.standardize_text, column=column, case=case
                ),
                args_schema=StandardizeTextInput,
            ),
            StructuredTool.from_function(
                name="handle_outliers",
                description=(
                    "Handle outliers in a numeric column. "
                    "'clip' caps values at bounds; 'fill_median' replaces outliers with the median. "
                    "If lower/upper are omitted, IQR bounds are used automatically."
                ),
                func=lambda column, method, lower=None, upper=None: agent_self._apply(
                    T.handle_outliers, column=column, method=method, lower=lower, upper=upper
                ),
                args_schema=OutlierInput,
            ),
            StructuredTool.from_function(
                name="strip_whitespace",
                description="Strip leading and trailing whitespace from all values in a string column.",
                func=lambda column: agent_self._apply(T.strip_whitespace, column=column),
                args_schema=ColumnInput,
            ),
            StructuredTool.from_function(
                name="finish_task",
                description="Call this when all cleaning steps are complete. Pass a brief summary of what was done.",
                func=lambda summary: agent_self._set_summary(summary),
                args_schema=FinishInput,
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

    def stream_run(self, instruction: str = "Clean this dataset thoroughly."):
        """Yield events as the agent works, then yield ('done', df, summary) at the end.

        Event types:
          ('tool_call',  tool_name: str, args: dict)
          ('tool_result', tool_name: str, result: str)
          ('done',        df: DataFrame,  summary: str)
        """
        from langchain_core.messages import ToolMessage as TM

        for chunk in self.graph.stream(
            {"messages": [("human", instruction)]},
            config={"recursion_limit": 80},
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
                                self._final_summary = text
                    elif isinstance(msg, TM):
                        yield ("tool_result", getattr(msg, "name", None) or "tool", str(msg.content))

        yield ("done", self.df, self._final_summary)

    def run(self, instruction: str = "Clean this dataset thoroughly.") -> Tuple[pd.DataFrame, str]:
        df, summary = None, ""
        for event in self.stream_run(instruction):
            if event[0] == "done":
                df, summary = event[1], event[2]
        return df, summary
