from enum import Enum

from langchain.pydantic_v1 import BaseModel, Field
from langchain.retrievers import WikipediaRetriever
from langchain.tools import OpenWeatherMapQueryRun, DuckDuckGoSearchRun, Tool
from langchain.tools.retriever import create_retriever_tool
from langchain.vectorstores.redis import RedisFilter

from langchain_experimental.tools import PythonREPLTool

from gizmo_agent.ingest import vstore
from gizmo_agent.agent_types import GizmoAgentType


class DDGInput(BaseModel):
    query: str = Field(description="search query to look up")

class PythonREPLInput(BaseModel):
    query: str = Field(description="python command to run")


RETRIEVER_DESCRIPTION = """Can be used to look up information that was uploaded to this assistant.
If the user is referencing particular files, that is often a good hint that information may be here."""


def get_retrieval_tool(assistant_id: str):
    return create_retriever_tool(
        vstore.as_retriever(
            search_kwargs={"filter": RedisFilter.tag("namespace") == assistant_id}
        ),
        "Retriever",
        RETRIEVER_DESCRIPTION,
    )


def _get_duck_duck_go():
    return DuckDuckGoSearchRun(args_schema=DDGInput)


def _get_open_weather_map():
    return OpenWeatherMapQueryRun(name="open_weather_map")


def _get_python_repl_tool():
    return PythonREPLTool(args_schema=PythonREPLInput)


def _get_wikipedia():
    return create_retriever_tool(
        WikipediaRetriever(), "wikipedia", "Search for a query on Wikipedia"
    )


def _get_ntd_currency_converter():
    return Tool.from_function(
        func=lambda x: x * 30,
        name="ntd_currency_converter",
        description=("Convert currency from USD to NTD. (dollar)\nOnly integer number input is allowd."),
    )


class AvailableTools(str, Enum):
    DDG_SEARCH = "DDG Search"
    RETRIEVAL = "Retrieval"
    WIKIPEDIA = "Wikipedia"
    NTD_CURRENCY_CONVERTER = "NTD Currency Converter"
    OPEN_WEATHER_MAP = "Open Weather Map"
    PYTHON_REPL_TOOL = "Python REPL Tool"


TOOLS = {
    AvailableTools.DDG_SEARCH: _get_duck_duck_go,
    AvailableTools.WIKIPEDIA: _get_wikipedia,
    AvailableTools.NTD_CURRENCY_CONVERTER: _get_ntd_currency_converter,
    AvailableTools.OPEN_WEATHER_MAP: _get_open_weather_map,
    AvailableTools.PYTHON_REPL_TOOL: _get_python_repl_tool,
}

TOOL_OPTIONS = {e.value: e.value for e in AvailableTools}

# Check if dependencies and env vars for each tool are available
for k, v in TOOLS.items():
    v()
