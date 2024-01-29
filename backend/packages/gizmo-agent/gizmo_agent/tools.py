from enum import Enum

from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools.retriever import create_retriever_tool
from langchain.tools.tavily_search import TavilyAnswer, TavilySearchResults
from langchain.tools import Tool
from langchain.utilities.tavily_search import TavilySearchAPIWrapper
from langchain.vectorstores.redis import RedisFilter
from langchain.tools.google_search import GoogleSearchRun, GoogleSearchResults

from langchain_community.retrievers import WikipediaRetriever
from langchain_community.tools import OpenWeatherMapQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities.google_search import GoogleSearchAPIWrapper

from langchain_experimental.tools import PythonREPLTool

from gizmo_agent.ingest import vstore


class DDGInput(BaseModel):
    query: str = Field(description="search query to look up")


class PythonREPLInput(BaseModel):
    query: str = Field(description="python command to run")


RETRIEVAL_DESCRIPTION = """Can be used to look up information that was uploaded to this assistant.
If the user is referencing particular files, that is often a good hint that information may be here."""


def get_retrieval_tool(assistant_id: str, description: str):
    return create_retriever_tool(
        vstore.as_retriever(
            search_kwargs={"filter": RedisFilter.tag("namespace") == assistant_id}
        ),
        "Retriever",
        description,
    )


def _get_google_search():
    return GoogleSearchResults(api_wrapper=GoogleSearchAPIWrapper())

def _get_duck_duck_go():
    return DuckDuckGoSearchRun(args_schema=DDGInput)


def _get_wikipedia():
    return create_retriever_tool(
        WikipediaRetriever(), "wikipedia", "Search for a query on Wikipedia"
    )


def _get_tavily():
    tavily_search = TavilySearchAPIWrapper()
    return TavilySearchResults(api_wrapper=tavily_search)

def _get_open_weather_map():
    return OpenWeatherMapQueryRun(name="open_weather_map")

def _get_python_repl_tool():
    return PythonREPLTool(args_schema=PythonREPLInput)

def _get_tavily_answer():
    tavily_search = TavilySearchAPIWrapper()
    return TavilyAnswer(api_wrapper=tavily_search)


def _get_ntd_currency_converter():
    return Tool.from_function(
        func=lambda x: x * 30,
        name="ntd_currency_converter",
        description=("Convert currency from USD to NTD. (dollar)\nOnly integer number input is allowd."),
    )


class AvailableTools(str, Enum):
#   DDG_SEARCH = "DDG Search"
#   TAVILY = "Search (Tavily)"
#   TAVILY_ANSWER = "Search (short answer, Tavily)"
    RETRIEVAL = "Retrieval"
    WIKIPEDIA = "Wikipedia"
#   NTD_CURRENCY_CONVERTER = "NTD Currency Converter"
    OPEN_WEATHER_MAP = "Open Weather Map"
    PYTHON_REPL_TOOL = "Python REPL Tool"
    GOOGLE_SEARCH = "Google Search"


TOOLS = {
#   AvailableTools.DDG_SEARCH: _get_duck_duck_go,
    AvailableTools.WIKIPEDIA: _get_wikipedia,
#   AvailableTools.TAVILY_ANSWER: _get_tavily_answer,
#   AvailableTools.TAVILY: _get_tavily,
#   AvailableTools.NTD_CURRENCY_CONVERTER: _get_ntd_currency_converter,
    AvailableTools.OPEN_WEATHER_MAP: _get_open_weather_map,
    AvailableTools.PYTHON_REPL_TOOL: _get_python_repl_tool,
    AvailableTools.GOOGLE_SEARCH: _get_google_search,
}

TOOL_OPTIONS = {e.value: e.value for e in AvailableTools}

# Check if dependencies and env vars for each tool are available
for k, v in TOOLS.items():
    v()
