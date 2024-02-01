import os
from enum import Enum

from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools.retriever import create_retriever_tool
from langchain.tools import Tool

from langchain_community.retrievers import (
    KayAiRetriever,
    PubMedRetriever,
    WikipediaRetriever,
)
from langchain_community.tools import ArxivQueryRun, DuckDuckGoSearchRun, GoogleSearchResults, OpenWeatherMapQueryRun, DuckDuckGoSearchRun
from langchain_community.retrievers.you import YouRetriever
from langchain_community.tools.tavily_search import TavilyAnswer, TavilySearchResults
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.utilities.google_search import GoogleSearchAPIWrapper
from langchain_community.vectorstores.redis import RedisFilter
from langchain_robocorp import ActionServerToolkit

from langchain_experimental.tools import PythonREPLTool

from app.upload import vstore


class DDGInput(BaseModel):
    query: str = Field(description="search query to look up")


class PythonREPLInput(BaseModel):
    query: str = Field(description="python command to run")


RETRIEVAL_DESCRIPTION = """Can be used to look up information that was uploaded to this assistant.
If the user is referencing particular files, that is often a good hint that information may be here.
If the user asks a vague question, they are likely meaning to look up info from this retriever, and you should call it!"""


def get_retriever(assistant_id: str):
    return vstore.as_retriever(
        search_kwargs={"filter": RedisFilter.tag("namespace") == assistant_id}
    )


def get_retrieval_tool(assistant_id: str, description: str):
    return create_retriever_tool(
        get_retriever(assistant_id),
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


def _get_action_server():
    toolkit = ActionServerToolkit(
        url=os.environ.get("ROBOCORP_ACTION_SERVER_URL"),
        api_key=os.environ.get("ROBOCORP_ACTION_SERVER_KEY"),
    )
    tools = toolkit.get_tools()
    return tools


class AvailableTools(str, Enum):
    ACTION_SERVER = "Action Server by Robocorp"
    DDG_SEARCH = "DDG Search"
    TAVILY = "Search (Tavily)"
    TAVILY_ANSWER = "Search (short answer, Tavily)"
    RETRIEVAL = "Retrieval"
    WIKIPEDIA = "Wikipedia"
    OPEN_WEATHER_MAP = "Open Weather Map"
    PYTHON_REPL_TOOL = "Python REPL Tool"
    GOOGLE_SEARCH = "Google Search"


TOOLS = {
    AvailableTools.ACTION_SERVER: _get_action_server,
    AvailableTools.DDG_SEARCH: _get_duck_duck_go,
    AvailableTools.ARXIV: _get_arxiv,
    AvailableTools.YOU_SEARCH: _get_you_search,
    AvailableTools.SEC_FILINGS: _get_sec_filings,
    AvailableTools.PRESS_RELEASES: _get_press_releases,
    AvailableTools.PUBMED: _get_pubmed,
    AvailableTools.TAVILY: _get_tavily,
    AvailableTools.WIKIPEDIA: _get_wikipedia,
    AvailableTools.TAVILY_ANSWER: _get_tavily_answer,
    AvailableTools.TAVILY: _get_tavily,
    AvailableTools.OPEN_WEATHER_MAP: _get_open_weather_map,
    AvailableTools.PYTHON_REPL_TOOL: _get_python_repl_tool,
    AvailableTools.GOOGLE_SEARCH: _get_google_search,
}

TOOL_OPTIONS = {e.value: e.value for e in AvailableTools}

# Check if dependencies and env vars for each tool are available
for k, v in TOOLS.items():
    v()
