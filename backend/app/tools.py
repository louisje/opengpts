import os
from enum import Enum
from functools import lru_cache

from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools.retriever import create_retriever_tool
from langchain_community.agent_toolkits.connery import ConneryToolkit
from langchain_community.retrievers import (
    KayAiRetriever,
    PubMedRetriever,
    WikipediaRetriever,
)
from langchain_community.retrievers.you import YouRetriever
from langchain_community.tools.google_search import GoogleSearchResults
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.tools.openweathermap import OpenWeatherMapQueryRun
from langchain_community.tools.connery import ConneryService
from langchain_community.tools.tavily_search import TavilyAnswer, TavilySearchResults
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.utilities.google_search import GoogleSearchAPIWrapper
from langchain_community.vectorstores.redis import RedisFilter

from langchain_experimental.tools import PythonREPLTool

from app.upload import vstore
from langchain.pydantic_v1 import SecretStr


class DDGInput(BaseModel):
    query: str = Field(description="search query to look up")


class ArxivInput(BaseModel):
    query: str = Field(description="search query to look up")


class PythonREPLInput(BaseModel):
    query: str = Field(description="python command to run")

class GoogleSearchInput(BaseModel):
    query: str = Field(description="search query to look up")


RETRIEVAL_DESCRIPTION = """Can be used to look up information that was uploaded to this assistant.
If the user is referencing particular files, that is often a good hint that information may be here.
If the user asks a vague question, they are likely meaning to look up info from this retriever, and you should call it!"""


def get_retriever(assistant_id: str):
    return vstore.as_retriever(
        search_kwargs={"filter": RedisFilter.tag("namespace") == assistant_id}
    )


@lru_cache(maxsize=5)
def get_retrieval_tool(assistant_id: str, description: str):
    return create_retriever_tool(
        get_retriever(assistant_id),
        "Retriever",
        description,
    )


@lru_cache(maxsize=1)
def _get_google_search():
    return GoogleSearchResults(args_schema=GoogleSearchInput, api_wrapper=GoogleSearchAPIWrapper(search_engine=None))


@lru_cache(maxsize=1)
def _get_duck_duck_go():
    return DuckDuckGoSearchRun(args_schema=DDGInput)


@lru_cache(maxsize=1)
def _get_arxiv():
    return ArxivQueryRun(api_wrapper=ArxivAPIWrapper(arxiv_search=None, arxiv_exceptions=[]), args_schema=ArxivInput)


@lru_cache(maxsize=1)
def _get_you_search():
    return create_retriever_tool(
        YouRetriever(),
        "you_search",
        "Searches for documents using You.com",
    )


@lru_cache(maxsize=1)
def _get_sec_filings():
    return create_retriever_tool(
        KayAiRetriever.create(
            dataset_id="company", data_types=["10-K", "10-Q"], num_contexts=3
        ),
        "sec_filings_search",
        "Search for a query among SEC Filings",
    )


@lru_cache(maxsize=1)
def _get_press_releases():
    return create_retriever_tool(
        KayAiRetriever.create(
            dataset_id="company", data_types=["PressRelease"], num_contexts=6
        ),
        "press_release_search",
        "Search for a query among press releases from US companies",
    )


@lru_cache(maxsize=1)
def _get_pubmed():
    return create_retriever_tool(
        PubMedRetriever(parse=True), "pub_med_search", "Search for a query on PubMed"
    )


@lru_cache(maxsize=1)
def _get_wikipedia():
    return create_retriever_tool(
        WikipediaRetriever(lang="zh", wiki_client=None), "wikipedia", "Search for a query on Wikipedia"
    )


@lru_cache(maxsize=1)
def _get_tavily():
    tavily_search = TavilySearchAPIWrapper(tavily_api_key=SecretStr(os.environ["TAVILY_API_KEY"]))
    return TavilySearchResults(api_wrapper=tavily_search)

def _get_open_weather_map():
    return OpenWeatherMapQueryRun(name="open_weather_map")

def _get_python_repl_tool():
    return PythonREPLTool(args_schema=PythonREPLInput)

@lru_cache(maxsize=1)
def _get_tavily_answer():
    tavily_search = TavilySearchAPIWrapper(tavily_api_key=SecretStr(os.environ["TAVILY_API_KEY"]))
    return TavilyAnswer(api_wrapper=tavily_search)


@lru_cache(maxsize=1)
def _get_connery_actions():
    connery_service = ConneryService(
        runner_url=os.environ.get("CONNERY_RUNNER_URL"),
        api_key=os.environ.get("CONNERY_RUNNER_API_KEY"),
    )
    connery_toolkit = ConneryToolkit.create_instance(connery_service)
    tools = connery_toolkit.get_tools()
    return tools


class AvailableTools(str, Enum):
    CONNERY = '"AI Action Runner" by Connery'
    DDG_SEARCH = "DDG Search"
    TAVILY = "Search (Tavily)"
    TAVILY_ANSWER = "Search (short answer, Tavily)"
    RETRIEVAL = "Retrieval"
    ARXIV = "Arxiv"
    WIKIPEDIA = "Wikipedia"
    OPEN_WEATHER_MAP = "Open Weather Map"
    PYTHON_REPL_TOOL = "Python REPL Tool"
    GOOGLE_SEARCH = "Google Search"


TOOLS = {
    AvailableTools.DDG_SEARCH: _get_duck_duck_go,
    AvailableTools.ARXIV: _get_arxiv,
    AvailableTools.TAVILY: _get_tavily,
    AvailableTools.WIKIPEDIA: _get_wikipedia,
    AvailableTools.TAVILY_ANSWER: _get_tavily_answer,
    AvailableTools.OPEN_WEATHER_MAP: _get_open_weather_map,
    AvailableTools.PYTHON_REPL_TOOL: _get_python_repl_tool,
    AvailableTools.GOOGLE_SEARCH: _get_google_search,
}

TOOL_OPTIONS = {e.value: e.value for e in AvailableTools}

# Check if dependencies and env vars for each tool are available
for k, v in TOOLS.items():
    # Connery requires env vars to be valid even if the tool isn't used,
    # so we'll skip the check for it
    if k != AvailableTools.CONNERY:
        v()
