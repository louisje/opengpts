import os

from enum import Enum
from functools import lru_cache, partial
from typing import Optional,  List

from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools.retriever import create_retriever_tool
from langchain_community.agent_toolkits.connery import ConneryToolkit
from langchain_community.retrievers.kay import KayAiRetriever
from langchain_community.retrievers.pubmed import PubMedRetriever
from langchain_community.retrievers.wikipedia import WikipediaRetriever
from langchain_community.retrievers.you import YouRetriever
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.tools.connery import ConneryService
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain_google_community import GoogleSearchResults, GoogleSearchRetriever
from langchain_community.tools.openweathermap.tool import OpenWeatherMapQueryRun
from langchain_community.tools.tavily_search import (
    TavilyAnswer as _TavilyAnswer,
)
from langchain_community.tools.tavily_search import (
    TavilySearchResults,
)
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_community.utilities.openweathermap import OpenWeatherMapAPIWrapper
from langchain_community.utilities.google_search import GoogleSearchAPIWrapper

from langchain_core.documents.base import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import Tool
from langchain_experimental.cpal.templates.univariate import query
from langchain_experimental.tools import PythonREPLTool
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_core.tools import Tool
from langchain_robocorp import ActionServerToolkit
from typing_extensions import TypedDict

from app.upload import vstore
from langchain.pydantic_v1 import SecretStr


class DDGInput(BaseModel):
    query: str = Field(description="search query to look up")


class ArxivInput(BaseModel):
    query: str = Field(description="search query to look up")


class PythonREPLInput(BaseModel):
    query: str = Field(description="python command to run")

class GoogleSearchRetrieverInput(BaseModel):
    query: str = Field(description="search query to look up")

class GoogleSearchInput(BaseModel):
    query: str = Field(description="search query to look up")

class OpenWeatherMapInput(BaseModel):
    location: str = Field(description="location to query")


class DallEInput(BaseModel):
    query: str = Field(description="image description to generate image from")


class AvailableTools(str, Enum):
    DDG_SEARCH = "ddg_search"
    TAVILY = "search_tavily"
    TAVILY_ANSWER = "search_tavily_answer"
    RETRIEVAL = "retrieval"
    ARXIV = "arxiv"
    YOU_SEARCH = "you_search"
    WIKIPEDIA = "wikipedia"
    OPEN_WEATHER_MAP = "Open Weather Map"
    PYTHON_REPL_TOOL = "Python REPL Tool"
    GOOGLE_SEARCH = "Google Search"
    DALL_E = "dall_e"


class ToolConfig(TypedDict):
    ...


class BaseTool(BaseModel):
    type: AvailableTools
    name: Optional[str]
    description: Optional[str]
    config: Optional[ToolConfig]
    multi_use: Optional[bool] = False


class ActionServerConfig(ToolConfig):
    url: str
    api_key: str


class DDGSearch(BaseTool):
    type: AvailableTools = Field(AvailableTools.DDG_SEARCH, const=True)
    name: str = Field("DuckDuckGo Search", const=True)
    description: str = Field(
        "Search the web with [DuckDuckGo](https://pypi.org/project/duckduckgo-search/).",
        const=True,
    )


class Arxiv(BaseTool):
    type: AvailableTools = Field(AvailableTools.ARXIV, const=True)
    name: str = Field("Arxiv", const=True)
    description: str = Field("Searches [Arxiv](https://arxiv.org/).", const=True)


class YouSearch(BaseTool):
    type: AvailableTools = Field(AvailableTools.YOU_SEARCH, const=True)
    name: str = Field("You.com Search", const=True)
    description: str = Field(
        "Uses [You.com](https://you.com/) search, optimized responses for LLMs.",
        const=True,
    )


class Wikipedia(BaseTool):
    type: AvailableTools = Field(AvailableTools.WIKIPEDIA, const=True)
    name: str = Field("Wikipedia", const=True)
    description: str = Field(
        "Searches [Wikipedia](https://pypi.org/project/wikipedia/).", const=True
    )

class GoogleSearch(BaseTool):
    type: AvailableTools = Field(AvailableTools.GOOGLE_SEARCH, const=True)
    name: str = Field("Google Search", const=True)
    description: str = Field("Search the web with [Google Search](https://www.google.com/).", const=True)

class Tavily(BaseTool):
    type: AvailableTools = Field(AvailableTools.TAVILY, const=True)
    name: str = Field("Search (Tavily)", const=True)
    description: str = Field(
        (
            "Uses the [Tavily](https://app.tavily.com/) search engine. "
            "Includes sources in the response."
        ),
        const=True,
    )


class TavilyAnswer(BaseTool):
    type: AvailableTools = Field(AvailableTools.TAVILY_ANSWER, const=True)
    name: str = Field("Search (short answer, Tavily)", const=True)
    description: str = Field(
        (
            "Uses the [Tavily](https://app.tavily.com/) search engine. "
            "This returns only the answer, no supporting evidence."
        ),
        const=True,
    )


class Retrieval(BaseTool):
    type: AvailableTools = Field(AvailableTools.RETRIEVAL, const=True)
    name: str = Field("Retrieval", const=True)
    description: str = Field("Look up information in uploaded files.", const=True)


class DallE(BaseTool):
    type: AvailableTools = Field(AvailableTools.DALL_E, const=True)
    name: str = Field("Generate Image (Dall-E)", const=True)
    description: str = Field(
        "Generates images from a text description using OpenAI's DALL-E model.",
        const=True,
    )


RETRIEVAL_DESCRIPTION = """Can be used to look up information that was uploaded to this assistant.
If the user is referencing particular files, that is often a good hint that information may be here.
If the user asks a vague question, they are likely meaning to look up info from this retriever, and you should call it!"""


def get_retriever(assistant_id: str, thread_id: str):
    return vstore.as_retriever(
        search_kwargs={"filter": {"namespace": {"$in": [assistant_id, thread_id]}}}
    )


@lru_cache(maxsize=5)
def get_retrieval_tool(assistant_id: str, thread_id: str, description: str):
    return create_retriever_tool(
        get_retriever(assistant_id, thread_id),
        "Retriever",
        description,
    )


@lru_cache(maxsize=1)
def _get_google_search():
    return GoogleSearchResults(args_schema=GoogleSearchInput, api_wrapper=GoogleSearchAPIWrapper(search_engine=None), num_results=4)

@lru_cache(maxsize=1)
def _get_google_search_retriever():
    retriever: BaseRetriever = GoogleSearchRetriever(api_wrapper=GoogleSearchAPIWrapper(search_engine=None))
    def _get_relevant_documents(
        query: str,
        retriever: BaseRetriever,
    ) -> List[Document]:

        return retriever.get_relevant_documents(query)

    func = partial(
        _get_relevant_documents,
        retriever=retriever,
    )
    return Tool(
        name=retriever.name,
        description=retriever.description,
        func=func,
        args_schema=GoogleSearchRetrieverInput,
    )



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
    return WikipediaRetriever(lang="zh", wiki_client=None, name="Wikipedia", description="Search for a query on Wikipedia")


@lru_cache(maxsize=1)
def _get_tavily():
    tavily_search = TavilySearchAPIWrapper()
    return TavilySearchResults(api_wrapper=tavily_search, name="search_tavily")

def _get_open_weather_map():
    return OpenWeatherMapQueryRun(name="open_weather_map",args_schema=OpenWeatherMapInput, api_wrapper=OpenWeatherMapAPIWrapper())

def _get_python_repl_tool():
    return PythonREPLTool(args_schema=PythonREPLInput)

@lru_cache(maxsize=1)
def _get_tavily_answer():
    tavily_search = TavilySearchAPIWrapper()
    return _TavilyAnswer(api_wrapper=tavily_search, name="search_tavily_answer")


@lru_cache(maxsize=1)
def _get_connery_actions():
    connery_service = ConneryService()
    connery_toolkit = ConneryToolkit.create_instance(connery_service)
    tools = connery_toolkit.get_tools()
    return tools


@lru_cache(maxsize=1)
def _get_dalle_tools():
    return Tool(
        "Dall-E-Image-Generator",
        DallEAPIWrapper(size="1024x1024", quality="hd").run,
        "A wrapper around OpenAI DALL-E API. Useful for when you need to generate images from a text description. Input should be an image description.",
    )


TOOLS = {
    AvailableTools.DDG_SEARCH: _get_duck_duck_go,
    AvailableTools.ARXIV: _get_arxiv,
    AvailableTools.TAVILY: _get_tavily,
    AvailableTools.WIKIPEDIA: _get_wikipedia,
    AvailableTools.TAVILY_ANSWER: _get_tavily_answer,
    AvailableTools.OPEN_WEATHER_MAP: _get_open_weather_map,
    AvailableTools.PYTHON_REPL_TOOL: _get_python_repl_tool,
    AvailableTools.GOOGLE_SEARCH: _get_google_search,
    AvailableTools.DALL_E: _get_dalle_tools,
}
