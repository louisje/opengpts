import pickle
from enum import Enum
import os
from typing import Any, Dict, Mapping, Optional, Sequence, Union

from langgraph.graph.graph import CompiledGraph
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import AnyMessage
from langchain_core.runnables import (
    ConfigurableField,
    RunnableBinding,
)
from langgraph.checkpoint import CheckpointAt
from langgraph.graph.message import Messages
from langgraph.pregel import Pregel

from app.agent_types.ffm_agent import get_ffm_agent_executor
from app.agent_types.tools_agent import get_tools_agent_executor
from app.chatbot import get_chatbot_executor
from app.checkpoint import PostgresCheckpoint
from app.retrieval import get_retrieval_executor
from app.llms import (
    get_anthropic_llm,
    get_google_llm,
    get_mixtral_fireworks,
    get_openai_llm,
    get_ffm_llm,
    get_ollama_llm,
)
from app.tools import (
    RETRIEVAL_DESCRIPTION,
    TOOLS,
    Arxiv,
    AvailableTools,
    DallE,
    DDGSearch,
    Retrieval,
    Tavily,
    TavilyAnswer,
    Wikipedia,
    GoogleSearch,
    get_retrieval_tool,
    get_retriever,
)

from langchain_core.tools import BaseTool

OLLAMA_MODEL_NAME = os.environ["OLLAMA_MODEL"]
FFM_MODEL_NAME = os.environ["FFM_MODEL"]
GEMINI_MODEL_NAME = os.environ["GEMINI_MODEL"]
GPT_35_TURBO_MODEL_NAME = os.environ["GPT_35_TURBO_MODEL"]
MISTRAL_MODEL_NAME = os.environ["MISTRAL_MODEL"]
CLAUDE_MODEL_NAME = os.environ["CLAUDE_MODEL"]

Tool = Union[
    DDGSearch,
    Arxiv,
    Wikipedia,
    Tavily,
    TavilyAnswer,
    Retrieval,
    GoogleSearch,
    DallE,
]

class AgentType(str, Enum):
    GPT_35_TURBO = f"{GPT_35_TURBO_MODEL_NAME} (FreeDuckDuckGo)"
    CLAUDE = f"{CLAUDE_MODEL_NAME} (FreeDuckDuckGo)"
    GEMINI = f"{GEMINI_MODEL_NAME} (Google)"
    MISTRAL = f"{MISTRAL_MODEL_NAME} (Mistral)"
    FFM = f"{FFM_MODEL_NAME} (FFM)"
    OLLAMA = f"{OLLAMA_MODEL_NAME} (Ollama)"


DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant."

CHECKPOINTER = PostgresCheckpoint(serde=pickle, at=CheckpointAt.END_OF_STEP)


def get_agent_executor(
    tools: list[BaseTool],
    agent: AgentType,
    system_message: str,
    interrupt_before_action: bool,
) -> CompiledGraph:
    if agent == AgentType.GPT_35_TURBO:
        llm = get_openai_llm()
        return get_tools_agent_executor(
            tools, llm, system_message, interrupt_before_action, CHECKPOINTER
        )
    elif agent == AgentType.CLAUDE:
        llm = get_openai_llm()
        return get_tools_agent_executor(
             tools, llm, system_message, interrupt_before_action, CHECKPOINTER
        )
    elif agent == AgentType.GEMINI:
        llm = get_google_llm()
        return get_tools_agent_executor(
            tools, llm, system_message, interrupt_before_action, CHECKPOINTER
        )
    elif agent == AgentType.MISTRAL:
       llm = get_mixtral_fireworks()
       return get_tools_agent_executor(
           tools, llm, system_message, interrupt_before_action, CHECKPOINTER
       )
    elif agent == AgentType.FFM:
        llm = get_ffm_llm(model=FFM_MODEL_NAME)
        return get_ffm_agent_executor(
            tools, llm, system_message, interrupt_before_action, CHECKPOINTER
        )
    elif agent == AgentType.OLLAMA:
        llm = get_ollama_llm()
        return get_tools_agent_executor(
            tools, llm, system_message, interrupt_before_action, CHECKPOINTER
        )
    else:
        raise ValueError(f"Unexpected agent type {agent}")


class ConfigurableAgent(RunnableBinding):
    tools: Sequence[Tool]
    agent: AgentType
    system_message: str = DEFAULT_SYSTEM_MESSAGE
    retrieval_description: str = RETRIEVAL_DESCRIPTION
    interrupt_before_action: bool = False
    assistant_id: Optional[str] = None
    thread_id: Optional[str] = None
    user_id: Optional[str] = None

    def __init__(
        self,
        *,
        tools: Sequence[Tool],
        agent: AgentType = AgentType.GPT_35_TURBO,
        system_message: str = DEFAULT_SYSTEM_MESSAGE,
        assistant_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        retrieval_description: str = RETRIEVAL_DESCRIPTION,
        interrupt_before_action: bool = False,
        kwargs: Optional[Mapping[str, Any]] = None,
        config: Optional[RunnableConfig] = None,
        **others: Any,
    ) -> None:
        others.pop("bound", None)
        _tools = []
        for _tool in tools:
            if _tool["type"] == AvailableTools.RETRIEVAL: # type: ignore
                if assistant_id is None or thread_id is None:
                    raise ValueError(
                        "Both assistant_id and thread_id must be provided if Retrieval tool is used"
                    )
                _tools.append(
                    get_retrieval_tool(assistant_id, thread_id, retrieval_description)
                )
            else:
                tool_config = _tool["config"] # type: ignore
                _returned_tools = TOOLS[_tool["type"]](**tool_config) # type: ignore
                if isinstance(_returned_tools, list):
                    _tools.extend(_returned_tools)
                else:
                    _tools.append(_returned_tools)
        _agent = get_agent_executor(
            _tools, agent, system_message, interrupt_before_action
        )
        agent_executor = _agent.with_config({"recursion_limit": 50})
        super().__init__(
            tools=tools,  # type: ignore
            agent=agent,  # type: ignore
            system_message=system_message,  # type: ignore
            retrieval_description=retrieval_description,  # type: ignore
            bound=agent_executor,
            kwargs=kwargs or {},
            config=config or {},
        )


class LLMType(str, Enum):
    GPT_35_TURBO = f"{GPT_35_TURBO_MODEL_NAME} (FreeDuckDuckGo)"
    CLAUDE = f"{CLAUDE_MODEL_NAME} (FreeDuckDuckGo)"
    GEMINI = f"{GEMINI_MODEL_NAME} (Google)"
    MISTRAL = f"{MISTRAL_MODEL_NAME} (Mistral)"
    FFM = f"{FFM_MODEL_NAME} (FFM)"
    OLLAMA = f"{OLLAMA_MODEL_NAME} (Ollama)"


def get_chatbot(
    llm_type: LLMType,
    system_message: str,
):
    if llm_type == LLMType.GPT_35_TURBO:
        llm = get_openai_llm()
    elif llm_type == LLMType.CLAUDE:
        llm = get_openai_llm()
    elif llm_type == LLMType.GEMINI:
        llm = get_google_llm()
    elif llm_type == LLMType.MISTRAL:
        llm = get_mixtral_fireworks()
    elif llm_type == LLMType.FFM:
        llm = get_ffm_llm(model=FFM_MODEL_NAME)
    elif llm_type == LLMType.OLLAMA:
        llm = get_ollama_llm()
    else:
        raise ValueError(f"Unexpected llm type {llm_type}")
    return get_chatbot_executor(llm, system_message, CHECKPOINTER)


class ConfigurableChatBot(RunnableBinding):
    llm: LLMType
    system_message: str = DEFAULT_SYSTEM_MESSAGE
    user_id: Optional[str] = None

    def __init__(
        self,
        *,
        llm: LLMType = LLMType.GPT_35_TURBO,
        system_message: str = DEFAULT_SYSTEM_MESSAGE,
        kwargs: Optional[Mapping[str, Any]] = None,
        config: Optional[RunnableConfig] = None,
        **others: Any,
    ) -> None:
        others.pop("bound", None)

        chatbot = get_chatbot(llm, system_message)
        super().__init__(
            llm=llm,  # type: ignore
            system_message=system_message,  # type: ignore
            bound=chatbot,
            kwargs=kwargs or {},
            config=config or {},
        )


chatbot = (
    ConfigurableChatBot(llm=LLMType.GPT_35_TURBO, checkpoint=CHECKPOINTER)
    .configurable_fields(
        llm=ConfigurableField(id="llm_type", name="LLM Type"),
        system_message=ConfigurableField(id="system_message", name="Instructions"),
    )
    .with_types(
        input_type=Messages,
        output_type=Sequence[AnyMessage],
    )
)


class ConfigurableRetrieval(RunnableBinding):
    llm_type: LLMType
    system_message: str = DEFAULT_SYSTEM_MESSAGE
    assistant_id: Optional[str] = None
    thread_id: Optional[str] = None
    user_id: Optional[str] = None

    def __init__(
        self,
        *,
        llm_type: LLMType = LLMType.GPT_35_TURBO,
        system_message: str = DEFAULT_SYSTEM_MESSAGE,
        assistant_id: str = "",
        thread_id: str = "",
        kwargs: Optional[Mapping[str, Any]] = None,
        config: Optional[RunnableConfig] = None,
        **others: Any,
    ) -> None:
        others.pop("bound", None)
        retriever = get_retriever(assistant_id, thread_id)
        if llm_type == LLMType.GPT_35_TURBO:
            llm = get_openai_llm()
        elif llm_type == LLMType.CLAUDE:
            llm = get_anthropic_llm()
        elif llm_type == LLMType.GEMINI:
            llm = get_google_llm()
        elif llm_type == LLMType.MISTRAL:
            llm = get_mixtral_fireworks()
        elif llm_type == LLMType.FFM:
            llm = get_ffm_llm(model=FFM_MODEL_NAME)
        elif llm_type == LLMType.OLLAMA:
            llm = get_ollama_llm()
        else:
            raise ValueError("Unexpected llm type")
        chatbot = get_retrieval_executor(llm, retriever, system_message, CHECKPOINTER)
        super().__init__(
            llm_type=llm_type,  # type: ignore
            system_message=system_message,  # type: ignore
            bound=chatbot,
            kwargs=kwargs or {},
            config=config or {},
        )


chat_retrieval = (
    ConfigurableRetrieval(llm_type=LLMType.GPT_35_TURBO, checkpoint=CHECKPOINTER)
    .configurable_fields(
        llm_type=ConfigurableField(id="llm_type", name="LLM Type"),
        system_message=ConfigurableField(id="system_message", name="Instructions"),
        assistant_id=ConfigurableField(
            id="assistant_id", name="Assistant ID", is_shared=True
        ),
        thread_id=ConfigurableField(id="thread_id", name="Thread ID", is_shared=True),
    )
    .with_types(
        input_type=Dict[str, Any],
        output_type=Dict[str, Any],
    )
)


agent: Pregel = (
    ConfigurableAgent(
        agent=AgentType.GPT_35_TURBO,
        tools=[],
        system_message=DEFAULT_SYSTEM_MESSAGE,
        retrieval_description=RETRIEVAL_DESCRIPTION,
        assistant_id=None,
        thread_id=None,
    )
    .configurable_fields(
        agent=ConfigurableField(id="agent_type", name="Agent Type"),
        system_message=ConfigurableField(id="system_message", name="Instructions"),
        interrupt_before_action=ConfigurableField(
            id="interrupt_before_action",
            name="Tool Confirmation",
            description="If Yes, you'll be prompted to continue before each tool is executed.\nIf No, tools will be executed automatically by the agent.",
        ),
        assistant_id=ConfigurableField(
            id="assistant_id", name="Assistant ID", is_shared=True
        ),
        thread_id=ConfigurableField(id="thread_id", name="Thread ID", is_shared=True),
        tools=ConfigurableField(id="tools", name="Tools"),
        retrieval_description=ConfigurableField(
            id="retrieval_description", name="Retrieval Description"
        ),
    )
    .configurable_alternatives(
        ConfigurableField(id="type", name="Bot Type"),
        default_key="agent",
        prefix_keys=True,
        chatbot=chatbot,
        chat_retrieval=chat_retrieval,
    )
    .with_types(
        input_type=Messages,
        output_type=Sequence[AnyMessage],
    )
)

if __name__ == "__main__":
    import asyncio

    from langchain.schema.messages import HumanMessage

    async def run():
        async for m in agent.astream_events(
            HumanMessage(content="whats your name"),
            config={"configurable": {"user_id": "2", "thread_id": "test1"}},
            version="v1",
        ):
            print(m)

    asyncio.run(run())
