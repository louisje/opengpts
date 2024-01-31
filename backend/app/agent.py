from enum import Enum
from typing import Any, Mapping, Optional, Sequence

from langchain_core.messages import AnyMessage
from langchain_core.runnables import (
    ConfigurableField,
    ConfigurableFieldMultiOption,
    RunnableBinding,
)

from app.agent_types.google_agent import get_google_agent_executor
from app.agent_types.openai_agent import get_openai_agent_executor
from app.agent_types.xml_agent import get_xml_agent_executor
from app.agent_types.ffm_agent import get_ffm_agent_executor
from app.chatbot import get_chatbot_executor
from app.agent_types.ollama_agent import get_ollama_agent_executor
from app.checkpoint import RedisCheckpoint
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
    TOOL_OPTIONS,
    TOOLS,
    AvailableTools,
    get_retrieval_tool,
    get_retriever,
)


class AgentType(str, Enum):
    GPT_35_TURBO = "GPT 3.5 Turbo"
    GPT_4 = "GPT 4"
    CLAUDE2 = "Claude 2"
    GEMINI = "Gemini (Google)"
    FFM = "ffm-llama2-70b-exp (FFM)"
    OLLAMA = "llama2-7b-chat (Ollama)"


DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant."

CHECKPOINTER = RedisCheckpoint()


def get_agent_executor(
    tools: list,
    agent: AgentType,
    system_message: str,
):
    if agent == AgentType.GPT_35_TURBO:
        llm = get_openai_llm()
        return get_openai_agent_executor(tools, llm, system_message, CHECKPOINTER)
    elif agent == AgentType.GPT_4:
        llm = get_openai_llm(gpt_4=True)
        return get_openai_agent_executor(tools, llm, system_message, CHECKPOINTER)
    elif agent == AgentType.CLAUDE2:
        llm = get_anthropic_llm()
        return get_xml_agent_executor(tools, llm, system_message, CHECKPOINTER)
    elif agent == AgentType.GEMINI:
        llm = get_google_llm()
        return get_google_agent_executor(tools, llm, system_message, CHECKPOINTER)
    elif agent == AgentType.FFM:
        llm = get_ffm_llm()
        return get_ffm_agent_executor(tools, llm, system_message, CHECKPOINTER)
    elif agent == AgentType.OLLAMA:
        llm = get_ollama_llm(model="llama2")
        return get_ollama_agent_executor(tools, llm, system_message, CHECKPOINTER)
    else:
        raise ValueError("Unexpected agent type")


class ConfigurableAgent(RunnableBinding):
    tools: Sequence[str]
    agent: AgentType
    system_message: str = DEFAULT_SYSTEM_MESSAGE
    retrieval_description: str = RETRIEVAL_DESCRIPTION
    assistant_id: Optional[str] = None
    user_id: Optional[str] = None

    def __init__(
        self,
        *,
        tools: Sequence[str],
        agent: AgentType = AgentType.GPT_35_TURBO,
        system_message: str = DEFAULT_SYSTEM_MESSAGE,
        assistant_id: Optional[str] = None,
        retrieval_description: str = RETRIEVAL_DESCRIPTION,
        kwargs: Optional[Mapping[str, Any]] = None,
        config: Optional[Mapping[str, Any]] = None,
        **others: Any,
    ) -> None:
        others.pop("bound", None)
        _tools = []
        for _tool in tools:
            if _tool == AvailableTools.RETRIEVAL:
                if assistant_id is None:
                    raise ValueError(
                        "assistant_id must be provided if Retrieval tool is used"
                    )
                _tools.append(get_retrieval_tool(assistant_id, retrieval_description))
            else:
                _returned_tools = TOOLS[_tool]()
                if isinstance(_returned_tools, list):
                    _tools.extend(_returned_tools)
                else:
                    _tools.append(_returned_tools)
        _agent = get_agent_executor(_tools, agent, system_message)
        agent_executor = _agent.with_config({"recursion_limit": 50})
        super().__init__(
            tools=tools,
            agent=agent,
            system_message=system_message,
            retrieval_description=retrieval_description,
            bound=agent_executor,
            kwargs=kwargs or {},
            config=config or {},
        )


class LLMType(str, Enum):
    GPT_35_TURBO = "GPT 3.5 Turbo"
    GPT_4 = "GPT 4"
    CLAUDE2 = "Claude 2"
    GEMINI = "Gemini (Google)"
    FFM = "ffm-llama2-70b-exp (FFM)"
    MIXTRAL = "Mixtral"
    OLLAMA = "llama2-7b-chat (Ollama)"


def get_chatbot(
    llm_type: LLMType,
    system_message: str,
):
    if llm_type == LLMType.GPT_35_TURBO:
        llm = get_openai_llm()
    elif llm_type == LLMType.GPT_4:
        llm = get_openai_llm(gpt_4=True)
    elif llm_type == LLMType.CLAUDE2:
        llm = get_anthropic_llm()
    elif llm_type == LLMType.GEMINI:
        llm = get_google_llm()
    elif llm_type == LLMType.MIXTRAL:
        llm = get_mixtral_fireworks()
    elif llm_type == LLMType.FFM:
        llm = get_ffm_llm()
    elif llm_type == LLMType.OLLAMA:
        llm = get_ollama_llm(model="llama2")
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
        config: Optional[Mapping[str, Any]] = None,
        **others: Any,
    ) -> None:
        others.pop("bound", None)

        chatbot = get_chatbot(llm, system_message)
        super().__init__(
            llm=llm,
            system_message=system_message,
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
    .with_types(input_type=Sequence[AnyMessage], output_type=Sequence[AnyMessage])
)


class ConfigurableRetrieval(RunnableBinding):
    llm_type: LLMType
    system_message: str = DEFAULT_SYSTEM_MESSAGE
    assistant_id: Optional[str] = None
    user_id: Optional[str] = None

    def __init__(
        self,
        *,
        llm_type: LLMType = LLMType.GPT_35_TURBO,
        system_message: str = DEFAULT_SYSTEM_MESSAGE,
        assistant_id: Optional[str] = None,
        kwargs: Optional[Mapping[str, Any]] = None,
        config: Optional[Mapping[str, Any]] = None,
        **others: Any,
    ) -> None:
        others.pop("bound", None)
        retriever = get_retriever(assistant_id)
        if llm_type == LLMType.GPT_35_TURBO:
            llm = get_openai_llm()
        elif llm_type == LLMType.GPT_4:
            llm = get_openai_llm(gpt_4=True)
        elif llm_type == LLMType.CLAUDE2:
            llm = get_anthropic_llm()
        elif llm_type == LLMType.GEMINI:
            llm = get_google_llm()
        elif llm_type == LLMType.MIXTRAL:
            llm = get_mixtral_fireworks()
        elif llm_type == LLMType.FFM:
            llm = get_ffm_llm()
        elif llm_type == LLMType.OLLAMA:
            llm = get_ollama_llm(model="llama2")
        else:
            raise ValueError("Unexpected llm type")
        chatbot = get_retrieval_executor(llm, retriever, system_message, CHECKPOINTER)
        super().__init__(
            llm_type=llm_type,
            system_message=system_message,
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
    )
    .with_types(input_type=Sequence[AnyMessage], output_type=Sequence[AnyMessage])
)


agent = (
    ConfigurableAgent(
        agent=AgentType.GPT_35_TURBO,
        tools=[],
        system_message=DEFAULT_SYSTEM_MESSAGE,
        retrieval_description=RETRIEVAL_DESCRIPTION,
        assistant_id=None,
    )
    .configurable_fields(
        agent=ConfigurableField(id="agent_type", name="Agent Type"),
        system_message=ConfigurableField(id="system_message", name="Instructions"),
        assistant_id=ConfigurableField(
            id="assistant_id", name="Assistant ID", is_shared=True
        ),
        tools=ConfigurableFieldMultiOption(
            id="tools",
            name="Tools",
            options=TOOL_OPTIONS,
            default=[],
        ),
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
    .with_types(input_type=Sequence[AnyMessage], output_type=Sequence[AnyMessage])
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