import json

from langchain_community.chat_models.ollama import ChatOllama

from langchain_core.language_models.base import LanguageModelLike
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.chat import ChatMessage
from langchain_core.messages.function import FunctionMessage
from langchain_core.messages.system import SystemMessage
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_function

from langgraph.checkpoint import BaseCheckpointSaver
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langgraph.graph.message import MessageGraph
from langgraph.graph import END

from app.message_types import LiberalFunctionMessage

from langchain_experimental.llms.ollama_functions import DEFAULT_RESPONSE_FUNCTION

def get_ollama_agent_executor(
    tools: list[BaseTool],
    llm: ChatOllama,
    system_message: str,
    interrupt_before_action: bool,
    checkpoint: BaseCheckpointSaver
):
    functions = [convert_to_openai_function(t) for t in tools]

    def _get_messages(messages):
        msgs = []
        prompt = None
        for m in messages:
            if isinstance(m, LiberalFunctionMessage):
                _dict = m.dict()
                _dict["content"] = str(_dict["content"])
                m_c = FunctionMessage(**_dict)
                msgs.append(m_c)
            else:
                msgs.append(m)

        return [SystemMessage(content=system_message)] + msgs

    def _parse_function(msg: AIMessage) -> AIMessage:
        if not isinstance(msg.content, str):
            raise ValueError("OllamaFunctions does not support non-string output.")
        try:
            print([msg.content]) # Qoo
            parsed_chat_result = json.loads(msg.content)
        except json.JSONDecodeError:
            print("Unable to parse a function call from OllamaFunctions output.")
            return msg
        called_tool_name = parsed_chat_result["tool_name"]
        called_tool_arguments = parsed_chat_result["tool_input"]
        called_tool = next(
            (fn for fn in functions if fn["name"] == called_tool_name), None
        )
        if called_tool is None:
            raise ValueError(
                f"Failed to parse a function call from {llm.name} output: {msg.content}"
            )
        if called_tool["name"] == DEFAULT_RESPONSE_FUNCTION["name"]:
            return AIMessage(
                content=called_tool_arguments["response"]
            )

        return AIMessage(
            content="",
            additional_kwargs={
                "function_call": {
                    "name": called_tool_name,
                    "arguments": json.dumps(called_tool_arguments, indent=4)
                    if called_tool_arguments
                    else "",
                },
            },
        )

    llm_with_tools = llm.bind(functions=functions)
    agent = _get_messages | llm_with_tools | _parse_function
    tool_executor = ToolExecutor(tools)

    def should_continue(messages):
        last_message: ChatMessage = messages[-1]
        # If there is no function call, then we finish
        if "function_call" not in last_message.additional_kwargs:
            return "end"
        # Otherwise if there is, we continue
        else:
            return "continue"

    # Define the function to execute tools
    async def call_tool(messages):
        actions: list[ToolInvocation] = []
        # Based on the continue condition
        # we know the last message involves a function call
        last_message = messages[-1]
        # We construct an ToolInvocation from the function_call
        action = ToolInvocation(
            tool=last_message.additional_kwargs["function_call"]["name"],
            tool_input=json.loads(
                last_message.additional_kwargs["function_call"]["arguments"]
            ),
        )
        # We call the tool_executor and get back a response
        response = await tool_executor.ainvoke(action)
        # We use the response to create a FunctionMessage
        function_message = FunctionMessage(content=str(response), name=action.tool)
        # We return a list, because this will get added to the existing list
        return function_message

    workflow = MessageGraph()

    # Define the two nodes we will cycle between
    workflow.add_node("agent", agent)
    workflow.add_node("action", call_tool)

    # Set the entrypoint as `agent`
    # This means that this node is the first one called
    workflow.set_entry_point("agent")

    # We now add a conditional edge
    workflow.add_conditional_edges(
        # First, we define the start node. We use `agent`.
        # This means these are the edges taken after the `agent` node is called.
        "agent",
        # Next, we pass in the function that will determine which node is called next.
        should_continue,
        # Finally we pass in a mapping.
        # The keys are strings, and the values are other nodes.
        # END is a special node marking that the graph should finish.
        # What will happen is we will call `should_continue`, and then the output of that
        # will be matched against the keys in this mapping.
        # Based on which one it matches, that node will then be called.
        {
            # If `tools`, then we call the tool node.
            "continue": "action",
            # Otherwise we finish.
            "end": END,
        },
    )

    # We now add a normal edge from `tools` to `agent`.
    # This means that after `tools` is called, `agent` node is called next.
    workflow.add_edge("action", "agent")

    # Finally, we compile it!
    # This compiles it into a LangChain Runnable,
    # meaning you can use it as you would any other runnable
    app = workflow.compile(checkpointer=checkpoint)
    if interrupt_before_action:
        app.interrupt = ["action:inbox"]
    return app
