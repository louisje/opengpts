import os

from langchain.tools.render import render_text_description
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_community.chat_models import ChatOllama
from langchain_core.agents import AgentAction

from langchain_core.language_models.base import LanguageModelLike
from langchain_core.messages.function import FunctionMessage
from langchain_core.messages.system import SystemMessage
from langchain_core.prompts.chat import SystemMessagePromptTemplate
from langchain_core.tools import BaseTool

from .prompt_template import template
from .output_parser import parse_output

from langgraph.checkpoint import BaseCheckpointSaver
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langgraph.graph.message import MessageGraph
from langgraph.graph import END

from app.message_types import LiberalFunctionMessage

def get_ollama_agent_executor(
    tools: list[BaseTool],
    llm: LanguageModelLike,
    system_message: str,
    checkpoint: BaseCheckpointSaver
):
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
        if tools:
            parcial_variables = {
                "tools": render_text_description(tools),
                "tool_names": ", ".join([t.name for t in tools]),
                "system_message": system_message,
            }
            prompt = SystemMessagePromptTemplate.from_template(template=template,partial_variables=parcial_variables)
            return [prompt.format()] + msgs

        return [SystemMessage(content=system_message)] + msgs

    agent = _get_messages | llm
    tool_executor = ToolExecutor(tools)

    def should_continue(messages):
        last_message = messages[-1]
        if isinstance(parse_output(last_message), AgentAction):
            return "continue"
        else:
            return "end"

    # Define the function to execute tools
    async def call_tool(messages):
        # Based on the continue condition
        # we know the last message involves a function call
        last_message = messages[-1]
        # We construct an ToolInvocation from the function_call
        _action = parse_output(last_message)
        if not action or not isinstance(action, AgentAction):
            raise ValueError("Invalid action type")
        # We call the tool_executor and get back a response
        action = ToolInvocation(
            tool=_action.tool,
            tool_input=_action.tool_input,
        )
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
    return app
