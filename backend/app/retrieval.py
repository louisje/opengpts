from argparse import Action
from email import charset
from email.policy import strict
import json
from textwrap import indent
from typing import List
from unittest.mock import Base

from app.tools import TOOLS, AvailableTools
from langchain_community.document_loaders.chromium import AsyncChromiumLoader
from langchain_community.document_transformers.html2text import Html2TextTransformer

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.retrievers import BaseRetriever

from langgraph.checkpoint import BaseCheckpointSaver
from langgraph.graph import END
from langgraph.graph.message import MessageGraph
from langgraph.prebuilt import ToolExecutor, ToolInvocation

from app.message_types import LiberalFunctionMessage


response_prompt_template = """{instructions}

Respond to the user using ONLY the context provided below. Do not make anything up.

{context}"""

ask_prompt_template = """Here is the question:
{question}

Is the following content can answer to the question? Please answer in exactly "Yes" or "No" or "Not sure" without any other explaination.

{context}"""

search_prompt_template = """Here is the question:
{question}

If I want to search query on internet, how do I compose search query string? Please provide search query string itself ONLY, without any other explaination."""


def get_retrieval_executor(
    llm: BaseChatModel,
    retriever: BaseRetriever,
    system_message: str,
    checkpoint: BaseCheckpointSaver,
):
    tool_executor = ToolExecutor([TOOLS[AvailableTools.GOOGLE_SEARCH]()])

    def _get_messages(messages):
        chat_history = []
        for msg in messages:
            if isinstance(msg, AIMessage):
                if "function_call" not in msg.additional_kwargs:
                    chat_history.append(msg)
            if isinstance(msg, HumanMessage):
                chat_history.append(msg)
        docs = messages[-1].content
        content = "\n...\n".join([doc.page_content for doc in docs])
        return [
            SystemMessage(
                content=response_prompt_template.format(
                    instructions=system_message, context=content
                )
            )
        ] + chat_history

    async def invoke_retrieval(messages):
        human_input = messages[-1].content
        return AIMessage(
            content="",
            additional_kwargs={
                "function_call": {
                    "name": "retrieval",
                    "arguments": json.dumps({"query": human_input}),
                }
            },
        )

    def ask_retrieval(messages) -> str:
        question = ""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                question = msg.content
        docs = messages[-1].content
        content = "\n...\n".join([doc.page_content for doc in docs])
        response = llm.invoke(input=ask_prompt_template.format(question=question, context=content))
        answer = str(response.content)
        print("ANSWER: ", answer) # DEBUG
        return answer

    async def retrieve(messages):
        params = messages[-1].additional_kwargs["function_call"]
        query = json.loads(params["arguments"])["query"]
        response = await retriever.ainvoke(query)
        msg = LiberalFunctionMessage(name="retrieval", content=response)
        return msg
    
    async def invoke_search_full(messages):
        retrieve = messages.pop()

        search_results = ask_to_search_google(messages)

        loader = AsyncChromiumLoader([result.link for result in search_results])
        htmls = loader.load()
        html2text = Html2TextTransformer()
        docs = html2text.transform_documents(htmls)
        retrieve.content = docs

        return retrieve

    def ask_to_search_google(messages):
        chat_history: List[BaseMessage] = []
        question = ""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                question = msg.content
            if isinstance(msg, AIMessage):
                if "function_call" not in msg.additional_kwargs:
                    chat_history.append(msg)
            if isinstance(msg, HumanMessage):
                chat_history.append(msg)

        chat_history.pop()
        chat_history.append(HumanMessage(content=search_prompt_template.format(question=question)))
        response = llm.generate(messages=[chat_history])
        search_keyword = response.generations[0][0].text
        print("SEARCH KEYWORD: ", search_keyword)
        action = ToolInvocation(tool=TOOLS[AvailableTools.GOOGLE_SEARCH]().name, tool_input={"query":search_keyword})

        return tool_executor.invoke(action)


    async def invoke_search_half(messages):
        retrieve = messages.pop()

        search_results = ask_to_search_google(messages)

        loader = AsyncChromiumLoader([result.link for result in search_results[:2]])
        htmls = loader.load()
        html2text = Html2TextTransformer()
        docs = html2text.transform_documents(htmls)
        if len(docs) > 0:
            retrieve.content[2] = docs[0]
        if len(docs) > 1:
            retrieve.content[3] = docs[1]

        return retrieve


    response = _get_messages | llm

    workflow = MessageGraph()

    workflow.add_node("invoke_retrieval", invoke_retrieval)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("response", response)
    workflow.add_node("invoke_search_full", invoke_search_full)
    workflow.add_node("invoke_search_half", invoke_search_half)

    workflow.set_entry_point("invoke_retrieval")
 
    workflow.add_edge("invoke_retrieval", "retrieve")
    workflow.add_conditional_edges(
        "retrieve",
        ask_retrieval,
        {
            "Yes": "response",
            "No": "invoke_search_full",
            "Not sure": "invoke_search_half",
        },
    )
    workflow.add_edge("invoke_search_full", "response")
    workflow.add_edge("invoke_search_half", "response")
    workflow.add_edge("response", END)
 
    return workflow.compile(checkpointer=checkpoint)
