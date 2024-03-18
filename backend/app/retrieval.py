import json
from typing import List

from app import stream
from app.tools import _get_google_search

from langchain_community.document_loaders.chromium import AsyncChromiumLoader
from langchain_community.document_transformers.html2text import Html2TextTransformer
from langchain_core.documents.base import Document
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

ask_prompt_template = """This is the question:
{question}

Can the following content answer to the question? Please reply me in only this json format, no other description.
{{
    "answer": "YOUR ANSWER HERE"
}}
The answer can ONLY be `Yes`, `No`, or `Not sure`.

Here is the content:
{context}"""

summary_prompt_template = """This is the question:
{question}

Summarize the following content against the question in around 500 words.

Here is the content:
{context}"""

search_prompt_template = """This is the question:
{question}

If I want to search on internet, how do I compose search keyword? Please provide in this json format, no any other explaination.
{{
    "answer": "YOUR ANSWER HERE"
}}"""


def get_retrieval_executor(
    llm: BaseChatModel,
    retriever: BaseRetriever,
    system_message: str,
    checkpoint: BaseCheckpointSaver,
):
    google_search_results = _get_google_search()
    tool_executor = ToolExecutor([google_search_results])

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
        response = llm.invoke(input=ask_prompt_template.format(question=question, context=content), stream=False)
        answer = str(response.content)
        print("ANSWER: ", answer) # DEBUG
        try:
            json_load = json.loads(answer)
            answer = json_load["answer"]
        except:
            pass
        return answer

    async def retrieve(messages):
        params = messages[-1].additional_kwargs["function_call"]
        query = json.loads(params["arguments"])["query"]
        response = await retriever.ainvoke(query)
        msg = LiberalFunctionMessage(name="retrieval", content=response)
        return msg
    
    def invoke_search_full(messages):

        retrieve = messages.pop()
        question, search_results = ask_to_search_google(messages)
        results = Html2TextTransformer().transform_documents(AsyncChromiumLoader([result["link"] for result in search_results]).load())
        summary_results = llm.generate(messages=[[HumanMessage(content=summary_prompt_template.format(question=question, context=result.page_content))] for result in results], stream=False)
        for i in range(len(results)):
            results[i].page_content = summary_results.generations[i][0].text
        retrieve.content = results

        return retrieve

    def ask_to_search_google(messages):

        chat_history: List[BaseMessage] = []
        question = ""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                question = msg.content
            if isinstance(msg, AIMessage):
                if "function_call" not in msg.additional_kwargs or not msg.additional_kwargs["function_call"]:
                    chat_history.append(msg)
            if isinstance(msg, HumanMessage):
                chat_history.append(msg)

        chat_history.pop()
        response = llm.generate(messages=[chat_history+[HumanMessage(content=search_prompt_template.format(question=question))]], stream=False)
        answer = response.generations[0][0].text
        print("ANSWER: ", answer) # DEBUG
        try:
            json_load = json.loads(answer)
            answer = json_load["answer"]
        except:
            pass

        action = ToolInvocation(tool=google_search_results.name, tool_input={"query":answer})

        return question, json.loads(tool_executor.invoke(action))


    def invoke_search_half(messages):

        retrieve = messages.pop()
        retrieve.content.pop()
        retrieve.content.pop()
        question, search_results = ask_to_search_google(messages)
        results = Html2TextTransformer().transform_documents(AsyncChromiumLoader([result["link"] for result in search_results[:2]]).load())
        summary_results = llm.generate(messages=[[HumanMessage(content=summary_prompt_template.format(question=question, context=result.page_content))] for result in results], stream=False)
        for i in range(len(results)):
            results[i].page_content = summary_results.generations[i][0].text
        retrieve.content.extend(results)

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
