import os

from langchain.chat_models import ChatOllama
from langchain.tools.render import render_text_description
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from .prompts import conversational_prompt, parse_output

def get_ollama_function_agent(tools, system_message):
    llm = ChatOllama(
        base_url="http://ollama:11434",
        model="llama2",
        stop=["[INST]"],
        streaming=False,
        num_gpu=0,
    )

    if tools:
        prompt = conversational_prompt.partial(
            tools=render_text_description(tools),
            tool_names=", ".join([t.name for t in tools]),
            system_message=system_message,
        )
    else:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
    agent = prompt | llm | parse_output

    return agent
