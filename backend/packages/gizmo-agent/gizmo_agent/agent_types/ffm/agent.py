import os

from langchain.agents import initialize_agent, AgentType
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.chat_models import ChatFFM
from langchain.tools.render import render_text_description
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.tools.render import format_tool_to_openai_function

from .prompts import conversational_prompt, parse_output

def get_ffm_function_agent(tools, system_message):
    llm = ChatFFM(
        model="ffm-llama2-70b-chat",
        max_new_tokens=1024,
        temperature=0.5,
        top_k=50,
        top_p=1.0,
        frequence_penalty=1.0,
        ffm_api_key=os.environ["FFM_API_KEY"],
        base_url=os.environ["FFM_BASE_URL"],
        streaming=True,
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
