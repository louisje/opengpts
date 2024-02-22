from json import tool
import os
from functools import lru_cache
from typing import Dict
import httpx
import boto3
from langchain_community.chat_models import BedrockChat, ChatAnthropic, ChatFireworks, ChatOllama
from langchain_community.chat_models.ffm import ChatFFM
from langchain_google_vertexai import ChatVertexAI
from langchain_google_vertexai._enums import HarmBlockThreshold, HarmCategory
from langchain_openai import AzureChatOpenAI, ChatOpenAI

def get_ollama_llm(model):
    llm = ChatOllama(
        base_url="http://ollama:11434",
        model=model,
        stop=["[INST]","[/INST]"],
        streaming=True,
        num_gpu=0,
    )
    return llm

def get_ffm_llm():
    llm = ChatFFM(
        model="ffm-llama2-70b-exp", # candidates: "ffm-llama2-70b-chat", "codellama-70b-instruct"
        max_new_tokens=1024,
        temperature=0.5,
        top_k=50,
        top_p=1.0,
        frequence_penalty=1.0,
        ffm_api_key=os.environ["FFM_API_KEY"],
        base_url=os.environ["FFM_BASE_URL"],
        streaming=True,
        stop=None,
    )
    return llm

@lru_cache(maxsize=4)
def get_openai_llm(gpt_4: bool = False, azure: bool = False):
    proxy_url = os.getenv("PROXY_URL")
    if proxy_url is not None or proxy_url != "":
        http_client = httpx.AsyncClient(proxies=proxy_url)
    else:
        http_client = None
    if not azure:
        if gpt_4:
            llm = ChatOpenAI(
                http_client=http_client,
                model="gpt-4-1106-preview",
                temperature=0,
                streaming=True,
            )
        else:
            llm = ChatOpenAI(
                http_client=http_client,
                model="gpt-3.5-turbo-1106",
                temperature=0,
                streaming=True,
            )
    else:
        llm = AzureChatOpenAI(
            http_client=http_client,
            temperature=0,
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            base_url=os.environ["AZURE_OPENAI_API_BASE"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"], # type: ignore
            streaming=True,
        )
    return llm


@lru_cache(maxsize=2)
def get_anthropic_llm(bedrock: bool = False):
    if bedrock:
        client = boto3.client(
            "bedrock-runtime",
            region_name="us-west-2",
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        )
        model = BedrockChat(model_id="anthropic.claude-v2", client=client)
    else:
        model = ChatAnthropic(temperature=0, max_tokens=2000)
    return model


@lru_cache(maxsize=1)
def get_google_llm():
    return ChatVertexAI(
        project=os.environ["GOOGLE_CLOUD_PROJECT_ID"],
        model_name="gemini-pro",
        convert_system_message_to_human=True,
        streaming=True,
#       safety_settings={
#           HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
#           HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
#           HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
#           HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
#       },
    )


@lru_cache(maxsize=1)
def get_mixtral_fireworks():
    return ChatFireworks(model="accounts/fireworks/models/mixtral-8x7b-instruct")
