import logging
import os
import httpx
import boto3

from functools import lru_cache
from urllib.parse import urlparse

from langchain_community.chat_models import BedrockChat, ChatAnthropic, ChatFireworks, ChatOllama
from langchain_community.chat_models.ffm import ChatFFM
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import AzureChatOpenAI, ChatOpenAI

def get_ollama_llm(model):
    llm = ChatOllama(
        base_url="http://ollama:11434",
        model=model,
        stop=["[INST]","[/INST]","<start_of_turn>","<end_of_turn>"],
        streaming=True,
        num_gpu=0,
        temperature=0.5,
        top_k=50,
        top_p=1.0,
        repeat_penalty=1.0,
    )
    return llm

def get_ffm_llm(model: str):
    llm = ChatFFM(
        model=model,
        max_new_tokens=1024,
        temperature=0.5,
        top_k=50,
        top_p=1.0,
        frequence_penalty=1.0,
        ffm_api_key=os.environ["FFM_API_KEY"],
        base_url=os.environ["FFM_BASE_URL"],
        streaming=True,
        stop=["<|func_end|>"],
    )
    return llm

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def get_openai_llm(gpt_4: bool = False, azure: bool = False):
    proxy_url = os.getenv("PROXY_URL")
    http_client = None
    if proxy_url:
        parsed_url = urlparse(proxy_url)
        if parsed_url.scheme and parsed_url.netloc:
            http_client = httpx.AsyncClient(proxies=proxy_url)
        else:
            logger.warn("Invalid proxy URL provided. Proceeding without proxy.")

    if not azure:
        if gpt_4:
            llm = ChatOpenAI(
                http_client=http_client,
                model=os.environ["GPT_35_TURBO_MODEL"],
                temperature=0.1,
                streaming=True,
            )
        else:
            llm = ChatOpenAI(
                http_client=http_client,
                model=os.environ["GPT_35_TURBO_MODEL"],
                temperature=0.1,
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
        model_name=os.environ["GEMINI_MODEL"],
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
