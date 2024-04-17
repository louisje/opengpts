import logging
import os
from functools import lru_cache
from urllib.parse import urlparse

import boto3
import httpx
from langchain_community.chat_models import BedrockChat, ChatFireworks
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.chat_models.ffm import ChatFFM
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory
from langchain_anthropic import ChatAnthropic


def get_ffm_llm(model: str):
    llm = ChatFFM(
        model=model,
        max_new_tokens=1024,
        temperature=0.1,
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
def get_openai_llm(gpt_4: bool = False):
    proxy_url = os.getenv("PROXY_URL")
    http_client = None
    if proxy_url:
        parsed_url = urlparse(proxy_url)
        if parsed_url.scheme and parsed_url.netloc:
            http_client = httpx.AsyncClient(proxies=proxy_url)
        else:
            logger.warn("Invalid proxy URL provided. Proceeding without proxy.")

    openai_model = os.environ["GPT_4_MODEL"] if gpt_4 else os.environ["GPT_35_TURBO_MODEL"]
    llm = ChatOpenAI(
        http_client=http_client,
        model=openai_model,
        temperature=0,
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
        model = ChatAnthropic(
            model_name="claude-3-haiku-20240307",
            max_tokens_to_sample=2000,
            temperature=0,
        )
    return model


@lru_cache(maxsize=1)
def get_google_llm():
    return ChatVertexAI(
        project=os.environ["GOOGLE_CLOUD_PROJECT_ID"],
        model_name=os.environ["GEMINI_MODEL"],
        convert_system_message_to_human=True,
        streaming=False,
        safety_settings={
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        },
    )


@lru_cache(maxsize=1)
def get_mixtral_fireworks():
    return ChatFireworks(model="accounts/fireworks/models/mixtral-8x7b-instruct")


@lru_cache(maxsize=1)
def get_ollama_llm():
    model_name = os.environ.get("OLLAMA_MODEL")
    if not model_name:
        model_name = "llama2"
    ollama_base_url = os.environ.get("OLLAMA_BASE_URL")
    if not ollama_base_url:
        ollama_base_url = "http://localhost:11434"

    return ChatOllama(
        model=model_name,
        base_url=ollama_base_url,
        stop=["[INST]","[/INST]","<start_of_turn>","<end_of_turn>"],
        num_gpu=0,
        temperature=0.5,
        top_k=50,
        top_p=1.0,
        repeat_penalty=1.0,
    )
