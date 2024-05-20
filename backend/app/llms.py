import os
from functools import lru_cache
from urllib.parse import urlparse

import httpx
import structlog
from pydantic import SecretStr
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.chat_models.ffm import ChatFFM
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory


def get_ffm_llm(model: str):
    llm = ChatFFM(
        model=model,
        max_new_tokens=1024,
        temperature=0.2,
        top_k=50,
        top_p=1.0,
        frequence_penalty=1.0,
        ffm_api_key=os.environ["FFM_API_KEY"],
        base_url=os.environ["FFM_BASE_URL"],
        streaming=True,
        stop=["<|func_end|>"],
    )
    return llm

logger = structlog.get_logger(__name__)


@lru_cache(maxsize=4)
def get_openai_llm(openai: bool = False, gpt4: bool = False, model = None):
    proxy_url = os.getenv("PROXY_URL")
    http_client = None
    if proxy_url:
        parsed_url = urlparse(proxy_url)
        if parsed_url.scheme and parsed_url.netloc:
            http_client = httpx.AsyncClient(proxies=proxy_url)
        else:
            logger.warn("Invalid proxy URL provided. Proceeding without proxy.")

    llm = ChatOpenAI(
        http_async_client=http_client,
        base_url="https://api.openai.com/v1" if openai else (os.getenv("OPENAI_BASE_URL") or None),
        model=model if model else os.environ["GPT_4_MODEL"] if gpt4 else os.environ["GPT_35_TURBO_MODEL"],
        temperature=0.5,
        streaming=True,
    )
    return llm


@lru_cache(maxsize=2)
def get_anthropic_llm():
        return ChatAnthropic(
            model_name=os.environ["CLAUDE_MODEL"],
            max_tokens_to_sample=2000,
            temperature=0,
            timeout=10000,
            api_key=SecretStr(os.environ["ANTHROPIC_API_KEY"])
        )


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
        stop=["[INST]","[/INST]","<start_of_turn>","<end_of_turn>","<|end_header_id|>","<|end_id|>","<|eot_header_id|>","<|eot_id|>"],
        num_gpu=0,
        temperature=0.5,
        top_k=50,
        top_p=1.0,
        repeat_penalty=1.0,
    )
