import os

import boto3
from langchain_community.chat_models import BedrockChat, ChatAnthropic, ChatFireworks, ChatFFM, ChatOllama
from langchain_google_vertexai import ChatVertexAI
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
        model="ffm-llama2-70b-exp",
        max_new_tokens=1024,
        temperature=0.5,
        top_k=50,
        top_p=1.0,
        frequence_penalty=1.0,
        ffm_api_key=os.environ["FFM_API_KEY"],
        base_url=os.environ["FFM_BASE_URL"],
        streaming=True,
    )
    return llm

def get_openai_llm(gpt_4: bool = False, azure: bool = False):
    if not azure:
        if gpt_4:
            llm = ChatOpenAI(model="gpt-4-1106-preview", temperature=0, streaming=True)
        else:
            llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0, streaming=True)
    else:
        llm = AzureChatOpenAI(
            temperature=0,
            deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            openai_api_base=os.environ["AZURE_OPENAI_API_BASE"],
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
            streaming=True,
        )
    return llm


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
        model = ChatAnthropic(temperature=0, max_tokens_to_sample=2000)
    return model


def get_google_llm():
    return ChatVertexAI(
        model_name="gemini-pro", convert_system_message_to_human=True, streaming=True
    )


def get_mixtral_fireworks():
    return ChatFireworks(model="accounts/fireworks/models/mixtral-8x7b-instruct")
