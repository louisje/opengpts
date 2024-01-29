from enum import Enum

from .openai import get_openai_function_agent
from .ffm import get_ffm_function_agent
from .ollama import get_ollama_function_agent
from .xml.agent import get_xml_agent


class GizmoAgentType(str, Enum):
    GPT_4 = "OpenAI (gpt-4)"
    GPT_35_TURBO = "OpenAI (gpt-3.5-turbo)"
    FFM = "FFM (ffm-llama2-70b-chat)"
    OLLAMA = "Ollama (llama2-7b-chat)"
    MISTRAL = "Ollama (mistral)"


__all__ = [
    "get_ffm_function_agent",
    "get_openai_function_agent",
    "get_ollama_function_agent",
    "get_xml_agent",
    "GizmoAgentType",
]
