from enum import Enum

from .openai import get_openai_function_agent
from .ffm import get_ffm_function_agent
from .ollama import get_ollama_function_agent
from .xml.agent import get_xml_agent


class GizmoAgentType(str, Enum):
    GPT_4 = "GPT 4"
    GPT_35_TURBO = "GPT 3.5 Turbo"
    FFM = "FFM"
    OLLAMA = "Ollama (llama2-7b-chat)"
    TAIWAN = "Ollama (taiwan-llm-7b-v2.1-chat)"


__all__ = [
    "get_ffm_function_agent",
    "get_openai_function_agent",
    "get_ollama_function_agent",
    "get_xml_agent",
    "GizmoAgentType",
]
