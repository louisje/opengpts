from enum import Enum

from .openai import get_openai_function_agent
from .ffm.agent import get_ffm_function_agent
from .ollama.agent import get_ollama_function_agent
from .xml.agent import get_xml_agent


class GizmoAgentType(str, Enum):
    GPT_4 = "GPT 4"
    GPT_35_TURBO = "GPT 3.5 Turbo"
    FFM = "FFM"
    OLLAMA = "Ollama"


__all__ = [
    "get_ffm_function_agent",
    "get_openai_function_agent",
    "get_ollama_function_agent",
    "get_xml_agent",
    "GizmoAgentType",
]
