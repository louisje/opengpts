import json
import re

from langchain.schema import AgentAction, AgentFinish
from langchain_core.messages.chat import ChatMessage

from langchain_core.outputs.chat_result import ChatResult

def _replace_new_line(match: re.Match[str]) -> str:
    value = match.group(2)
    value = re.sub(r"\n", r"\\n", value)
    value = re.sub(r"\r", r"\\r", value)
    value = re.sub(r"\t", r"\\t", value)
    value = re.sub(r'(?<!\\)"', r"\"", value)

    return match.group(1) + value + match.group(3)


def _custom_parser(multiline_string: str) -> str:
    """
    The LLM response for `action_input` may be a multiline
    string containing unescaped newlines, tabs or quotes. This function
    replaces those characters with their escaped counterparts.
    (newlines in JSON must be double-escaped: `\\n`)
    """
    if isinstance(multiline_string, (bytes, bytearray)):
        multiline_string = multiline_string.decode()

    multiline_string = re.sub(
        r'("action_input"\:\s*")(.*)(")',
        _replace_new_line,
        multiline_string,
        flags=re.DOTALL,
    )

    return multiline_string


def parse_output(msg) -> AgentAction | AgentFinish:
    try:
        matches = re.findall(r"```(json)?(.*)```", msg.content, re.DOTALL)
        if not matches:
            return AgentFinish(return_values={"output": msg.content}, log=msg.content)
        json_str = matches[0][1]
        json_str = json_str.strip()
        json_str = _custom_parser(json_str)
        parsed = json.loads(json_str)

        if parsed["action"] is not None:
            if parsed["action"] == "Final Answer":
                return AgentFinish(return_values={"output": parsed["action_input"]}, log=msg.content)
            else:
                return AgentAction(parsed["action"], parsed["action_input"], log=msg.content)
        else:
            return AgentFinish(return_values={"output": msg.content}, log=msg.content)
    except Exception as e:
        print(f"Could not parse LLM output. ({e})")
        return AgentFinish(return_values={"output": msg.content}, log=msg.content)

