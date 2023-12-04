import json
import re

from typing import Union

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import AgentAction, AgentFinish
from langchain.output_parsers.json import parse_json_markdown

from langchain_core.exceptions import OutputParserException

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

def parse_output(msg):
    try:
        matches = re.findall(r"```(json)?(.*)```", msg.content, re.DOTALL)
        print(matches) ##
        if not matches:
            return AgentFinish(return_values={"output": msg.content}, log=msg.content)
        json_str = matches[0][1]
        json_str = json_str.strip()
        json_str = _custom_parser(json_str)
        parsed = json.loads(json_str)

        if parsed["action"] is not None:
            return AgentAction(parsed["action"], parsed["action_input"], msg.content)
        else:
            return AgentFinish(return_values={"output": msg.content}, log=msg.content)
    except Exception as e:
        print(f"Could not parse LLM output. ({e})")
        return AgentFinish(return_values={"output": msg.content}, log=msg.content)

template = """{system_message}

工具的使用
----------

你可以要求使用工具取得更多有用的資訊來回答提問。
下面是你可以用的工具：

{tools}

回覆內容格式
------------

如果你想要使用工具，請以下面 Markdown 格式回覆（請務必包含"```"的部分）：

```json
{{
  "action": string, \\ 你要使用的工具名稱。必須是這幾個其中之一 {tool_names}
  "action_input": string \\ 要輸入給工具的內容
}}
```"""

conversational_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

