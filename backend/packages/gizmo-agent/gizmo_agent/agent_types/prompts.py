from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

template = """
角色說明
--------

{system_message}

工具的使用說明
--------------

你可以要求使用工具取得更多有用的資訊來回答提問。
下面是你所可以用的工具：

{tools}

回覆內容格式說明
----------------

如果你覺得需要使用工具，請以下面 Markdown 格式回覆（請務必包含"```"的部分）：

```json
{{
  "action": string, \\ 你要使用的工具名稱。必須是這幾個其中之一 {tool_names}
  "action_input": string \\ 要輸入給工具的內容
}}
```

---

好了，接下來我們開始對話，請不吝告訴我你的想法。

"""

conversational_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

