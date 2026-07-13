import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import BaseTool
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    ToolMessage,
    AIMessage,
    BaseMessage
)
from prompts import SYSTEM_PROMPT, INPUT_PROMPT
from tools import (
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_txt
)

load_dotenv(Path(__file__).resolve().parent / ".env")

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
log = logging.getLogger(__name__)

local_host = os.getenv("LOCALHOST")

llm = init_chat_model(
    model=os.getenv("MODEL"),
    model_provider=os.getenv("MODEL_PROVIDER"),
    base_url=local_host,
    api_key=os.getenv("API_KEY"),
)

system_message = SystemMessage(content=SYSTEM_PROMPT)

messages: list[BaseMessage] = [system_message]

tools: list[BaseTool] = [extract_text_from_pdf, extract_text_from_docx, extract_text_from_txt]
tools_by_name: dict[str, BaseTool] = {tool.name: tool for tool in tools}

llm_with_tools = llm.bind_tools(tools)

while True:
    human_message = HumanMessage(content=input(INPUT_PROMPT))
    messages.append(human_message)
    
    llm_response = llm_with_tools.invoke(messages)
    messages.append(llm_response)
    if human_message.content.lower() in ("quit", "exit"):
        break

    if not llm_response.tool_calls:
        log.info("Model responded without tool calls")
        print(llm_response.content)
        continue

    elif isinstance(llm_response, AIMessage) and getattr(llm_response, "tool_calls", None):
        log.info("Model requested %s tool call(s)", len(llm_response.tool_calls))
        for tool_call in llm_response.tool_calls:
            tool_name, args, id_ = tool_call["name"], tool_call["args"], tool_call["id"]
            log.info("Calling tool '%s' with args=%s", tool_name, args)
            try:
                content = tools_by_name[tool_name].invoke(args)
                status = "success"
                log.info("Tool '%s' finished successfully", tool_name)
            except Exception as e:
                content = f"Erro ao executar a ferramenta {tool_name}: {e}"
                status = "error"
                log.error("Tool '%s' failed: %s", tool_name, e)
            messages.append(ToolMessage(content=content, tool_call_id=id_, tool_name=tool_name, status=status))

            llm_response = llm_with_tools.invoke(messages)
            messages.append(llm_response)

    print(llm_response.content)