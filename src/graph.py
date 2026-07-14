from typing import Literal
import logging

from state import State
from langgraph.constants import START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph, StateGraph
from langchain_core.messages import AIMessage, ToolMessage

from llm import load_llm
from tools import TOOLS, TOOLS_BY_NAME

logger = logging.getLogger(__name__)
MAX_TOOL_CALLS_PER_TOOL = 3

def call_llm(state: State) -> State:
    logger.info("Node call_llm: %d mensagens no estado", len(state["messages"]))
    llm_with_tools = load_llm().bind_tools(TOOLS)
    result = llm_with_tools.invoke(state["messages"])
    tool_calls = getattr(result, "tool_calls", None) or []
    logger.info(
        "LLM respondeu: content_len=%s tool_calls=%s",
        len(str(result.content)),
        [tc.get("name") for tc in tool_calls],
    )
    return {"messages": [result], "tool_calls_count": {}}


def tool_node(state: State) -> State:
    llm_response = state["messages"][-1]
    if not isinstance(llm_response, AIMessage) or not llm_response.tool_calls:
        logger.warning("tool_node chamado sem tool_calls")
        return state
    
    tool_messages = []

    for tool_call in llm_response.tool_calls:
        if state["tool_calls_count"].get(tool_call["name"], 0) >= MAX_TOOL_CALLS_PER_TOOL:
            tool_messages.append(ToolMessage(
                content=f"Tool {tool_call['name']} has reached the maximum number of calls",
                tool_call_id=tool_call["id"],
                tool_name=tool_call["name"],
                status="error",
            ))
            continue
        tool_name, args, id_ = tool_call["name"], tool_call["args"], tool_call["id"]
        logger.info("Node tool_node: executando %s(%s)", tool_name, args)
        tool = TOOLS_BY_NAME[tool_name]
        try:
            content = tool.invoke(args)
            status = "success"
        except Exception as e:
            content = f"Erro ao executar a ferramenta {tool_name}: {e}"
            status = "error"
            logger.exception("Falha na ferramenta %s", tool_name)

        tool_message = ToolMessage(
            content=content,
            tool_call_id=id_,
            tool_name=tool_name,
            status=status,
        )
        logger.info("Ferramenta %s finalizou com status=%s", tool_name, status)
        tool_messages.append(tool_message)

    counts = {}
    for tm in tool_messages:
        counts[tm.tool_name] = counts.get(tm.tool_name, 0) + 1
    return {"messages": tool_messages, "tool_calls_count": counts}

def router(state: State) -> Literal["tool_node", "__end__"]:
    llm_response = state["messages"][-1]
    if isinstance(llm_response, AIMessage) and getattr(llm_response, "tool_calls", None):
        return "tool_node"
    return "__end__"


def build_graph() -> CompiledStateGraph[State, None, State, State]:
    graph = StateGraph(State)
    graph.add_node("call_llm", call_llm)
    graph.add_node("tool_node", tool_node)
    
    graph.add_edge(START, "call_llm")
    graph.add_conditional_edges("call_llm", router, ["tool_node", END])
    graph.add_edge("tool_node", "call_llm")

    return graph.compile(checkpointer=InMemorySaver())