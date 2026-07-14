from collections.abc import Sequence
from typing import Annotated, TypedDict

from langgraph.graph.message import BaseMessage, add_messages

def merge_counts(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    merged = dict(left or {})
    for k, v in (right or {}).items():
        merged[k] = merged.get(k, 0) + v
    return merged

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    tool_calls_count: Annotated[dict[str, int], merge_counts]
