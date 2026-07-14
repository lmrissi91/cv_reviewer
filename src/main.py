import uuid
import logging

from langgraph.graph.state import RunnableConfig
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage
)
from rich import print
from rich.markdown import Markdown
from rich.prompt import Prompt

from prompts import SYSTEM_PROMPT, INPUT_PROMPT
from graph import build_graph

logger = logging.getLogger(__name__)

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("Iniciando main")
    graph = build_graph()
    config = RunnableConfig(configurable={"thread_id": str(uuid.uuid4())})
    all_messages: list[BaseMessage] = []

    prompt = Prompt()
    Prompt.prompt_suffix = ""

    while True:
        logger.info("Iniciando loop principal")
        user_input = prompt.ask(INPUT_PROMPT)
        
        if user_input.lower() in ("quit", "exit", "sair"):
            break

        human_message = HumanMessage(content=user_input)
        current_loop_messages = [human_message]

        if not all_messages:
            logger.info("Adicionando sistema message")
            all_messages = [SystemMessage(content=SYSTEM_PROMPT)]

        result = graph.invoke({"messages": current_loop_messages}, config=config)

        print("[bold cyan]RESPOSTA: \n")
        for message in result["messages"]:
            print(Markdown(message.content))
            print("\n\n  ---  \n\n")

        all_messages = result["messages"]

    print(all_messages)

if __name__ == "__main__":
    main()