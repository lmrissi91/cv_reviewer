import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

def load_llm():
    return init_chat_model(
        model=os.getenv("MODEL"),
        model_provider=os.getenv("MODEL_PROVIDER"),
        base_url=os.getenv("LOCALHOST"),
        api_key=os.getenv("API_KEY"),
    )