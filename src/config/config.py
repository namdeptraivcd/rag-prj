from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings

class Config():
    def __init__(self):
        self.llm = init_chat_model("gpt-4o-mini", model_provider="openai")
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

