import os 
import getpass
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain import hub


class Config():
    def __init__(self):
        #API key
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = ""

        if not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = ""
        
        self.llm = init_chat_model("gpt-4o-mini", model_provider="openai")
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.prompt = hub.pull("rlm/rag-prompt")
        
        self.chunk_size = 1000
        self.chunk_overlap = 200
        
        self.web_data_paths = ("https://lilianweng.github.io/posts/2023-06-23-agent/",)
        self.pdf_data_path = "data/Understanding_Climate_Change.pdf"