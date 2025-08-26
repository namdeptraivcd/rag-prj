from src.model.rag.generator import Generator
from src.model.rag.retriever import Retriever
from src.model.rag.prompts import Prompt
from src.model.rag.state import State
from src.config.config import Config 

class RAG:
    def __init__(self, vector_store):
        self.prompt = Prompt()
        self.state = State()
        self.vector_store = vector_store
        self.llm = Config()

    def retrieve(self):
        Retriever(self.vector_store, self.state)
    
    def generate(self):
        Generator(self.llm, self.prompt, self.state)
