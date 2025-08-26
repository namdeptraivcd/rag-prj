from src.config.config import Config
from langchain_core.vectorstores import InMemoryVectorStore


embeddings = Config().embeddings
class Vector_store:
    def __init__ (self):


        self.vector_store = InMemoryVectorStore(embeddings)        
