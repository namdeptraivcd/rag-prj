from src.config.config import Config
from langchain_core.vectorstores import InMemoryVectorStore
from src.utils.utils import read_source, split_text
config = Config()

class Vector_store:
    def __init__ (self):
        self.embeddings = config.embeddings
        self.vector_store = InMemoryVectorStore(self.embeddings)    

    def load_web(self, web_path):
        docs = read_source(web_path)
        splited_texts = split_text(docs)
        self.vector_store.add_documents(splited_texts)
        return self.vector_store    
