from src.model.rag.state import State
from src.config.config import Config 
config = Config()

class RAG:
    def __init__(self, vector_store):
        self.prompt = config.prompt
        self.llm = config.llm
        
        self.state = State()
        
        self.vector_store = vector_store

    def retrieve(self):
        retrieved_docs = self.vector_store.similarity_search(self.state["question"])
        self.state["context"] = retrieved_docs
    
    def generate(self):
        docs_content = "\n\n".join(doc.page_content for doc in self.state["context"])
        messages = self.prompt.invoke({"question": self.state["question"], "context": docs_content})
        respone = self.llm.invoke(messages)
        self.state["answer"] = respone.content
