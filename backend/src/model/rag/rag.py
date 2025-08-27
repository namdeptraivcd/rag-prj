from src.model.rag.state.state import State, Search
from src.config.config import Config
config = Config()

class RAG:
    def __init__(self, vector_store):
        self.prompt = config.prompt
        self.llm = config.llm
        
        self.state = State()
        
        self.vector_store = vector_store

    def analyze_query(self):
        structured_llm = config.llm.with_structured_output(Search)
        query = structured_llm.invoke(self.state["question"])
        self.state["query"] = query


    def retrieve(self):
        query = self.state["query"]
        section = query["section"]
        if section:
            retrieved_docs = self.vector_store.similarity_search(
            query["query"],
            filter=lambda doc: doc.metadata.get("section") == query["section"],
    )
        else:
            retrieved_docs = self.vector_store.similarity_search(query["query"])
        self.state["context"] = retrieved_docs

    
    def generate(self):
        docs_content = "\n\n".join(doc.page_content for doc in self.state["context"])
        messages = self.prompt.invoke({"question": self.state["question"], "context": docs_content})
        respone = self.llm.invoke(messages)
        self.state["answer"] = respone.content
