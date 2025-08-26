
class Retriever:
    def __init__ (self, vector_store, state):

        retrieved_docs = vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}