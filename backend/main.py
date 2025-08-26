from .src.model.vectorstore import Vector_store
from .src.model.rag.rag import RAG


# @TODO: deploy this chatbot to web using Next.js
# @TODO: handle large number of users, suggest: use vector database qdrant, milvus
def main():
    # Load and Index
    vs = Vector_store()
    vs.load_web(("https://lilianweng.github.io/posts/2023-06-23-agent/",))

    # Model
    model = RAG(vs.vector_store)

    # Set question
    model.state["question"] = input("Enter your question: ")

    # Retrieve
    model.retrieve()

    # Generate
    model.generate()
    
    # Print answer
    print(model.state["answer"])
    # @TODO: is the answer retrieved from the corpus or the answer of the llm its self?

if __name__ =="__main__":
    main()
