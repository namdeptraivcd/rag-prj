from src.model.vectorstore import Vector_store
from src.model.rag import RAG
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage


# @TODO: deploy this chatbot to web using Next.js
# @TODO: handle large number of users, suggest: use vector database qdrant, milvus
def main():
    # Load and Index
    vs = Vector_store()
    vs.load_web(("https://lilianweng.github.io/posts/2023-06-23-agent/",))

    # Model
    model = RAG(vs.vector_store)
    
    
    #The history of the conversation
    if "messages" not in model.state:
        model.state["messages"] = []
    system_prompt = (
        "You are a retrieval-augmented assistant. "
        "ALWAYS use the retrieve tool to get context from the provided documents before answering any user question. "
        "Do not answer from your own knowledge unless the tool result is empty."
    )
    model.state["messages"].insert(0, SystemMessage(content=system_prompt))

    while True:
        # Set question
        question = input("YOU: ")
        model.state["messages"].append(HumanMessage(content = question))

        # query_or_respone
        message = model.query_or_respond()
        model.state["messages"].extend(message)

        # Generate
        model.generate()

        #Print answer
        print(f"BOT: {model.state["answer"]}")
    # @TODO: check if the answer is retrieved from the corpus or the llm its self?


if __name__ =="__main__": 
    main()
