from src.model.rag import RAG
from langchain_core.messages import HumanMessage


def main():
    # Model
    model = RAG()

    while True:
        # Set question
        question = input("YOU: ")
        model.state["messages"].append(HumanMessage(content = question))

        # Query or respond
        message, answer_type = model.query_or_respond()
        model.state["messages"].extend(message)

        # Generate
        model.generate()

        # Print answer
        print(f"BOT: ({answer_type}) {model.state["answer"]}")


if __name__ =="__main__": 
    main()
