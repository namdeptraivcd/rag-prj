from src.model.rag import RAG
from src.model.vectorstore import Vector_store

def main():
    # Model
    model = RAG()

    while True:
        # Set question
        # Example: What contribution did the son of Euler's teacher make?
        query = input("YOU: ")
        
        # Query
        model.query(query)

        # Print answer
        print(f"BOT: ({model.state['answer_type']}) {model.state['answer']}")
if __name__ =="__main__": 
    main()
