from src.model.vectorstore import Vector_store
from src.model.rag.rag import RAG
import getpass
import os


def main():
    #API key
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass()

    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")


    #Load and Index
    vector_store = Vector_store()
    vector_store.load_web("https://lilianweng.github.io/posts/2023-06-23-agent/")


    #Model
    model=RAG(vector_store)

    
    #Retrieve
    model.retrieve()


    #Generate
    answer = model.generate()
    print(answer)

if __name__ =="__main__":
    main()
