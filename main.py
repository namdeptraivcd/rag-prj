from src.model.vectorstore import Vector_store
from src.model.rag.rag import RAG

def main():
    #Load and Index
    vector_store = Vector_store(web_path ="https://lilianweng.github.io/posts/2023-06-23-agent/",)


    #Model
    model=RAG(vector_store)

    
    #Retrieve
    model.retrieve()


    #Generate
    answer = model.generate()
    print(answer)

if __name__ =="__main__":
    main()
