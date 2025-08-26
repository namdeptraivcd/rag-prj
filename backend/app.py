from fastapi import FastAPI, Request
from src.model.vectorstore import Vector_store
from src.model.rag.rag import RAG
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize vector store with Qdrant/Milvus here
vs = Vector_store()
vs.load_web(["https://lilianweng.github.io/posts/2023-06-23-agent/"])
model = RAG(vs.vector_store)

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    question = data.get("question")
    model.state["question"] = question
    model.retrieve()
    model.generate()
    return {"answer": model.state["answer"]}