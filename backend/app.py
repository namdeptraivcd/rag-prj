from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Configure CORS to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    question = data.get("question", "")
    # TODO: Call RAG model to get actual answer
    # answer = model.answer(question)
    answer = f"You asked: {question}"  # Temporary response
    return {"response": answer}