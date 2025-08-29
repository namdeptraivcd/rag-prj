from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Cấu hình CORS cho phép frontend truy cập
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
    # TODO: Gọi RAG model để lấy câu trả lời thực tế
    # answer = model.answer(question)
    answer = f"Bạn hỏi: {question}"  # Trả về tạm thời
    return {"response": answer}