from langchain_openai import ChatOpenAI
class Config:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini") 