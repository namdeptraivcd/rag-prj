class Generator():
    def __init__(self, llm, prompt, state):
        self.docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        self.messages = prompt.invoke({"question": state["question"], "context": docs_content})
        self.respone = llm.invoke(self.messages)
        return {"answer": self.respone.content}