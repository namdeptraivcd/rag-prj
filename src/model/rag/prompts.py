from langchain import hub


class Prompt:
    def __init__(self):
        self.prompt = hub.pull("rlm/rag-prompt")

