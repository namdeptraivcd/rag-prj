from src.config.config import Config


prompt = Config().prompt
class Prompt:
    def __init__(self):
        self.prompt = prompt

