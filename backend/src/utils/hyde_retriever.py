from src.config.config import Config
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

cfg = Config()


class HyDERetriever: 
    def __init__(self):
        self.document_gen_prompt = """Given the question '{query}', generate a hypothetical document that directly answers this question.
        The document should be detailed and in-depth. The document size has be exactly {chunk_size} characters."""
        self.chunk_size = cfg.chunk_size
        self.embeddings = cfg.embeddings


        self.hyde_prompt = PromptTemplate(
            input_variables = ["query", "chunk_size"],
            template= self.document_gen_prompt,
        )
        
        '''self.hyde_generator = self.hyde_prompt | cfg.llm | StrOutputParser()'''
        def hyde_generator(input_variables: dict):
            prompt_text = self.hyde_prompt.invoke(input_variables)
            result = cfg.llm.invoke(prompt_text)
            return result.content
        
        self.hyde_generator = hyde_generator

    def generate_hypothetical_document(self, query):
        input_variables = {"query": query, "chunk_size": self.chunk_size}
        return self.hyde_generator(input_variables)