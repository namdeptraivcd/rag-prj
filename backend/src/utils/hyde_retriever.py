from src.config.config import Config
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


cfg = Config()


class HyDERetriever: #Debug
    def __init__(self):
        self.document_gen_prompt = """Given the question '{query}', generate a hypothetical document that directly answers this question.
        The document should be detailed and in-depth. the document size has be exactly {chunk_size} characters."""
        self.chunk_size = cfg.chunk_size
        self.chunk_overlap = cfg.chunk_overlap
        self.embeddings = cfg.embeddings


        self.hyde_prompt = PromptTemplate(
            input_variables = ["query", "chunk_size"],
            template= self.document_gen_prompt,
        )
        self.hyde_chain = self.hyde_prompt | cfg.llm | StrOutputParser()

    def generate_hypothetical_document(self, query):
        input_variables = {"query": query, "chunk_size": self.chunk_size}
        return self.hyde_chain.invoke(input_variables)

