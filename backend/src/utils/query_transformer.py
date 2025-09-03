from langchain.prompts import PromptTemplate
from src.config.config import Config

cfg = Config()


class QueryTransformer:
    def __init__(self):
        self.llm = cfg.llm
        
        # Query Rewriting
        self.query_rewrite_template = """You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system. 
            Given the original query, rewrite it to be more specific, detailed, and likely to retrieve relevant information.

            Original query: {original_query}

            Rewritten query:"""
        
        self.query_rewrite_prompt = PromptTemplate(
            input_variables=["original_query"],
            template=self.query_rewrite_template
        )
        self.query_rewriter = self.query_rewrite_prompt | self.llm
        
        # Step-back Prompting
        self.step_back_template = """You are an AI assistant tasked with generating broader, more general queries to improve context retrieval in a RAG system.
            Given the original query, generate a step-back query that is more general and can help retrieve relevant background information.

            Original query: {original_query}

            Step-back query:"""
        
        self.step_back_prompt = PromptTemplate(
            input_variables=["original_query"],
            template=self.step_back_template
        )
        self.step_back_chain = self.step_back_prompt | self.llm
        
        # Sub-query Decomposition
        self.subquery_decomposition_template = """You are an AI assistant tasked with breaking down complex queries into simpler sub-queries for a RAG system.
            Given the original query, decompose it into 2-4 simpler sub-queries that, when answered together, would provide a comprehensive response to the original query.

            Original query: {original_query}

            example: What are the impacts of climate change on the environment?

            Sub-queries:
            1. What are the impacts of climate change on biodiversity?
            2. How does climate change affect the oceans?
            3. What are the effects of climate change on agriculture?
            4. What are the impacts of climate change on human health?"""

        self.subquery_decomposition_prompt = PromptTemplate(
            input_variables=["original_query"],
            template=self.subquery_decomposition_template
        )
        self.subquery_decomposer_chain = self.subquery_decomposition_prompt | self.llm

    def rewrite_query(self, original_query):
        response = self.query_rewriter.invoke({"original_query": original_query})
        return response.content
    
    def generate_step_back_query(self, original_query):
        response = self.step_back_chain.invoke({"original_query": original_query})
        return response.content
    
    def decompose_query(self, original_query):
        response = self.subquery_decomposer_chain.invoke({"original_query": original_query}).content
        sub_queries = [q.strip() for q in response.split('\n') if q.strip() and not q.strip().startswith('Sub-queries:')]
        return sub_queries