from src.config.config import Config
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.prompts import PromptTemplate
from src.evaluation.proposition_evaluator import Proposition_evaluator


evaluator = Proposition_evaluator()
cfg = Config()


class GenerateProposition(BaseModel):
    """List of all the proposition in a given document"""

    propositions: list[str] = Field(
        description="List of propositions (factual, self-contained, and concise information)"
    )


class Proposition_chunk_Gen():
    def __init__(self):
        self.template = """Please break down the following text into simple, self-contained proposition. Ensure that each proposition meets the following criteria:
        
        1. Express a Single Fact: Each proposition should state one specific fact or claim.
        2. Be Understandable Without Context: The proposition should be self-contained, meaning it can be understood without needing additional context.
        3. Use Full Names, Not Pronouns: Avoid pronouns or ambiguous references; use full entity names.
        4. Include Relevant Dates/Qualifiers: If applicable, include necessary dates, times, and qualifiers to make the fact precise.
        5. Contain One Subject-Predicate Relationship: Focus on a single subject and its corresponding action or attribute, without conjunctions or multiple clauses.
        
        Text:
        {splited_doc}"""
        
        self.llm = cfg.llm
        self.structured_llm = self.llm.with_structured_output(GenerateProposition)

        
        self.pc_prompt = PromptTemplate(
            input_variable = ["splited_doc"],
            template = self.template
            )
        self.proposition_generator = self.pc_prompt | self.structured_llm

        #Init Proposition chunking
        self.proposition = []
    def generate_raw_proposition(self, splited_docs):
        for splited_doc in splited_docs:
                result = self.proposition_generator.invoke({"splited_doc": splited_doc})
                for proposition in result.propositions:
                    self.proposition.append(proposition)
        
        #Init list satisfied proposition
        self.evaluated_proposition = []
    def generate_evaluated_proposition(self, splited_docs):

        self.generate_raw_proposition(splited_docs)

        for i, proposition in enumerate (self.proposition):
            doc = splited_docs[i % len(splited_docs)]
            Satisfied = evaluator.evaluate_proposition(proposition, doc=doc)
            if Satisfied:
                self.evaluated_proposition.append(proposition)
        return self.evaluated_proposition

             


            