from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from src.config.config import Config


cfg = Config()


class GradeProposition(BaseModel):
    accuracy: int = Field(description="Rate from 1-10 based on how well the proposition reflects the original text.")
    
    clarity: int = Field(description="Rate from 1-10 based on how easy it is to understand the proposition without additional context.")
    
    completeness: int = Field(description="Rate from 1-10 based on whether the proposition includes necessary details (e.g., dates, qualifiers).")
    
    conciseness: int = Field(description="Rate from 1-10 based on whether the proposition is concise without losing important information.")

class Proposition_evaluator():
    def __init__(self):
        self.llm = cfg.llm
        self.structured_llm = self.llm.with_structured_output(GradeProposition)
        self.template = """ Evaluate the following proposition based on the criteria below
                                Accuracy: Rate from 1-10 based on how well the proposition reflects the original text.
                                Clarity: Rate from 1-10 based on how easy it is to understand the proposition without additional context.
                                Completeness: Rate from 1-10 based on whether the proposition includes necessary details (e.g., dates, qualifiers).
                                Conciseness: Rate from 1-10 based on whether the proposition is concise without losing important information.
                                
                            Example:
                            Docs: In 1969, Neil Armstrong became the first person to walk on the Moon during the Apollo 11 mission.

                            Propositons_1: Neil Armstrong was an astronaut.
                            Evaluation_1: "accuracy": 10, "clarity": 10, "completeness": 10, "conciseness": 10

                            Propositons_2: Neil Armstrong walked on the Moon in 1969.
                            Evaluation_3: "accuracy": 10, "clarity": 10, "completeness": 10, "conciseness": 10

                            Propositons_3: Neil Armstrong was the first person to walk on the Moon.
                            Evaluation_3: "accuracy": 10, "clarity": 10, "completeness": 10, "conciseness": 10

                            Propositons_4: Neil Armstrong walked on the Moon during the Apollo 11 mission.
                            Evaluation_4: "accuracy": 10, "clarity": 10, "completeness": 10, "conciseness": 10

                            Propositons_5: The Apollo 11 mission occurred in 1969.
                            Evaluation_5: "accuracy": 10, "clarity": 10, "completeness": 10, "conciseness": 10
                            
                            
                            Format:
                                Proposition: {proposition}
                                Original text: {doc}"""
        
        self.prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", self.template),
                        ("human", "{proposition}, {doc}" )
                    ]
        )
        self.proposition_evaluator_chain = self.prompt | self.structured_llm

        self.Filter = {"accuracy": 7, "clarity": 7, "completeness": 7, "conciseness": 7}

        
    def evaluate_proposition(self, proposition, doc):
        respone = self.proposition_evaluator_chain.invoke({"proposition": proposition, "doc": doc })
        scores = {"accuracy": respone.accuracy, "clarity": respone.clarity, "completeness": respone.completeness, "conciseness": respone.conciseness}

        for category, score in scores.items():
            if score < self.Filter[category]:
                return False
            return True
        




        





