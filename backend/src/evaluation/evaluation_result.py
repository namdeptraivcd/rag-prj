from typing import List
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    question: str
    answer: str
    
    retrieved_docs: List
    
    response_time: float
    
    relevance_score: str
    faithfulness_score: str
    answer_quality_score: int
