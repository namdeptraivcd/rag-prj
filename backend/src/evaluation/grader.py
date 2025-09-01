from pydantic import BaseModel, Field


class RelevanceGrader(BaseModel):
    """Binary score for relevance check on retrieved documents 
    """
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


class FaithfulnessGrader(BaseModel):
    """Binary score for hallucination present in genearted answer
    """
    binary_score: str = Field(
        description="Answer is based on the ground truth, 'yes' or 'no'"
    )


class AnswerQualityGrader(BaseModel):
    """Score for overall answer quality
    """
    score: int = Field(
        description="Answer score range from 1 to 5, where 5 is excellent"
    )