from typing_extensions import TypedDict, List
from langchain_core.documents import Document
from typing import Literal
from typing_extensions import Annotated


class Search(TypedDict):
    """Search query"""
    query: Annotated[str, ..., "Search query to run"]
    section: Annotated[Literal["beginning", "middle", "end"], ..., "Section to query"]
    
    
class State(TypedDict):
    question: str
    query: Search
    context: List[Document]
    answer: str

