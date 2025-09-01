from typing import List
from src.model.rag import RAG
from src.evaluation.rag_evaluator import RAGEvaluator
from src.config.config import Config

cfg = Config()


class ChunkSizeEvaluator:
    def __init__(self):
        self.evaluator = RAGEvaluator()
    
    def evaluate_chunk_size(self, questions, chunk_size: int, chunk_overlap: int = None) -> dict:
        """Evaluate performance for a specific chunk size
        """
        
        if chunk_overlap is None:
            chunk_overlap = chunk_size // 5
            
        # Temporarily modify config
        original_chunk_size = cfg.chunk_size
        original_chunk_overlap = cfg.chunk_overlap
        
        cfg.chunk_size = chunk_size
        cfg.chunk_overlap = chunk_overlap
        
        try:
            # Create new RAG model with updated chunk size
            model = RAG()
            
            # Evaluate
            results = self.evaluator.evaluate_multiple_queries(model, questions)
            metrics = self.evaluator.compute_metrics(results)
            
            return {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                **metrics
            }
            
        finally:
            # Restore original config
            cfg.chunk_size = original_chunk_size
            cfg.chunk_overlap = original_chunk_overlap
    
    def compare_chunk_sizes(self, questions, chunk_sizes: List[int]) -> List[dict]:
        """Compare performance across different chunk sizes
        """
        results = []
        
        for chunk_size in chunk_sizes:
            print(f"Evaluating chunk size: {chunk_size}")
            result = self.evaluate_chunk_size(questions, chunk_size)
            results.append(result)
            
            print(f"Results for chunk size {chunk_size}:")
            print(f"Average Response Time: {result['average_response_time']:.2f}s")
            print(f"Average Relevance Score: {result['average_relevance_score']:.2f}/1.00")
            print(f"Average Faithfulness Score: {result['average_faithfulness_score']:.2f}/1.00")
            print(f"Average Answer Quality Score: {result['average_answer_quality_score']:.2f}/5.00")
            print("-" * 50)
            
        return results