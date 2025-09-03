import json
from datetime import datetime
from src.model.rag import RAG
from src.evaluation.rag_evaluator import RAGEvaluator
from src.evaluation.chunk_size_evaluator import ChunkSizeEvaluator
from src.config.config import Config

cfg = Config()


class EvaluationRunner:
    def __init__(self, questions: list):
        self.evaluator = RAGEvaluator()
        self.chunk_evaluator = ChunkSizeEvaluator()
        self.questions = questions
        
    def run_basic_evaluation(self):
        
        print("Starting basic experiment...")
        print("=" * 60)
        
        model = RAG()
        results = self.evaluator.evaluate_multiple_queries(model, self.questions)
        metrics = self.evaluator.compute_metrics(results)
        
        # Print detailed results
        for i, result in enumerate(results, 1):
            print(f"\nQuestion {i}: {result.question}")
            print(f"Answer: {result.answer}")
            print(f"Response Time: {result.response_time:.2f}s")
            print(f"Relevance Score: {result.relevance_score}")
            print(f"Faithfulness Score: {result.faithfulness_score}")
            print(f"Quality Score: {result.answer_quality_score}/5")
            print("-" * 40)
        
        # Print aggregate metrics
        print(f"\nAGGREGATE METRICS:")
        print(f"Average Response Time: {metrics['average_response_time']:.2f}s")
        print(f"Average Relevance Score: {metrics['average_relevance_score']:.2f}/1.00")
        print(f"Average Faithfulness Score: {metrics['average_faithfulness_score']:.2f}/1.00")
        print(f"Average Answer Quality Score: {metrics['average_answer_quality_score']:.2f}/5.00")
        
        return results, metrics
    
    def run_chunk_size_evaluation(self, chunk_sizes: list = None):
        """Run evaluation across different chunk sizes
        """
        if chunk_sizes is None:
            chunk_sizes = [256, 512, 1000, 1500, 2000]
            
        print("Starting chunk size experiment...")
        print("=" * 60)
        
        results = self.chunk_evaluator.compare_chunk_sizes(self.questions, chunk_sizes)
        
        # Find best chunk size
        best_result = max(results, key=lambda x: x['average_faithfulness_score'])
        print(f"\nBEST CHUNK SIZE: {best_result['chunk_size']}")
        print(f"Best Average Relevance Score: {best_result['average_relevance_score']:.2f}/1.00")
        print(f"Best Average Faithfulness Score: {best_result['average_faithfulness_score']:.2f}/1.00")
        print(f"Best Average Quality Score: {best_result['average_answer_quality_score']:.2f}/5.00")
        
        return results
    
    # @TODO
    def run_query_transformation_evaluation(self):
        pass
    
    # @TODO
    def run_hype_evaluation(self):
        pass
    
    def save_results(self, results, filename: str = None):
        """Save evaluation results to JSON file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = cfg.results_path + f"_{timestamp}_" + filename
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Results saved to {filename}")