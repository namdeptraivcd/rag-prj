from src.evaluation.evaluation_runner import EvaluationRunner
from src.config.config import Config

cfg = Config()


def main():
    runner = EvaluationRunner(cfg.experiment_questions)
    
    # Run basic evaluation
    print("Running basic evaluation...")
    results, metrics = runner.run_basic_evaluation()
    runner.save_results({"results": results, "metrics": metrics}, "basic_evaluation.json")
    
    # Run chunk size evaluation
    print("\nRunning chunk size evaluation...")
    chunk_results = runner.run_chunk_size_evaluation([256, 512, 1000, 1500])
    runner.save_results(chunk_results, "chunk_size_evaluation.json")


if __name__ == "__main__":
    main()