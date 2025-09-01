from src.evaluation.evaluation_runner import EvaluationRunner


def main():
    runner = EvaluationRunner()
    
    # Run basic evaluation
    print("Running basic evaluation...")
    results, metrics = runner.run_basic_evaluation()
    runner.save_results({"results": results, "metrics": metrics})


if __name__ == "__main__":
    main()