"""
Run DS-STAR on DABStep benchmark.

DABStep benchmark: https://huggingface.co/datasets/adyen/DABstep
Leaderboard: https://huggingface.co/spaces/adyen/DABstep
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import datasets
import pandas as pd
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv

from src import DSStar
from src.config import DSStarConfig, LLMConfig

# Load environment variables
load_dotenv()


# Context files needed for DABStep (with their repo paths)
CONTEXT_FILES_REPO = [
    "data/context/acquirer_countries.csv",
    "data/context/payments-readme.md",
    "data/context/payments.csv",
    "data/context/merchant_category_codes.csv",
    "data/context/fees.json",
    "data/context/merchant_data.json",
    "data/context/manual.md",
]

# Local filenames (what we'll store them as)
CONTEXT_FILES_LOCAL = [
    "acquirer_countries.csv",
    "payments-readme.md",
    "payments.csv",
    "merchant_category_codes.csv",
    "fees.json",
    "merchant_data.json",
    "manual.md",
]


def setup_data_directory() -> str:
    """
    Download context files from DABStep dataset to ./data/
    
    Returns:
        Data directory path
    """
    print("📥 Downloading DABStep context files to ./data/...")
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Download files and move them to data/ (flattening the structure)
    for repo_path, local_name in zip(CONTEXT_FILES_REPO, CONTEXT_FILES_LOCAL):
        # Download from HuggingFace (this will cache)
        downloaded_path = hf_hub_download(
            repo_id="adyen/DABstep",
            repo_type="dataset",
            filename=repo_path,
            force_download=False,
        )
        
        # Copy to our data/ directory with simple filename
        import shutil
        target_path = os.path.join("data", local_name)
        shutil.copy2(downloaded_path, target_path)
        
        if os.path.exists(target_path):
            print(f"✓ data/{local_name}")
        else:
            print(f"✗ data/{local_name} failed!")
    
    return "data"


def run_ds_star_on_task(agent: DSStar, task: dict) -> dict:
    """
    Run DS-STAR on a single task.
    
    Args:
        agent: DS-STAR instance (already configured with data_dir)
        task: Task dictionary with question, guidelines, task_id
        
    Returns:
        Dictionary with task_id, agent_answer, and logs
    """
    task_id = task['task_id']
    question = task['question']
    guidelines = task.get('guidelines', '')
    
    print(f"\n{'='*80}")
    print(f"Task {task_id}: {question[:100]}...")
    print(f"{'='*80}")
    
    try:
        # Run DS-STAR (uses data_dir from config)
        start_time = time.time()
        final_code, final_result = agent.run(
            question=question,
            guidelines=guidelines if guidelines else None,
            query_id=f"task_{task_id}",
        )
        elapsed_time = time.time() - start_time
        
        print(f"\n✓ Completed in {elapsed_time:.2f}s")
        print(f"Answer: {final_result[:200]}...")
        
        return {
            "task_id": str(task_id),
            "agent_answer": str(final_result),
            "final_code": final_code,
            "elapsed_time": elapsed_time,
            "success": True,
        }
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        return {
            "task_id": str(task_id),
            "agent_answer": "",
            "error": str(e),
            "success": False,
        }


def run_benchmark(
    split: str = "dev",
    max_tasks: int = None,
    model: str = "gemini-2.5-pro",
    provider: str = "gemini",
    max_rounds: int = 20,
) -> tuple[list[dict], str]:
    """
    Run DS-STAR on DABStep benchmark.
    
    Args:
        split: Dataset split ('dev' or 'default')
        max_tasks: Maximum number of tasks to run (None = all)
        model: Model name
        provider: LLM provider
        max_rounds: Max refinement rounds
        
    Returns:
        Tuple of (agent_answers, run_id)
    """
    # Setup data directory (downloads to ./data/context/)
    data_dir = setup_data_directory()
    
    # Load dataset from HuggingFace
    print(f"\n📊 Loading DABStep dataset (split: {split})...")
    if max_tasks:
        dataset = datasets.load_dataset(
            "adyen/DABstep", 
            name="tasks", 
            split=f"{split}[:{max_tasks}]"
        )
    else:
        dataset = datasets.load_dataset(
            "adyen/DABstep", 
            name="tasks", 
            split=split
        )
    
    print(f"✓ Loaded {len(dataset)} tasks")
    
    # Initialize DS-STAR
    print(f"\n🤖 Initializing DS-STAR...")
    print(f"   Provider: {provider}")
    print(f"   Model: {model}")
    print(f"   Max rounds: {max_rounds}")
    print(f"   Data directory: {data_dir}")
    
    config = DSStarConfig(
        llm=LLMConfig(provider=provider, model=model),
        max_rounds=max_rounds,
        data_dir=data_dir,
        log_dir="logs/dabstep",
    )
    agent = DSStar(config)
    
    # Analyze data files once (will be cached for all tasks)
    print(f"\n📊 Analyzing data files (one-time setup)...")
    agent.prepare_data()
    print(f"✓ Data analysis complete and cached")
    
    # Run on all tasks
    print(f"\n🚀 Running DS-STAR on {len(dataset)} tasks...")
    agent_answers = []
    
    for i, task in enumerate(dataset):
        print(f"\n[{i+1}/{len(dataset)}]", end=" ")
        try:
            result = run_ds_star_on_task(agent, task)
            agent_answers.append(result)
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user. Saving partial results...")
            break
        except Exception as e:
            print(f"\n✗ Unexpected error in task loop: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            agent_answers.append({
                "task_id": str(task.get('task_id', 'unknown')),
                "agent_answer": "",
                "error": f"Loop error: {type(e).__name__}: {str(e)}",
                "success": False,
            })
            continue
    
    # Generate run ID
    run_id = f"ds-star_{split}_{int(time.time())}"
    
    return agent_answers, run_id


def save_results(agent_answers: list[dict], run_id: str, output_dir: str = "./runs"):
    """Save results to JSONL file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"{run_id}.jsonl"
    
    with open(output_file, "w") as f:
        for answer in agent_answers:
            f.write(json.dumps(answer) + "\n")
    
    print(f"\n💾 Results saved to: {output_file}")
    return output_file


def save_submission(agent_answers: list[dict], run_id: str, output_dir: str = "./runs"):
    """
    Save results in DABStep submission format.
    
    Submission format (JSONL):
    - task_id: required
    - agent_answer: required
    - reasoning_trace: optional
    
    Args:
        agent_answers: List of agent answer dictionaries
        run_id: Run identifier
        output_dir: Output directory
        
    Returns:
        Path to submission file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    submission_file = output_dir / f"{run_id}_submission.jsonl"
    
    with open(submission_file, "w") as f:
        for answer in agent_answers:
            submission_entry = {
                "task_id": answer["task_id"],
                "agent_answer": answer.get("agent_answer", ""),
            }

            if "final_code" in answer:
                submission_entry["reasoning_trace"] = answer["final_code"]
            
            f.write(json.dumps(submission_entry) + "\n")
    
    print(f"📤 Submission file saved to: {submission_file}")
    return submission_file


def evaluate_results(agent_answers: list[dict], split: str = "dev"):
    """
    Evaluate results using DABStep evaluation logic.
    
    Args:
        agent_answers: List of agent answers
        split: Dataset split used
    """
    try:
        from dabstep_benchmark.utils import evaluate
        
        print(f"\n📈 Evaluating results...")
        
        tasks_df = datasets.load_dataset(
            "adyen/DABstep", 
            name="tasks", 
            split=split
        ).to_pandas()
        
        agent_answers_df = pd.DataFrame(agent_answers)
        
        task_scores = evaluate(
            agent_answers=agent_answers_df, 
            tasks_with_gt=tasks_df
        )
        
        task_scores_df = pd.DataFrame(task_scores)
        task_scores_df["correct_answer"] = tasks_df["answer"]
        task_scores_df["question"] = tasks_df["question"]
        
        print(f"\n{'='*80}")
        print("EVALUATION RESULTS")
        print(f"{'='*80}")
        print(f"Total tasks: {len(task_scores_df)}")
        print(f"Correct: {task_scores_df['is_correct'].sum()}")
        print(f"Accuracy: {task_scores_df['is_correct'].mean():.2%}")
        print(f"{'='*80}\n")
        
        print(task_scores_df[['task_id', 'is_correct', 'question']])
        
        return task_scores_df
        
    except ImportError:
        print("\n⚠️  dabstep_benchmark not installed. Skipping evaluation.")
        return None


def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run DS-STAR on DABStep benchmark")
    parser.add_argument(
        "--split", 
        type=str, 
        default="dev", 
        choices=["dev", "default"],
        help="Dataset split to use"
    )
    parser.add_argument(
        "--max-tasks", 
        type=int, 
        default=None,
        help="Maximum number of tasks to run (default: all)"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="gemini-2.5-pro",
        help="Model name"
    )
    parser.add_argument(
        "--provider", 
        type=str, 
        default="gemini",
        choices=["gemini", "openai", "anthropic"],
        help="LLM provider"
    )
    parser.add_argument(
        "--max-rounds", 
        type=int, 
        default=20,
        help="Maximum refinement rounds"
    )
    parser.add_argument(
        "--evaluate", 
        action="store_true",
        help="Run evaluation after completion"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("DS-STAR on DABStep Benchmark")
    print("="*80)
    
    start_time = time.time()
    agent_answers, run_id = run_benchmark(
        split=args.split,
        max_tasks=args.max_tasks,
        model=args.model,
        provider=args.provider,
        max_rounds=args.max_rounds,
    )
    total_time = time.time() - start_time
    
    output_file = save_results(agent_answers, run_id)
    submission_file = save_submission(agent_answers, run_id)
    
    successful = sum(1 for a in agent_answers if a.get("success", False))
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total tasks: {len(agent_answers)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(agent_answers) - successful}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Avg time per task: {total_time/len(agent_answers):.2f}s")
    print(f"Full results: {output_file}")
    print(f"Submission file: {submission_file}")
    print(f"{'='*80}\n")
    
    if args.evaluate:
        evaluate_results(agent_answers, args.split)


if __name__ == "__main__":
    main()

