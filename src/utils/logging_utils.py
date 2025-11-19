import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from src.core.models import IterationState


def setup_logger(name: str, log_dir: str, debug: bool = False) -> logging.Logger:
    """
    Set up a logger with file and console handlers.

    Args:
        name: Logger name
        log_dir: Directory for log files
        debug: Enable debug logging

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    # Create log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # File handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"{name}_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def log_iteration(
    logger: logging.Logger,
    iteration: IterationState,
    log_dir: str,
    question: str,
    query_id: Optional[str] = None,
) -> None:
    """
    Log details of an iteration to file.

    Args:
        logger: Logger instance
        iteration: Iteration state
        log_dir: Directory for detailed logs
        question: The question being answered
        query_id: Optional query identifier for subdirectory organization
    """
    logger.info(f"Round {iteration.round_number}: {len(iteration.plan)} steps in plan")

    # Save detailed iteration log
    iteration_log = {
        "round": iteration.round_number,
        "question": question,
        "plan": [str(step) for step in iteration.plan],
        "code": iteration.code,
        "execution": {
            "stdout": iteration.execution_result.stdout,
            "stderr": iteration.execution_result.stderr,
            "success": iteration.execution_result.success,
            "time": iteration.execution_result.execution_time,
        },
        "verification": (
            {
                "status": iteration.verification.status.value,
                "reasoning": iteration.verification.reasoning,
            }
            if iteration.verification
            else None
        ),
        "router": (
            {
                "decision": iteration.router_result.decision.value,
                "step_to_remove": iteration.router_result.step_to_remove,
                "reasoning": iteration.router_result.reasoning,
            }
            if iteration.router_result
            else None
        ),
    }

    if query_id:
        query_log_dir = Path(log_dir) / query_id
        query_log_dir.mkdir(parents=True, exist_ok=True)
        log_file = query_log_dir / f"iteration_{iteration.round_number}.json"
    else:
        log_file = Path(log_dir) / f"iteration_{iteration.round_number}.json"
    
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(iteration_log, f, indent=2)


def log_final_solution(
    logger: logging.Logger,
    log_dir: str,
    question: str,
    code: str,
    result: str,
    total_rounds: int,
    query_id: Optional[str] = None,
) -> None:
    """
    Log the final solution.

    Args:
        logger: Logger instance
        log_dir: Directory for logs
        question: The question
        code: Final code
        result: Final result
        total_rounds: Total number of rounds
        query_id: Optional query identifier for subdirectory organization
    """
    logger.info(f"Final solution generated after {total_rounds} rounds")

    final_log = {
        "question": question,
        "total_rounds": total_rounds,
        "final_code": code,
        "final_result": result,
        "timestamp": datetime.now().isoformat(),
    }

    if query_id:
        query_log_dir = Path(log_dir) / query_id
        query_log_dir.mkdir(parents=True, exist_ok=True)
        log_file = query_log_dir / "final_solution.json"
    else:
        log_file = Path(log_dir) / "final_solution.json"
    
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(final_log, f, indent=2)


def save_data_descriptions(
    descriptions: list,
    log_dir: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Save data descriptions to disk for inspection and caching.

    Args:
        descriptions: List of DataDescription objects
        log_dir: Directory for logs
        logger: Optional logger instance
    """
    from src.core.models import DataDescription
    
    descriptions_data = []
    for desc in descriptions:
        descriptions_data.append({
            "filename": desc.file.filename,
            "path": desc.file.path,
            "extension": desc.file.extension,
            "description": desc.description,
            "script": desc.script,
            "error": desc.error,
        })

    log_file = Path(log_dir) / "data_descriptions.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_files": len(descriptions),
            "successful": sum(1 for d in descriptions if not d.error),
            "descriptions": descriptions_data,
        }, f, indent=2)
    
    if logger:
        logger.info(f"Saved data descriptions to {log_file}")


def log_prompt(
    prompt: str,
    agent_name: str,
    step: str,
    log_dir: str,
    query_id: Optional[str] = None,
) -> None:
    """
    Save LLM prompts to disk for inspection and debugging.
    
    Args:
        prompt: The full prompt being sent to the LLM
        agent_name: Name of the agent (e.g., "planner", "coder")
        step: Description of the step (e.g., "initial_plan", "fix_code")
        log_dir: Directory for logs
        query_id: Optional query identifier
    """
    if query_id:
        prompt_dir = Path(log_dir) / query_id / "prompts"
    else:
        prompt_dir = Path(log_dir) / "prompts"
    
    prompt_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%H%M%S")
    filename = f"{timestamp}_{agent_name}_{step}.txt"
    prompt_file = prompt_dir / filename
    
    with open(prompt_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"AGENT: {agent_name}\n")
        f.write(f"STEP: {step}\n")
        f.write(f"TIMESTAMP: {datetime.now().isoformat()}\n")
        f.write("=" * 80 + "\n\n")
        f.write(prompt)
        f.write("\n\n")
        f.write("=" * 80 + "\n")


def load_data_descriptions(
    log_dir: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[list]:
    """
    Load cached data descriptions from disk if available.

    Args:
        log_dir: Directory containing cached descriptions
        logger: Optional logger instance

    Returns:
        List of DataDescription objects if cache exists, None otherwise
    """
    from src.core.models import DataDescription, DataFile
    
    cache_file = Path(log_dir) / "data_descriptions.json"
    
    if not cache_file.exists():
        if logger:
            logger.info("No cached data descriptions found")
        return None
    
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            cache_data = json.load(f)
        
        descriptions = []
        for desc_data in cache_data.get("descriptions", []):
            data_file = DataFile(
                path=desc_data["path"],
                filename=desc_data["filename"],
                extension=desc_data["extension"],
            )
            
            description = DataDescription(
                file=data_file,
                script=desc_data["script"],
                description=desc_data["description"],
                error=desc_data.get("error"),
            )
            descriptions.append(description)
        
        if logger:
            logger.info(f"Loaded {len(descriptions)} cached data descriptions from {cache_file}")
            logger.info(f"Cache timestamp: {cache_data.get('timestamp', 'unknown')}")
        
        return descriptions
    
    except Exception as e:
        if logger:
            logger.warning(f"Failed to load cached descriptions: {e}")
        return None

