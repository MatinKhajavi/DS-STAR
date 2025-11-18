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
) -> None:
    """
    Log details of an iteration to file.

    Args:
        logger: Logger instance
        iteration: Iteration state
        log_dir: Directory for detailed logs
        question: The question being answered
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

    # Write to JSON file
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
    """
    logger.info(f"Final solution generated after {total_rounds} rounds")

    final_log = {
        "question": question,
        "total_rounds": total_rounds,
        "final_code": code,
        "final_result": result,
        "timestamp": datetime.now().isoformat(),
    }

    log_file = Path(log_dir) / "final_solution.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(final_log, f, indent=2)

