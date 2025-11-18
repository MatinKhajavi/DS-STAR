from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMConfig:
    """Configuration for LLM provider."""
    
    provider: str = "gemini"  # gemini, openai, anthropic
    model: str = "gemini-2.5-pro"
    api_key: Optional[str] = None


@dataclass
class ExecutionConfig:
    """Configuration for code execution."""
    
    max_retries: int = 3
    capture_output: bool = True


@dataclass
class DSStarConfig:
    """Main configuration for DS-STAR."""
    
    llm: LLMConfig = field(default_factory=LLMConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    max_rounds: int = 20
    data_dir: str = "data"
    retrieval_threshold: int = 100  # not implemented
    top_k_files: int = 100  # not implemented
    debug_mode: bool = False
    log_dir: str = "logs"

