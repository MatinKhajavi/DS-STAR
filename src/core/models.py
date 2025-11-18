from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Any


class VerificationStatus(Enum):
    """Status of plan verification."""
    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"


class RouterDecision(Enum):
    """Router decision types."""
    ADD_STEP = "add_step"
    REMOVE_STEP = "remove_step"


@dataclass
class DataFile:
    """Represents a data file to be analyzed."""
    
    path: str
    filename: str
    extension: str


@dataclass
class DataDescription:
    """Description extracted from a data file."""
    
    file: DataFile
    script: str
    description: str
    error: Optional[str] = None


@dataclass
class PlanStep:
    """A single step in the analysis plan."""
    
    step_number: int
    description: str
    
    def __str__(self) -> str:
        return f"{self.step_number}. {self.description}"


@dataclass
class ExecutionResult:
    """Result from executing a Python script."""
    
    stdout: str
    stderr: str
    returncode: int
    execution_time: float
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        return self.returncode == 0 and self.error is None
    
    @property
    def output(self) -> str:
        """Combined output for display."""
        return self.stdout if self.success else (self.error or self.stderr)


@dataclass
class CodeSnippet:
    """A piece of generated code."""
    
    code: str
    language: str = "python"


@dataclass
class DebugTrace:
    """Debugging information for a failed execution."""
    
    original_code: str
    error_message: str
    traceback: str
    attempt_number: int


@dataclass
class VerificationResult:
    """Result of verifying a plan's sufficiency."""
    
    status: VerificationStatus
    reasoning: Optional[str] = None


@dataclass
class RouterResult:
    """Result from the router agent."""
    
    decision: RouterDecision
    step_to_remove: Optional[int] = None  # If decision is REMOVE_STEP
    reasoning: Optional[str] = None


@dataclass
class AgentResponse:
    """Generic response from an agent."""
    
    content: str
    raw_response: Any = None
    metadata: dict = field(default_factory=dict)


@dataclass
class IterationState:
    """State of a single iteration in the DS-STAR loop."""
    
    round_number: int
    plan: List[PlanStep]
    code: str
    execution_result: ExecutionResult
    verification: Optional[VerificationResult] = None
    router_result: Optional[RouterResult] = None

