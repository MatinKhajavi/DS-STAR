"""code execution engine for DS-STAR.

Simplified executor that allows LLM-generated code to run with minimal restrictions.
"""

import io
import time
import signal
import traceback
import sys
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Dict, Optional

from src.core.models import ExecutionResult


class TimeoutException(Exception):
    """Raised when code execution exceeds timeout."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for execution timeout."""
    raise TimeoutException("Code execution timed out")


class CodeExecutor:
    """Wrapper for code execution with output capture.
    
    NOTE: All safety restrictions removed for DS-STAR LLM-generated code execution.
    """

    def __init__(self, timeout: int = 300):
        """
        Initialize code executor.
        
        Args:
            timeout: Maximum execution time in seconds (default: 300 = 5 minutes)
        """
        self.timeout = timeout

    def execute(
        self, code: str, globals_dict: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """
        Execute Python code and return results with timeout protection.

        Args:
            code: Python code to execute
            globals_dict: Optional global variables to inject

        Returns:
            ExecutionResult with stdout, stderr, returncode, and timing
        """
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        start_time = time.time()
        returncode = 0
        error = None

        if globals_dict is None:
            globals_dict = {}
        
        if '__builtins__' not in globals_dict:
            globals_dict['__builtins__'] = __builtins__
        
        globals_dict['__name__'] = '__main__'

        old_handler = None
        try:
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.timeout)
        except (AttributeError, ValueError):
            pass

        print(f"[EXECUTOR] Starting code execution (timeout={self.timeout}s)...", file=sys.stderr, flush=True)
        
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, globals_dict)
            print(f"[EXECUTOR] Code execution completed successfully", file=sys.stderr, flush=True)
        except TimeoutException as e:
            returncode = 1
            error = f"TimeoutException: Code execution exceeded {self.timeout} seconds"
            stderr_capture.write(error)
        except KeyboardInterrupt:
            returncode = 1
            error = "KeyboardInterrupt: Execution interrupted by user"
            stderr_capture.write(error)
            raise
        except SystemExit as e:
            returncode = 1
            error = f"SystemExit: Code called sys.exit({e.code})"
            stderr_capture.write(error)
        except Exception as e:
            returncode = 1
            error = f"{type(e).__name__}: {str(e)}"
            stderr_capture.write(error)
            full_trace = traceback.format_exc()
            stderr_capture.write(f"\n\nFull traceback:\n{full_trace}")
        finally:
            try:
                signal.alarm(0)
                if old_handler is not None:
                    signal.signal(signal.SIGALRM, old_handler)
            except (AttributeError, ValueError):
                pass

        execution_time = time.time() - start_time

        return ExecutionResult(
            stdout=stdout_capture.getvalue(),
            stderr=stderr_capture.getvalue(),
            returncode=returncode,
            execution_time=execution_time,
            error=error,
        )

