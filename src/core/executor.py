"""code execution engine for DS-STAR.

Simplified executor that allows LLM-generated code to run with minimal restrictions.
"""

import io
import time
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Dict, Optional

from src.core.models import ExecutionResult


class CodeExecutor:
    """Wrapper for code execution with output capture.
    
    NOTE: All safety restrictions removed
    """

    def __init__(self):
        pass

    def execute(
        self, code: str, globals_dict: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """
        Execute Python code and return results.

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

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, globals_dict)
        except Exception as e:
            returncode = 1
            error = f"{type(e).__name__}: {str(e)}"
            stderr_capture.write(error)

        execution_time = time.time() - start_time

        return ExecutionResult(
            stdout=stdout_capture.getvalue(),
            stderr=stderr_capture.getvalue(),
            returncode=returncode,
            execution_time=execution_time,
            error=error,
        )

