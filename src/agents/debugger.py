import logging
from typing import List

from src.core.models import DataDescription, DebugTrace
from src.core.prompts import (
    DEBUGGER_SUMMARIZE_PROMPT,
    DEBUGGER_FIX_ANALYZER_PROMPT,
    DEBUGGER_FIX_SOLUTION_PROMPT,
)
from src.llm.base import BaseLLMClient


class DebuggerAgent:
    """
    Agent for debugging and fixing code errors.
    
    Implements Section 3.3: Debugging agent.
    """

    def __init__(self, llm_client: BaseLLMClient):
        self.llm = llm_client
        self.logger = logging.getLogger(__name__)

    def summarize_error(self, error: str, filename: str = "script") -> str:
        """
        Summarize an error traceback to remove unnecessary parts.

        Args:
            error: Full error traceback
            filename: Name of the file being debugged

        Returns:
            Summarized error message
        """
        prompt = DEBUGGER_SUMMARIZE_PROMPT.format(bug=error, filename=filename)
        try:
            response = self.llm.chat(prompt)
            return response.strip()
        except Exception as e:
            self.logger.warning(f"Failed to summarize error: {e}")
            return error  # Return original if summarization fails

    def fix_analyzer_script(self, code: str, error: str) -> str:
        """
        Fix an analyzer script that failed to execute.

        This is used when generating data descriptions (Section 3.1).

        Args:
            code: The failing code
            error: Error message or traceback

        Returns:
            Fixed code
        """
        summarized_error = self.summarize_error(error)
        prompt = DEBUGGER_FIX_ANALYZER_PROMPT.format(code=code, bug=summarized_error)

        response = self.llm.chat(prompt)
        return self.llm.extract_code_block(response)

    def fix_solution_script(
        self,
        code: str,
        error: str,
        data_descriptions: List[DataDescription],
    ) -> str:
        """
        Fix a solution script that failed to execute.

        This is used during the iterative planning phase (Section 3.2).

        Args:
            code: The failing code
            error: Error message or traceback
            data_descriptions: Available data descriptions for context

        Returns:
            Fixed code
        """
        summarized_error = self.summarize_error(error)

        # Format data descriptions
        filenames = ", ".join([desc.file.filename for desc in data_descriptions])
        data_desc_text = "\n\n".join(
            [
                f"{desc.file.filename}:\n{desc.description}"
                for desc in data_descriptions
            ]
        )

        prompt = DEBUGGER_FIX_SOLUTION_PROMPT.format(
            filenames=filenames,
            data_descriptions=data_desc_text,
            code=code,
            bug=summarized_error,
        )

        response = self.llm.chat(prompt)
        return self.llm.extract_code_block(response)

    def debug_with_retries(
        self,
        code: str,
        error: str,
        max_retries: int,
        is_analyzer: bool = False,
        data_descriptions: List[DataDescription] = None,
    ) -> str:
        """
        Attempt to fix code with multiple retries.

        Args:
            code: Original failing code
            error: Error message
            max_retries: Maximum retry attempts
            is_analyzer: Whether this is an analyzer script
            data_descriptions: Data descriptions (for solution scripts)

        Returns:
            Fixed code (or original if all retries fail)
        """
        current_code = code
        current_error = error

        for attempt in range(max_retries):
            self.logger.info(f"Debug attempt {attempt + 1}/{max_retries}")

            try:
                if is_analyzer:
                    current_code = self.fix_analyzer_script(current_code, current_error)
                else:
                    current_code = self.fix_solution_script(
                        current_code, current_error, data_descriptions or []
                    )
                return current_code
            except Exception as e:
                self.logger.warning(f"Debug attempt {attempt + 1} failed: {e}")
                current_error = str(e)

        # If all retries fail, return the last attempt
        return current_code

