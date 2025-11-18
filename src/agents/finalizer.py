import logging
from typing import List, Optional

from src.core.models import DataDescription
from src.core.prompts import FINALIZER_PROMPT
from src.llm.base import BaseLLMClient


class FinalizerAgent:
    """
    Agent for formatting final solutions according to guidelines.
    
    Implements Section 3.3: Additional modules - ensuring proper output format.
    """

    def __init__(self, llm_client: BaseLLMClient):
        self.llm = llm_client
        self.logger = logging.getLogger(__name__)

    def finalize_solution(
        self,
        question: str,
        data_descriptions: List[DataDescription],
        code: str,
        execution_result: str,
        guidelines: Optional[str] = None,
    ) -> str:
        """
        Finalize the solution code to ensure proper formatting.

        Args:
            question: User's question
            data_descriptions: Available data descriptions
            code: Current solution code
            execution_result: Execution result of the code
            guidelines: Optional formatting guidelines

        Returns:
            Finalized Python code
        """
        # If no guidelines, just return the code
        if not guidelines:
            self.logger.info("No guidelines provided, returning code as-is")
            return code

        self.logger.info("Finalizing solution with guidelines")

        # Format data descriptions
        filenames = ", ".join([desc.file.filename for desc in data_descriptions])
        formatted_desc = "\n\n".join(
            [
                f"{desc.file.filename}:\n{desc.description}"
                for desc in data_descriptions
                if desc.description
            ]
        )

        prompt = FINALIZER_PROMPT.format(
            filenames=filenames,
            data_descriptions=formatted_desc,
            code=code,
            result=execution_result,
            question=question,
            guidelines=guidelines,
        )

        response = self.llm.chat(prompt)
        return self.llm.extract_code_block(response)

