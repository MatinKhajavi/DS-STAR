import logging
from typing import List, Optional

from src.core.models import DataDescription, PlanStep
from src.core.prompts import CODER_INITIAL_PROMPT, CODER_NEXT_PROMPT
from src.llm.base import BaseLLMClient


class CoderAgent:
    """
    Agent for implementing plans as Python code.
    
    Implements the coding component of Section 3.2.
    """

    def __init__(self, llm_client: BaseLLMClient):
        self.llm = llm_client
        self.logger = logging.getLogger(__name__)

    def _format_data_descriptions(self, descriptions: List[DataDescription]) -> tuple[str, str]:
        """
        Format data descriptions for prompts.

        Returns:
            Tuple of (filenames_list, formatted_descriptions)
        """
        filenames = ", ".join([desc.file.filename for desc in descriptions])
        formatted_desc = "\n\n".join(
            [
                f"{desc.file.filename}:\n{desc.description}"
                for desc in descriptions
                if desc.description
            ]
        )
        return filenames, formatted_desc

    def implement_initial_plan(
        self, plan: PlanStep, data_descriptions: List[DataDescription]
    ) -> str:
        """
        Implement the initial plan step as code (s0 in Algorithm 1).

        Args:
            plan: The initial plan step
            data_descriptions: Available data descriptions

        Returns:
            Python code implementing the plan
        """
        self.logger.info("Implementing initial plan")

        filenames, formatted_desc = self._format_data_descriptions(data_descriptions)

        prompt = CODER_INITIAL_PROMPT.format(
            filenames=filenames,
            data_descriptions=formatted_desc,
            plan=str(plan),
        )

        response = self.llm.chat(prompt)
        return self.llm.extract_code_block(response)

    def implement_plan(
        self,
        plan: List[PlanStep],
        data_descriptions: List[DataDescription],
        base_code: Optional[str] = None,
    ) -> str:
        """
        Implement a multi-step plan as code.

        Args:
            plan: List of plan steps
            data_descriptions: Available data descriptions
            base_code: Optional base code from previous steps

        Returns:
            Python code implementing the plan
        """
        # If it's just the initial step or no base code, use initial implementation
        if len(plan) == 1 or base_code is None:
            return self.implement_initial_plan(plan[0], data_descriptions)

        self.logger.info(f"Implementing plan with {len(plan)} steps")

        filenames, formatted_desc = self._format_data_descriptions(data_descriptions)

        # Format previous plans (all but the last)
        previous_plans = "\n".join([str(step) for step in plan[:-1]])

        # Current plan to implement (the last one)
        current_plan = str(plan[-1])

        prompt = CODER_NEXT_PROMPT.format(
            filenames=filenames,
            data_descriptions=formatted_desc,
            base_code=base_code,
            previous_plans=previous_plans,
            current_plan=current_plan,
        )

        response = self.llm.chat(prompt)
        return self.llm.extract_code_block(response)

