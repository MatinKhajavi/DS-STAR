import logging
from typing import List

from src.core.models import DataDescription, PlanStep
from src.core.prompts import PLANNER_INITIAL_PROMPT, PLANNER_NEXT_PROMPT
from src.llm.base import BaseLLMClient


class PlannerAgent:
    """
    Agent for planning analysis steps.
    
    Implements the planning component of Section 3.2: Iterative plan generation.
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
                if desc.description  # Skip empty descriptions
            ]
        )
        return filenames, formatted_desc

    def generate_initial_plan(
        self, question: str, data_descriptions: List[DataDescription]
    ) -> PlanStep:
        """
        Generate the initial plan step (p0 in Algorithm 1).

        Args:
            question: User's question
            data_descriptions: Available data descriptions

        Returns:
            Initial plan step
        """
        self.logger.info("Generating initial plan")

        filenames, formatted_desc = self._format_data_descriptions(data_descriptions)

        prompt = PLANNER_INITIAL_PROMPT.format(
            question=question,
            filenames=filenames,
            data_descriptions=formatted_desc,
        )

        response = self.llm.chat(prompt)
        return PlanStep(step_number=0, description=response.strip())

    def generate_next_step(
        self,
        question: str,
        data_descriptions: List[DataDescription],
        current_plan: List[PlanStep],
        current_result: str,
    ) -> PlanStep:
        """
        Generate the next plan step (p_{k+1} in Algorithm 1).

        Args:
            question: User's question
            data_descriptions: Available data descriptions
            current_plan: Current plan steps
            current_result: Result from executing current plan

        Returns:
            Next plan step
        """
        self.logger.info(f"Generating step {len(current_plan)}")

        filenames, formatted_desc = self._format_data_descriptions(data_descriptions)

        # Format current plan
        plan_text = "\n".join([str(step) for step in current_plan])

        prompt = PLANNER_NEXT_PROMPT.format(
            question=question,
            filenames=filenames,
            data_descriptions=formatted_desc,
            current_plans=plan_text,
            current_result=current_result,
        )

        response = self.llm.chat(prompt)
        return PlanStep(step_number=len(current_plan), description=response.strip())

