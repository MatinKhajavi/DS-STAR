import logging
import re
from typing import List

from src.core.models import (
    DataDescription,
    PlanStep,
    RouterDecision,
    RouterResult,
)
from src.core.prompts import ROUTER_PROMPT
from src.llm.base import BaseLLMClient


class RouterAgent:
    """
    Agent for deciding whether to add or remove plan steps.
    
    Implements Section 3.2: Plan refinement.
    """

    def __init__(self, llm_client: BaseLLMClient):
        self.llm = llm_client
        self.logger = logging.getLogger(__name__)

    def route(
        self,
        question: str,
        plan: List[PlanStep],
        execution_result: str,
        data_descriptions: List[DataDescription],
    ) -> RouterResult:
        """
        Decide how to refine the current plan.

        This implements the routing logic (w = A_router(...) in Algorithm 1).
        Returns either "Add Step" or a step index to remove.

        Args:
            question: User's question
            plan: Current plan steps
            execution_result: Result from executing current plan
            data_descriptions: Available data descriptions

        Returns:
            RouterResult with decision and optional step to remove
        """
        self.logger.info(f"Routing decision for plan with {len(plan)} steps")

        # Format data descriptions
        filenames = ", ".join([desc.file.filename for desc in data_descriptions])
        formatted_desc = "\n\n".join(
            [
                f"{desc.file.filename}:\n{desc.description}"
                for desc in data_descriptions
                if desc.description
            ]
        )

        # Format current plan
        plan_text = "\n".join([str(step) for step in plan])

        prompt = ROUTER_PROMPT.format(
            question=question,
            filenames=filenames,
            data_descriptions=formatted_desc,
            current_plans=plan_text,
            current_result=execution_result,
            num_steps=len(plan),
        )

        response = self.llm.chat(prompt).strip()

        # Parse response
        # Look for "Add Step" or "Step N"
        if "add step" in response.lower():
            decision = RouterDecision.ADD_STEP
            step_to_remove = None
            self.logger.info("Router decision: Add next step")
        else:
            # Try to extract step number
            match = re.search(r"step\s+(\d+)", response, re.IGNORECASE)
            if match:
                step_num = int(match.group(1))
                # Validate step number
                if 1 <= step_num <= len(plan):
                    decision = RouterDecision.REMOVE_STEP
                    step_to_remove = step_num
                    self.logger.info(f"Router decision: Remove step {step_num}")
                else:
                    # Invalid step number, default to add
                    self.logger.warning(
                        f"Invalid step number {step_num}, defaulting to Add Step"
                    )
                    decision = RouterDecision.ADD_STEP
                    step_to_remove = None
            else:
                # Can't parse, default to add
                self.logger.warning(f"Could not parse router response: {response}")
                decision = RouterDecision.ADD_STEP
                step_to_remove = None

        return RouterResult(
            decision=decision,
            step_to_remove=step_to_remove,
            reasoning=response,
        )

