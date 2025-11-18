import logging
from typing import List

from src.core.models import PlanStep, VerificationStatus, VerificationResult
from src.core.prompts import VERIFIER_PROMPT
from src.llm.base import BaseLLMClient


class VerifierAgent:
    """
    Agent for verifying if a plan is sufficient to answer the question.
    
    Implements Section 3.2: Plan verification.
    """

    def __init__(self, llm_client: BaseLLMClient):
        self.llm = llm_client
        self.logger = logging.getLogger(__name__)

    def verify_plan(
        self,
        question: str,
        plan: List[PlanStep],
        code: str,
        execution_result: str,
    ) -> VerificationResult:
        """
        Verify if the current plan is sufficient to answer the question.

        This is the key verification step (v = A_verifier(...) in Algorithm 1).

        Args:
            question: User's question
            plan: Current plan steps
            code: Implementation of the plan
            execution_result: Result from executing the code

        Returns:
            VerificationResult indicating if plan is sufficient
        """
        self.logger.info(f"Verifying plan with {len(plan)} steps")

        plan_text = "\n".join([str(step) for step in plan])

        prompt = VERIFIER_PROMPT.format(
            plan=plan_text,
            code=code,
            result=execution_result,
            question=question,
        )

        response = self.llm.chat(prompt).strip().lower()

        if "yes" in response:
            status = VerificationStatus.SUFFICIENT
        else:
            status = VerificationStatus.INSUFFICIENT

        self.logger.info(f"Verification result: {status.value}")

        return VerificationResult(status=status, reasoning=response)

