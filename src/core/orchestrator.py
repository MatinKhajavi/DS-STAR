import logging
from pathlib import Path
from typing import List, Optional

from src.config import DSStarConfig, LLMConfig
from src.core.models import (
    DataDescription,
    DataFile,
    ExecutionResult,
    IterationState,
    PlanStep,
    RouterDecision,
    VerificationStatus,
)
from src.core.executor import CodeExecutor
from src.llm.base import BaseLLMClient
from src.llm.gemini_client import GeminiClient
from src.llm.openai_client import OpenAIClient
from src.llm.anthropic_client import AnthropicClient
from src.agents import (
    AnalyzerAgent,
    PlannerAgent,
    CoderAgent,
    VerifierAgent,
    RouterAgent,
    DebuggerAgent,
    FinalizerAgent,
)
from src.utils.filesystem import discover_data_files, ensure_directory
from src.utils.logging_utils import setup_logger, log_iteration, log_final_solution, save_data_descriptions


class DSStar:
    """
    DS-STAR: Data Science Agent via Iterative Planning and Verification.
    
    Implements Algorithm 1 from the paper.
    """

    def __init__(self, config: Optional[DSStarConfig] = None):
        """
        Initialize DS-STAR.

        Args:
            config: Configuration for DS-STAR
        """
        self.config = config or DSStarConfig()

        ensure_directory(self.config.log_dir)
        self.logger = setup_logger("ds_star", self.config.log_dir, self.config.debug_mode)

        self.llm = self._create_llm_client(self.config.llm)

        self.executor = CodeExecutor()

        self.analyzer = AnalyzerAgent(self.llm, self.executor)
        self.planner = PlannerAgent(self.llm)
        self.coder = CoderAgent(self.llm)
        self.verifier = VerifierAgent(self.llm)
        self.router = RouterAgent(self.llm)
        self.debugger = DebuggerAgent(self.llm)
        self.finalizer = FinalizerAgent(self.llm)

        # Cache for data descriptions (run once, reuse for all queries)
        self._data_descriptions: Optional[List[DataDescription]] = None
        self._analyzed_data_dir: Optional[str] = None

        self.logger.info(f"DS-STAR initialized with {self.config.llm.provider}/{self.config.llm.model}")

    def _create_llm_client(self, llm_config: LLMConfig) -> BaseLLMClient:
        """Create LLM client based on configuration."""
        if llm_config.provider == "gemini":
            return GeminiClient(
                model=llm_config.model,
                api_key=llm_config.api_key,
            )
        elif llm_config.provider == "openai":
            raise NotImplementedError("OpenAI is not supported yet")
        elif llm_config.provider == "anthropic":
            raise NotImplementedError("Anthropic is not supported yet")
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_config.provider}")

    def analyze_files(self, data_files: List[DataFile]) -> List[DataDescription]:
        """
        Analyze data files to generate descriptions (Section 3.1).

        This corresponds to lines 3-6 in Algorithm 1.

        Args:
            data_files: List of data files to analyze

        Returns:
            List of data descriptions
        """
        self.logger.info(f"Analyzing {len(data_files)} data files")
        descriptions = []

        for i, data_file in enumerate(data_files, 1):
            self.logger.info(f"[{i}/{len(data_files)}] Analyzing {data_file.path}")
            desc = self.analyzer.analyze_file(data_file)

            if desc.error:
                self.logger.warning(f"Analysis failed for {data_file.path}: {desc.error[:200]}")
                self.logger.info(f"Attempting to debug {data_file.filename}...")
                fixed_script = self.debugger.fix_analyzer_script(desc.script, desc.error)

                result = self.executor.execute(fixed_script)
                if result.success:
                    desc.script = fixed_script
                    desc.description = result.stdout
                    desc.error = None
                    self.logger.info(f"✓ Successfully fixed analyzer for {data_file.filename}")
                else:
                    self.logger.error(f"✗ Could not fix analyzer for {data_file.filename}: {result.error[:200] if result.error else 'Unknown error'}")

            else:
                self.logger.info(f"✓ Successfully analyzed {data_file.filename}")

            descriptions.append(desc)

        successful = sum(1 for d in descriptions if not d.error)
        self.logger.info(f"Analysis complete: {successful}/{len(descriptions)} files successfully analyzed")
        
        save_data_descriptions(descriptions, self.config.log_dir, self.logger)
        
        return descriptions

    def prepare_data(self, data_dir: Optional[str] = None) -> None:
        """
        Pre-analyze data files and cache descriptions.
        
        This is useful to analyze files once before running multiple queries.
        Call this method once after initialization to avoid re-analyzing files
        on every query.

        Args:
            data_dir: Directory containing data files (if None, uses config.data_dir)
        """
        data_dir = data_dir or self.config.data_dir
        
        self.logger.info(f"Pre-analyzing data files in {data_dir}")
        
        data_files = discover_data_files(data_dir)
        self.logger.info(f"Discovered {len(data_files)} data files")

        if not data_files:
            raise ValueError(f"No data files found in {data_dir}")

        self._data_descriptions = self.analyze_files(data_files)
        self._analyzed_data_dir = data_dir
        
        self.logger.info(f"Data preparation complete: cached {len(self._data_descriptions)} file descriptions")

    def _execute_with_debug(
        self, code: str, data_descriptions: List[DataDescription]
    ) -> ExecutionResult:
        """
        Execute code with automatic debugging on failure.

        Args:
            code: Code to execute
            data_descriptions: Available data descriptions

        Returns:
            Execution result
        """
        result = self.executor.execute(code)

        if not result.success:
            self.logger.warning("Execution failed, attempting to debug")
            fixed_code = self.debugger.fix_solution_script(
                code, result.error or result.stderr, data_descriptions
            )
            result = self.executor.execute(fixed_code)

        return result

    def run(
        self,
        question: str,
        data_dir: Optional[str] = None,
        guidelines: Optional[str] = None,
    ) -> tuple[str, str]:
        """
        Run DS-STAR to answer a question using data files.

        This implements Algorithm 1 from the paper.

        Args:
            question: The question to answer
            data_dir: Directory containing data files (if None, uses config.data_dir)
            guidelines: Optional formatting guidelines

        Returns:
            Tuple of (final_code, final_result)
        """
        self.logger.info(f"Starting DS-STAR for question: {question}")

        data_dir = data_dir or self.config.data_dir

        if self._data_descriptions is not None and self._analyzed_data_dir == data_dir:
            self.logger.info(f"Using cached descriptions for {len(self._data_descriptions)} data files")
            data_descriptions = self._data_descriptions
        else:
            data_files = discover_data_files(data_dir)
            self.logger.info(f"Discovered {len(data_files)} data files")

            if not data_files:
                raise ValueError(f"No data files found in {data_dir}")

            data_descriptions = self.analyze_files(data_files)
            
            self._data_descriptions = data_descriptions
            self._analyzed_data_dir = data_dir
            self.logger.info(f"Cached descriptions for {len(data_descriptions)} files")

        plan_step_0 = self.planner.generate_initial_plan(question, data_descriptions)
        plan = [plan_step_0]

        code = self.coder.implement_initial_plan(plan_step_0, data_descriptions)
        result = self._execute_with_debug(code, data_descriptions)

        self.logger.info(f"Initial plan executed: {result.success}")

        for round_num in range(self.config.max_rounds):
            iteration = IterationState(
                round_number=round_num,
                plan=plan.copy(),
                code=code,
                execution_result=result,
            )

            verification = self.verifier.verify_plan(
                question, plan, code, result.output
            )
            iteration.verification = verification

            log_iteration(self.logger, iteration, self.config.log_dir, question)

            if verification.status == VerificationStatus.SUFFICIENT:
                self.logger.info(f"Plan verified as sufficient at round {round_num}")
                break


            router_result = self.router.route(
                question, plan, result.output, data_descriptions
            )
            iteration.router_result = router_result

            if router_result.decision == RouterDecision.REMOVE_STEP:
                remove_idx = router_result.step_to_remove
                plan = plan[: remove_idx]  # Keep steps 0 to remove_idx-1
                self.logger.info(f"Truncated plan to {len(plan)} steps")

            next_step = self.planner.generate_next_step(
                question, data_descriptions, plan, result.output
            )

            plan.append(next_step)

            code = self.coder.implement_plan(plan, data_descriptions, code)

            result = self._execute_with_debug(code, data_descriptions)

            self.logger.info(
                f"Round {round_num + 1}: plan has {len(plan)} steps, "
                f"execution {'succeeded' if result.success else 'failed'}"
            )

        final_code = self.finalizer.finalize_solution(
            question, data_descriptions, code, result.output, guidelines
        )

        final_result = self.executor.execute(final_code)

        log_final_solution(
            self.logger,
            self.config.log_dir,
            question,
            final_code,
            final_result.output,
            len(plan),
        )

        self.logger.info("DS-STAR completed successfully")

        return final_code, final_result.output

