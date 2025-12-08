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
from src.utils.logging_utils import (
    setup_logger,
    log_iteration,
    log_final_solution,
    save_data_descriptions,
    load_data_descriptions,
    log_prompt,
)


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

        # Create executor with 5-minute timeout to prevent infinite loops
        self.executor = CodeExecutor(timeout=300)

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
                current_code = desc.script
                current_error = desc.error
                max_retries = self.config.execution.max_retries
                
                for attempt in range(max_retries):
                    self.logger.info(f"Attempting to debug {data_file.filename}... ({attempt + 1}/{max_retries})")
                    try:
                        fixed_script = self.debugger.fix_analyzer_script(current_code, current_error)
                    except Exception as e:
                        self.logger.error(f"Debugger failed on analyzer script: {e}")
                        break

                    result = self.executor.execute(fixed_script)
                    if result.success:
                        desc.script = fixed_script
                        desc.description = result.stdout
                        desc.error = None
                        self.logger.info(f"✓ Successfully fixed analyzer for {data_file.filename}")
                        break

                    current_code = fixed_script
                    current_error = result.stderr or result.error
                    self.logger.warning(f"Analyzer fix attempt failed: {current_error[:200] if current_error else 'Unknown error'}")

                if desc.error:
                    self.logger.error(f"✗ Could not fix analyzer for {data_file.filename} after {max_retries} attempts")

            else:
                self.logger.info(f"✓ Successfully analyzed {data_file.filename}")

            descriptions.append(desc)

        successful = sum(1 for d in descriptions if not d.error)
        self.logger.info(f"Analysis complete: {successful}/{len(descriptions)} files successfully analyzed")
        
        save_data_descriptions(descriptions, self.config.log_dir, self.logger)
        
        return descriptions

    def prepare_data(self, data_dir: Optional[str] = None, force_reanalyze: bool = False) -> None:
        """
        Pre-analyze data files and cache descriptions.
        
        This is useful to analyze files once before running multiple queries.
        Call this method once after initialization to avoid re-analyzing files
        on every query. Will attempt to load from cache if available.

        Args:
            data_dir: Directory containing data files (if None, uses config.data_dir)
            force_reanalyze: If True, ignore cache and re-analyze all files
        """
        data_dir = data_dir or self.config.data_dir
        
        if not force_reanalyze:
            self.logger.info("Checking for cached data descriptions...")
            cached_descriptions = load_data_descriptions(self.config.log_dir, self.logger)
            
            if cached_descriptions:
                data_files = discover_data_files(data_dir)
                cached_paths = {desc.file.path for desc in cached_descriptions}
                current_paths = {df.path for df in data_files}
                
                if cached_paths == current_paths:
                    self.logger.info(f"✓ Using cached descriptions for {len(cached_descriptions)} files")
                    self._data_descriptions = cached_descriptions
                    self._analyzed_data_dir = data_dir
                    return
                else:
                    self.logger.info("Cache mismatch: files have changed, re-analyzing...")
        
        self.logger.info(f"Analyzing data files in {data_dir}")
        
        data_files = discover_data_files(data_dir)
        self.logger.info(f"Discovered {len(data_files)} data files")

        if not data_files:
            raise ValueError(f"No data files found in {data_dir}")

        self._data_descriptions = self.analyze_files(data_files)
        self._analyzed_data_dir = data_dir
        
        self.logger.info(f"Data preparation complete: cached {len(self._data_descriptions)} file descriptions")

    def _execute_with_debug(
        self, code: str, data_descriptions: List[DataDescription]
    ) -> tuple[str, ExecutionResult]:
        """
        Execute code with automatic debugging on failure.

        Returns the possibly-updated code along with the execution result.
        """
        max_retries = self.config.execution.max_retries
        attempt = 0
        code_to_run = code
        result = self.executor.execute(code_to_run)

        while not result.success and attempt < max_retries:
            attempt += 1
            self.logger.warning(f"Execution failed, attempting to debug (attempt {attempt}/{max_retries})")

            try:
                bug_report = result.stderr or result.error or ""
                code_to_run = self.debugger.fix_solution_script(
                    code_to_run, bug_report, data_descriptions
                )
            except Exception as e:
                self.logger.error(f"Debugger failed to produce a fix: {e}")
                break

            result = self.executor.execute(code_to_run)

        return code_to_run, result

    def _build_code_for_plan(
        self, plan: List[PlanStep], data_descriptions: List[DataDescription]
    ) -> tuple[str, ExecutionResult]:
        """
        Rebuild code and execution result to match the current plan (used after backtracking).
        """
        if not plan:
            return "", ExecutionResult(stdout="", stderr="", returncode=0, execution_time=0.0, error=None)

        code = self.coder.implement_initial_plan(plan[0], data_descriptions)
        code, result = self._execute_with_debug(code, data_descriptions)

        for idx in range(1, len(plan)):
            code = self.coder.implement_plan(plan[: idx + 1], data_descriptions, code)
            code, result = self._execute_with_debug(code, data_descriptions)

        return code, result

    def run(
        self,
        question: str,
        data_dir: Optional[str] = None,
        guidelines: Optional[str] = None,
        query_id: Optional[str] = None,
    ) -> tuple[str, str]:
        """
        Run DS-STAR to answer a question using data files.

        This implements Algorithm 1 from the paper.

        Args:
            question: The question to answer
            data_dir: Directory containing data files (if None, uses config.data_dir)
            guidelines: Optional formatting guidelines
            query_id: Optional identifier for this query (for logging organization)

        Returns:
            Tuple of (final_code, final_result)
        """
        if query_id is None:
            import hashlib
            query_id = f"query_{hashlib.md5(question.encode()).hexdigest()[:8]}"
        
        # Set logging context for LLM prompts
        self.llm.set_logging_context(self.config.log_dir, query_id)
        
        self.logger.info(f"Starting DS-STAR for question: {question}")
        self.logger.info(f"Query ID: {query_id}")

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

        self.logger.info("Generating initial plan...")
        plan_step_0 = self.planner.generate_initial_plan(question, data_descriptions)
        self.logger.info(f"Initial plan: {plan_step_0.description[:100]}...")
        plan = [plan_step_0]

        self.logger.info("Implementing initial plan...")
        code = self.coder.implement_initial_plan(plan_step_0, data_descriptions)
        self.logger.info(f"Generated code ({len(code)} chars), executing...")
        
        try:
            code, result = self._execute_with_debug(code, data_descriptions)
            self.logger.info(f"Initial plan executed: {result.success}")
            if not result.success:
                self.logger.warning(f"Execution failed: {result.error}")
        except KeyboardInterrupt:
            self.logger.error("KeyboardInterrupt: User interrupted execution")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during execution: {type(e).__name__}: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise

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

            log_iteration(self.logger, iteration, self.config.log_dir, question, query_id)

            if verification.status == VerificationStatus.SUFFICIENT:
                self.logger.info(f"Plan verified as sufficient at round {round_num}")
                break


            router_result = self.router.route(
                question, plan, result.output, data_descriptions
            )
            iteration.router_result = router_result

            if router_result.decision == RouterDecision.REMOVE_STEP:
                remove_idx = router_result.step_to_remove  # Step number to remove
                if remove_idx:
                    plan = [step for step in plan if step.step_number < remove_idx]
                    self.logger.info(f"Truncated plan to {len(plan)} steps (removed step {remove_idx} and beyond)")
                    code, result = self._build_code_for_plan(plan, data_descriptions)

            next_step = self.planner.generate_next_step(
                question, data_descriptions, plan, result.output
            )

            plan.append(next_step)

            code = self.coder.implement_plan(plan, data_descriptions, code if plan else None)

            code, result = self._execute_with_debug(code, data_descriptions)

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
            query_id,
        )

        self.logger.info("DS-STAR completed successfully")

        return final_code, final_result.output

