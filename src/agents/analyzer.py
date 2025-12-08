import logging
from typing import List

from src.core.models import DataFile, DataDescription, ExecutionResult
from src.core.prompts import ANALYZER_PROMPT
from src.core.executor import CodeExecutor
from src.llm.base import BaseLLMClient


class AnalyzerAgent:
    """
    Agent for analyzing data files and generating descriptions.
    
    Implements Section 3.1 of the paper: Analyzing data files.
    """

    def __init__(self, llm_client: BaseLLMClient, executor: CodeExecutor):
        self.llm = llm_client
        self.executor = executor
        self.logger = logging.getLogger(__name__)

    def generate_script(self, data_file: DataFile) -> str:
        """
        Generate a Python script to analyze a data file.

        Args:
            data_file: The data file to analyze

        Returns:
            Python script as string
        """
        prompt = ANALYZER_PROMPT.format(filename=data_file.path)
        response = self.llm.chat(prompt)
        return self.llm.extract_code_block(response)

    def analyze_file(self, data_file: DataFile, max_retries: int = 3) -> DataDescription:
        """
        Analyze a data file and generate its description.

        This method generates a script, executes it, and uses the output
        as the file description. If execution fails, it retries with regeneration.

        Args:
            data_file: The data file to analyze
            max_retries: Maximum number of retry attempts (default: 3)

        Returns:
            DataDescription with the analysis results
        """
        self.logger.info(f"Analyzing file: {data_file.filename}")

        script = self.generate_script(data_file)
        
        for attempt in range(max_retries):
            result = self.executor.execute(script)

            if result.success:
                if attempt > 0:
                    self.logger.info(
                        f"✓ Analysis succeeded on attempt {attempt + 1}/{max_retries}"
                    )
                return DataDescription(
                    file=data_file,
                    script=script,
                    description=result.stdout,
                )

            error_msg = result.stderr or result.error or "Unknown error"
            if attempt < max_retries - 1:
                self.logger.warning(
                    f"Script execution failed for {data_file.filename} "
                    f"(attempt {attempt + 1}/{max_retries}), retrying..."
                )
                script = self.generate_script(data_file)
            else:
                self.logger.warning(
                    f"Script execution failed for {data_file.filename} after "
                    f"{max_retries} attempts: {error_msg[:200]}"
                )

        return DataDescription(
            file=data_file,
            script=script,
            description="",
            error=result.stderr or result.error or "",
        )

    def analyze_files(self, data_files: List[DataFile]) -> List[DataDescription]:
        """
        Analyze multiple data files.

        Args:
            data_files: List of data files to analyze

        Returns:
            List of data descriptions
        """
        descriptions = []
        for data_file in data_files:
            desc = self.analyze_file(data_file)
            descriptions.append(desc)

        return descriptions

