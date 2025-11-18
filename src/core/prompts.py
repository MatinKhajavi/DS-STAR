# ============================================================================
# Analyzer Agent (G.1)
# ============================================================================

ANALYZER_PROMPT = """You are an expert data analyst.

Generate a Python code that loads and describes the content of {filename}.

# Requirement
- The file can be both unstructured or structured data.
- If there are too many structured data, print out just few examples.
- Print out essential information. For example, print out all the column names.
- The Python code should print out the content of {filename}.
- The code should be a single-file Python program that is self-contained and can be executed as-is.
- Your response should only contain a single code block.
- Important: You should not include dummy contents since we will debug if error occurs.
- Do not use try: and except: to prevent error. I will debug it later."""

# ============================================================================
# Planner Agent (G.2)
# ============================================================================

PLANNER_INITIAL_PROMPT = """You are an expert data analyst.

In order to answer factoid questions based on the given data, you have to first plan effectively.

# Question
{question}

# Given data: {filenames}
{data_descriptions}

# Your task
- Suggest your very first step to answer the question above.
- Your first step does not need to be sufficient to answer the question.
- Just propose a very simple initial step, which can act as a good starting point to answer the question.
- Your response should only contain an initial step."""

PLANNER_NEXT_PROMPT = """You are an expert data analyst.

In order to answer factoid questions based on the given data, you have to first plan effectively.

Your task is to suggest next plan to do to answer the question.

# Question
{question}

# Given data: {filenames}
{data_descriptions}

# Current plans
{current_plans}

# Obtained results from the current plans:
{current_result}

# Your task
- Suggest your next step to answer the question above.
- Your next step does not need to be sufficient to answer the question, but if it requires only final simple last step you may suggest it.
- Just propose a very simple next step, which can act as a good intermediate point to answer the question.
- Of course your response can be a plan which could directly answer the question.
- Your response should only contain a next step without any explanation."""

# ============================================================================
# Coder Agent (G.3)
# ============================================================================

CODER_INITIAL_PROMPT = """# Given data: {filenames}
{data_descriptions}

# Plan
{plan}

# Your task
- Implement the plan with the given data.
- Your response should be a single markdown Python code (wrapped in ```).
- There should be no additional headings or text in your response."""

CODER_NEXT_PROMPT = """You are an expert data analyst.

Your task is to implement the next plan with the given data.

# Given data: {filenames}
{data_descriptions}

# Base code
```python
{base_code}
```

# Previous plans
{previous_plans}

# Current plan to implement
{current_plan}

# Your task
- Implement the current plan with the given data.
- The implementation should be done based on the base code.
- The base code is an implementation of the previous plans.
- Your response should be a single markdown Python code (wrapped in ```).
- There should be no additional headings or text in your response."""

# ============================================================================
# Verifier Agent (G.4)
# ============================================================================

VERIFIER_PROMPT = """You are an expert data analyst.

Your task is to check whether the current plan and its code implementation is enough to answer the question.

# Plan
{plan}

# Code
```python
{code}
```

# Execution result of code
{result}

# Question
{question}

# Your task
- Verify whether the current plan and its code implementation is enough to answer the question.
- Your response should be one of 'Yes' or 'No'.
- If it is enough to answer the question, please answer 'Yes'.
- Otherwise, please answer 'No'."""

# ============================================================================
# Router Agent (G.5)
# ============================================================================

ROUTER_PROMPT = """You are an expert data analyst.

Since current plan is insufficient to answer the question, your task is to decide how to refine the plan to answer the question.

# Question
{question}

# Given data: {filenames}
{data_descriptions}

# Current plans
{current_plans}

# Obtained results from the current plans:
{current_result}

# Your task
- If you think one of the steps of current plans is wrong, answer among the following options: Step 1, Step 2, ..., Step {num_steps}.
- If you think we should perform new NEXT step, answer as 'Add Step'.
- Your response should only be Step 1 - Step {num_steps} or Add Step."""

# ============================================================================
# Finalizer Agent (G.6)
# ============================================================================

FINALIZER_PROMPT = """You are an expert data analyst.
You will answer factoid question by loading and referencing the files/documents listed below.
You also have a reference code.
Your task is to make solution code to print out the answer of the question following the given guideline.

# Given data: {filenames}
{data_descriptions}

# Reference code
```python
{code}
```

# Execution result of reference code
{result}

# Question
{question}

# Guidelines
{guidelines}

# Your task
- Modify the solution code to print out answer to follow the given guidelines.
- If the answer can be obtained from the execution result of the reference code, just generate a Python code that prints out the desired answer.
- The code should be a single-file Python program that is self-contained and can be executed as-is.
- Your response should only contain a single code block.
- Do not include dummy contents since we will debug if error occurs.
- Do not use try: and except: to prevent error. I will debug it later.
- All files/documents are in `data/` directory."""

# ============================================================================
# Debugger Agent (G.7)
# ============================================================================

DEBUGGER_SUMMARIZE_PROMPT = """# Error report
{bug}

# Your task
- Remove all unnecessary parts of the above error report.
- We are now running {filename}.py. Do not remove where the error occurred."""

DEBUGGER_FIX_ANALYZER_PROMPT = """# Code with an error:
```python
{code}
```

# Error:
{bug}

# Your task
- Please revise the code to fix the error.
- Provide the improved, self-contained Python script again.
- There should be no additional headings or text in your response.
- Do not include dummy contents since we will debug if error occurs.
- All files/documents are in `data/` directory."""

DEBUGGER_FIX_SOLUTION_PROMPT = """# Given data: {filenames}
{data_descriptions}

# Code with an error:
```python
{code}
```

# Error:
{bug}

# Your task
- Please revise the code to fix the error.
- Provide the improved, self-contained Python script again.
- Note that you only have {filenames} available.
- There should be no additional headings or text in your response.
- Do not include dummy contents since we will debug if error occurs.
- All files/documents are in `data/` directory."""

