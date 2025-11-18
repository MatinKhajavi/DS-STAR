"""code execution engine for DS-STAR.

This file contains code adapted from llama_index, which is licensed under MIT.
Original source: https://github.com/run-llama/llama_index/blob/main/llama-index-experimental/llama_index/experimental/exec_utils.py
"""

import ast
import copy
import io
import sys
import time
from contextlib import redirect_stdout, redirect_stderr
from types import CodeType, ModuleType
from typing import Any, Dict, Mapping, Sequence, Union, Optional

from src.core.models import ExecutionResult

ALLOWED_IMPORTS = {
    "math",
    "time",
    "datetime",
    "pandas",
    "scipy",
    "numpy",
    "matplotlib",
    "plotly",
    "seaborn",
    "re",
    "json",
    "csv",
    "os",
    "pathlib",
    "collections",
    "itertools",
    "functools",
    "operator",
    "statistics",
    "random",
    "string",
    "warnings",
}


def _restricted_import(
    name: str,
    globals: Union[Mapping[str, object], None] = None,
    locals: Union[Mapping[str, object], None] = None,
    fromlist: Sequence[str] = (),
    level: int = 0,
) -> ModuleType:
    if name in ALLOWED_IMPORTS:
        return __import__(name, globals, locals, fromlist, level)
    raise ImportError(f"Import of module '{name}' is not allowed")


ALLOWED_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "ascii": ascii,
    "bin": bin,
    "bool": bool,
    "bytearray": bytearray,
    "bytes": bytes,
    "chr": chr,
    "complex": complex,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "format": format,
    "frozenset": frozenset,
    "hash": hash,
    "hex": hex,
    "int": int,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "oct": oct,
    "ord": ord,
    "pow": pow,
    "print": print,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "type": type,
    "zip": zip,
    "open": open,
    # Constants
    "True": True,
    "False": False,
    "None": None,
    "__import__": _restricted_import,
}


def _get_restricted_globals(__globals: Union[dict, None]) -> Dict[str, Any]:
    restricted_globals = copy.deepcopy(ALLOWED_BUILTINS)
    if __globals:
        restricted_globals.update(__globals)
    return restricted_globals


class DunderVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.has_access_to_private_entity = False
        self.has_access_to_disallowed_builtin = False
        builtins_obj = globals()["__builtins__"]
        if isinstance(builtins_obj, dict):
            self._builtins = builtins_obj.keys()
        else:
            import builtins

            self._builtins = dir(builtins)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("_"):
            self.has_access_to_private_entity = True
        if node.id not in ALLOWED_BUILTINS and node.id in self._builtins:
            self.has_access_to_disallowed_builtin = True
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("_"):
            self.has_access_to_private_entity = True
        if node.attr not in ALLOWED_BUILTINS and node.attr in self._builtins:
            self.has_access_to_disallowed_builtin = True
        self.generic_visit(node)


def _contains_protected_access(code: str) -> bool:
    """
    Detect direct imports or references to private/dunder objects
    that should not be accessible.
    """
    tree = ast.parse(code)

    # Check for disallowed imports
    has_disallowed_imports = False
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                if name.name not in ALLOWED_IMPORTS:
                    has_disallowed_imports = True
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module not in ALLOWED_IMPORTS:
                has_disallowed_imports = True

    dunder_visitor = DunderVisitor()
    dunder_visitor.visit(tree)

    return (
        dunder_visitor.has_access_to_private_entity
        or dunder_visitor.has_access_to_disallowed_builtin
        or has_disallowed_imports
    )


def _verify_source_safety(__source: Union[str, bytes, CodeType]) -> None:
    if isinstance(__source, CodeType):
        raise RuntimeError("Direct execution of CodeType is forbidden!")
    if isinstance(__source, bytes):
        __source = __source.decode()
    if _contains_protected_access(__source):
        raise RuntimeError(
            "Execution of code with private/dunder references or disallowed imports is forbidden!"
        )


def safe_exec(
    __source: Union[str, bytes, CodeType],
    __globals: Union[Dict[str, Any], None] = None,
    __locals: Union[Mapping[str, object], None] = None,
) -> None:
    """
    Execute code in a restricted environment.
    """
    _verify_source_safety(__source)
    exec(__source, _get_restricted_globals(__globals), __locals)


class CodeExecutor:
    """Wrapper for safe code execution and output capture."""

    def __init__(self):
        pass

    def execute(
        self, code: str, globals_dict: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """
        Execute Python code safely and return results.

        Args:
            code: Python code to execute
            globals_dict: Optional global variables to inject

        Returns:
            ExecutionResult with stdout, stderr, returncode, and timing
        """
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        start_time = time.time()
        returncode = 0
        error = None

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                safe_exec(code, globals_dict, None)
        except Exception as e:
            returncode = 1
            error = f"{type(e).__name__}: {str(e)}"
            stderr_capture.write(error)

        execution_time = time.time() - start_time

        return ExecutionResult(
            stdout=stdout_capture.getvalue(),
            stderr=stderr_capture.getvalue(),
            returncode=returncode,
            execution_time=execution_time,
            error=error,
        )

