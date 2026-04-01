```python
import ast
from collections import defaultdict
import logging

# Get a logger for this module. Logging configuration is expected to be handled
# by the main application, not by this library module.
logger = logging.getLogger(__name__)


class CodeAstParser:
    """
    Parses code content to generate an Abstract Syntax Tree (AST) and extract
    key structural and logical information such as functions, classes, imports,
    and a basic call graph.

    Currently supports Python code using the built-in `ast` module.
    Leverages `ast.unparse` which is available from Python 3.9+.
    """

    def __init__(self):
        """
        Initializes the CodeAstParser. No specific configuration needed at this stage.
        """
        pass

    def parse(self, code_content: str, file_path: str = None) -> dict:
        """
        Parses the given code content to extract AST-based features.

        Args:
            code_content (str): The source code as a string.
            file_path (str, optional): The path to the file, used for logging/context. Defaults to None.

        Returns:
            dict: A dictionary containing extracted features:
                - 'functions': List of dictionaries for each function (name, args, returns, docstring, start_line, end_line, calls, type).
                - 'classes': List of dictionaries for each class (name, inherits, docstring, methods, start_line, end_line).
                - 'imports': List of strings representing imported modules/objects.
                - 'call_graph': Dictionary mapping caller names (functions/methods) to a list of callees.
                - 'source_code_lines': List of code lines (for context).
                - 'ast_summary': High-level summary of the AST structure.

        Raises:
            TypeError: If `code_content` is not a string.
            SyntaxError: If the `code_content` cannot be parsed into a valid AST.
            Exception: For other unexpected errors during parsing.
        """
        if not isinstance(code_content, str):
            raise TypeError("code_content must be a string.")

        # Handle empty or whitespace-only code content gracefully
        if not code_content.strip():
            return self._empty_result(file_path)

        extracted_data = {
            "functions": [],
            "classes": [],
            "imports": [],
            "call_graph": defaultdict(list),  # Use defaultdict for easier accumulation
            "source_code_lines": code_content.splitlines(),
            "ast_summary": {},
        }

        try:
            # Parse the code into an AST
            tree = ast.parse(code_content)
        except SyntaxError as e:
            logger.error(f"Syntax error parsing code from {file_path or 'unknown source'}: {e}")
            raise  # Re-raise for upstream handling, as this indicates invalid input
        except Exception as e:
            logger.error(f"Unexpected error parsing code from {file_path or 'unknown source'}: {type(e).__name__}: {e}")
            raise  # Re-raise for other unexpected errors

        # Traverse the AST and extract information
        self._extract_from_ast(tree, extracted_data)

        # Convert defaultdict to a regular dict for final output (better for serialization)
        extracted_data['call_graph'] = {k: list(set(v)) for k, v in extracted_data['call_graph'].items()} # Remove duplicate callees

        # Add a summary of the AST itself
        extracted_data['ast_summary'] = self._summarize_ast(tree)

        return extracted_data

    def _empty_result(self, file_path: str) -> dict:
        """
        Generates an empty result dictionary for cases where input code is empty or only whitespace.
        """
        logger.info(f"Provided code_content for {file_path or 'unknown source'} is empty or whitespace-only. Returning empty result.")
        return {
            "functions": [],
            "classes": [],
            "imports": [],
            "call_graph": {},
            "source_code_lines": [],
            "ast_summary": {"node_count": 0, "node_types": {}},
        }

    def _summarize_ast(self, tree: ast.AST) -> dict:
        """
        Generates a high-level summary of the AST, including node counts and types.
        """
        node_types_counts = defaultdict(int)
        for node in ast.walk(tree):
            node_types_counts[type(node).__name__] += 1
        return {
            "node_count": sum(node_types_counts.values()),
            "node_types": dict(node_types_counts)
        }

    def _extract_from_ast(self, tree: ast.AST, extracted_data: dict):
        """
        Recursively walks the AST to identify and extract functions, classes, imports,
        and populate the call graph.
        """
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._extract_function_info(node, extracted_data)
            elif isinstance(node, ast.ClassDef):
                self._extract_class_info(node, extracted_data)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    extracted_data['imports'].append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module_name = node.module if node.module else ""
                for alias in node.names:
                    # Handle 'from . import foo' or 'from .. import bar'
                    full_import_name = f"{module_name}.{alias.name}" if module_name else alias.name
                    extracted_data['imports'].append(full_import_name)

    def _extract_function_info(self, node: ast.FunctionDef | ast.AsyncFunctionDef, extracted_data: dict):
        """
        Extracts detailed information for a function or async function definition.
        Includes function name, arguments, return type, docstring, line numbers,
        and calls made within its body.
        """
        func_name = node.name

        args = []
        # Positional arguments
        for arg in node.args.posonlyargs + node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation).strip()}"
            args.append(arg_str)

        # Handle default arguments by appending to the corresponding positional args
        defaults_start_index = len(args) - len(node.args.defaults)
        for i, default_node in enumerate(node.args.defaults):
            if defaults_start_index + i < len(args): # Ensure index is valid
                args[defaults_start_index + i] += f"={ast.unparse(default_node).strip()}"

        # Varargs (*args)
        if node.args.vararg:
            vararg_str = f"*{node.args.vararg.arg}"
            if node.args.vararg.annotation:
                vararg_str += f": {ast.unparse(node.args.vararg.annotation).strip()}"
            args.append(vararg_str)

        # Keyword-only arguments
        for arg in node.args.kwonlyargs:
            kwarg_str = arg.arg
            if arg.annotation:
                kwarg_str += f": {ast.unparse(arg.annotation).strip()}"
            # Add default value if present
            kwarg_index = node.args.kwonlyargs.index(arg)
            if kwarg_index < len(node.args.kw_defaults) and node.args.kw_defaults[kwarg_index]:
                 kwarg_str += f"={ast.unparse(node.args.kw_defaults[kwarg_index]).strip()}"
            args.append(kwarg_str)

        # Kwargs (**kwargs)
        if node.args.kwarg:
            kwarg_str = f"**{node.args.kwarg.arg}"
            if node.args.kwarg.annotation:
                kwarg_str += f": {ast.unparse(node.args.kwarg.annotation).strip()}"
            args.append(kwarg_str)
        
        returns = ast.unparse(node.returns).strip() if node.returns else None

        start_line = node.lineno
        end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line  # end_lineno in 3.8+

        calls_made_by_func = []
        for body_node in node.body:
            for sub_node in ast.walk(body_node):
                if isinstance(sub_node, ast.Call):
                    called_func_name = self._get_called_func_name(sub_node.func)
                    if called_func_name and called_func_name not in calls_made_by_func:
                        calls_made_by_func.append(called_func_name)
                        extracted_data['call_graph'][func_name].append(called_func_name)

        extracted_data['functions'].append({
            "name": func_name,
            "args": args,
            "returns": returns,
            "docstring": ast.get_docstring(node),
            "start_line": start_line,
            "end_line": end_line,
            "calls": calls_made_by_func,
            "type": "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function"
        })

    def _extract_class_info(self, node: ast.ClassDef, extracted_data: dict):
        """
        Extracts detailed information for a class definition.
        Includes class name, inheritance, docstring, line numbers, and its methods.
        """
        class_name = node.name
        inherits = [ast.unparse(base).strip() for base in node.bases]
        methods = []

        start_line = node.lineno
        end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line

        for body_node in node.body:
            if isinstance(body_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_name = body_node.name

                method_args = []
                for arg in body_node.args.posonlyargs + body_node.args.args:
                    arg_str = arg.arg
                    if arg.annotation:
                        arg_str += f": {ast.unparse(arg.annotation).strip()}"
                    method_args.append(arg_str)

                # Handle default arguments for methods
                defaults_start_index = len(method_args) - len(body_node.args.defaults)
                for i, default_node in enumerate(body_node.args.defaults):
                    if defaults_start_index + i < len(method_args):
                        method_args[defaults_start_index + i] += f"={ast.unparse(default_node).strip()}"

                # Varargs (*args)
                if body_node.args.vararg:
                    vararg_str = f"*{body_node.args.vararg.arg}"
                    if body_node.args.vararg.annotation:
                        vararg_str += f": {ast.unparse(body_node.args.vararg.annotation).strip()}"
                    method_args.append(vararg_str)

                # Keyword-only arguments
                for arg in body_node.args.kwonlyargs:
                    kwarg_str = arg.arg
                    if arg.annotation:
                        kwarg_str += f": {ast.unparse(arg.annotation).strip()}"
                    kwarg_index = body_node.args.kwonlyargs.index(arg)
                    if kwarg_index < len(body_node.args.kw_defaults) and body_node.args.kw_defaults[kwarg_index]:
                        kwarg_str += f"={ast.unparse(body_node.args.kw_defaults[kwarg_index]).strip()}"
                    method_args.append(kwarg_str)

                # Kwargs (**kwargs)
                if body_node.args.kwarg:
                    kwarg_str = f"**{body_node.args.kwarg.arg}"
                    if body_node.args.kwarg.annotation:
                        kwarg_str += f": {ast.unparse(body_node.args.kwarg.annotation).strip()}"
                    method_args.append(kwarg_str)


                method_returns = ast.unparse(body_node.returns).strip() if body_node.returns else None

                calls_made_by_method = []
                for sub_node in ast.walk(body_node):
                    if isinstance(sub_node, ast.Call):
                        called_func_name = self._get_called_func_name(sub_node.func)
                        if called_func_name and called_func_name not in calls_made_by_method:
                            calls_made_by_method.append(called_func_name)
                            # Prefix method calls with class name for clarity in call graph
                            extracted_data['call_graph'][f"{class_name}.{method_name}"].append(called_func_name)

                methods.append({
                    "name": method_name,
                    "args": method_args,
                    "returns": method_returns,
                    "docstring": ast.get_docstring(body_node),
                    "start_line": body_node.lineno,
                    "end_line": body_node.end_lineno if hasattr(body_node, 'end_lineno') else body_node.lineno,
                    "calls": calls_made_by_method,
                    "type": "async_method" if isinstance(body_node, ast.AsyncFunctionDef) else "method"
                })

        extracted_data['classes'].append({
            "name": class_name,
            "inherits": inherits,
            "docstring": ast.get_docstring(node),
            "methods": methods,
            "start_line": start_line,
            "end_line": end_line,
        })

    def _get_called_func_name(self, call_node_func: ast.expr) -> str | None:
        """
        Helper method to extract the string name of a function or method being called
        from an AST `func` expression (e.g., `ast.Name`, `ast.Attribute`).
        Uses `ast.unparse` for a robust string representation.
        """
        try:
            # ast.unparse works well for most expressions that represent a callable
            return ast.unparse(call_node_func).strip()
        except Exception as e:
            # Fallback for some common cases if unparse is too generic or fails
            logger.debug(f"Could not unparse callable expression type {type(call_node_func).__name__}: {e}")
            if isinstance(call_node_func, ast.Name):
                return call_node_func.id
            elif isinstance(call_node_func, ast.Attribute):
                # Try to get a more structured name for attributes like obj.method
                base = self._get_called_func_name(call_node_func.value)
                if base:
                    return f"{base}.{call_node_func.attr}"
                return call_node_func.attr  # Fallback for attributes if base is complex (e.g., 'self')
            # Give up on very complex or dynamic call expressions
            return None
```