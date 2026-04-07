```python
import json
import re
from typing import List, Dict, Any, Optional

class ChangelogProcessor:
    """
    Parses various types of external structured change logs (e.g., code diffs,
    task updates, documentation changes) to update the agent's understanding.
    """

    def __init__(self):
        """
        Initializes the ChangelogProcessor.
        Could be extended with configuration for specific parsing rules or regex patterns.
        """
        pass

    def process_changelog(self, changelog_content: str, changelog_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Processes a given changelog content string and extracts structured changes.

        Args:
            changelog_content: The raw string content of the changelog.
            changelog_type: Optional hint for the type of changelog (e.g., 'git_diff', 'task_list', 'json').
                            If None, the processor attempts to infer the type.

        Returns:
            A list of dictionaries, where each dictionary represents a parsed change.
            Each change dict will typically include:
            - 'type': Inferred or specified type of change (e.g., 'code_change', 'task_update', 'json_update', 'generic_update').
            - 'summary': A brief summary of the change.
            - 'details': More specific details or context (e.g., parsed lines for diff, full JSON object).
            - 'raw_snippet': The original text snippet related to this specific change.
        """
        if not changelog_content:
            return []

        # Attempt to infer type if not provided
        if changelog_type is None:
            changelog_type = self._infer_changelog_type(changelog_content)

        try:
            if changelog_type == 'git_diff':
                return self._parse_git_diff(changelog_content)
            elif changelog_type == 'json':
                return self._parse_json_changelog(changelog_content)
            elif changelog_type == 'task_list' or changelog_type == 'documentation':
                # Treat task lists and documentation updates similarly for simple text parsing
                return self._parse_simple_text_changelog(changelog_content, changelog_type)
            else:
                # Default to simple text parsing for unknown or generic types
                return self._parse_simple_text_changelog(changelog_content, 'generic_update')
        except Exception as e:
            # Catch all exceptions during parsing to prevent module crashes
            return [{
                "type": "parsing_error",
                "summary": f"Failed to parse changelog of type '{changelog_type}': {e}",
                "details": str(e),
                "raw_snippet": changelog_content
            }]

    def _infer_changelog_type(self, content: str) -> str:
        """
        Infers the type of changelog based on its content heuristics.
        """
        content_lower = content.strip().lower()

        # Git diff heuristic
        if content.startswith("diff --git") or re.search(r"--- a/", content) or re.search(r"\+\+\+ b/", content):
            return 'git_diff'

        # JSON heuristic
        if (content.strip().startswith('{') and content.strip().endswith('}')) or \
           (content.strip().startswith('[') and content.strip().endswith(']')):
            try:
                json.loads(content)
                return 'json'
            except json.JSONDecodeError:
                pass # Not a valid JSON, fall through

        # Task list/documentation heuristic (bullet points, numbered lists)
        if re.search(r"^\s*-\s*|^\s*\*\s*|^\s*\d+\.\s*", content, re.MULTILINE):
            return 'task_list' # Using task_list as a general term for list-like text updates

        return 'generic_update'

    def _parse_git_diff(self, diff_content: str) -> List[Dict[str, Any]]:
        """
        Parses a git diff formatted changelog.
        Extracts changed files, additions, and deletions, providing parsed details
        and the original raw diff snippet for each file's changes.
        """
        changes: List[Dict[str, Any]] = []
        lines = diff_content.splitlines()
        
        current_file = None
        current_file_diff_lines: List[str] = [] # Stores all lines for the current file's raw diff
        current_file_parsed_summary: List[str] = [] # Stores parsed summary lines for the current file

        line_num_a = 0 # Line number in original file
        line_num_b = 0 # Line number in new file

        for line in lines:
            if line.startswith("diff --git"):
                # If we were processing a previous file, finalize its changes
                if current_file and current_file_diff_lines:
                    changes.append(self._format_code_change(
                        current_file,
                        current_file_parsed_summary,
                        "\n".join(current_file_diff_lines)
                    ))
                
                # Start processing a new file
                current_file = self._extract_filename_from_diff_header(line)
                current_file_diff_lines = [line]
                current_file_parsed_summary = []
                line_num_a = 0
                line_num_b = 0
                continue

            # Add current line to the raw diff snippet for the current file
            current_file_diff_lines.append(line)

            if line.startswith("@@"):
                # Parse hunk header to get starting line numbers
                match = re.search(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
                if match:
                    line_num_a = int(match.group(1))
                    line_num_b = int(match.group(2))
                    current_file_parsed_summary.append(f"Hunk: original line {line_num_a}, new line {line_num_b}")
                else:
                    current_file_parsed_summary.append(f"Hunk: {line}") # Fallback if regex fails
                continue

            if line.startswith("+"):
                current_file_parsed_summary.append(f"Added line {line_num_b}: {line[1:].strip()}")
                line_num_b += 1
            elif line.startswith("-"):
                current_file_parsed_summary.append(f"Removed line {line_num_a}: {line[1:].strip()}")
                line_num_a += 1
            elif line.startswith(" "): # Context line
                current_file_parsed_summary.append(f"Context line {line_num_b}: {line[1:].strip()}")
                line_num_a += 1
                line_num_b += 1
            # Other diff meta-lines (e.g., '--- a/', '+++ b/') are included in raw_snippet
            # but not explicitly parsed into current_file_parsed_summary unless they were '@@'
            
        # Add the last file's changes if any
        if current_file and current_file_diff_lines:
            changes.append(self._format_code_change(
                current_file,
                current_file_parsed_summary,
                "\n".join(current_file_diff_lines)
            ))

        if not changes and diff_content.strip():
            # If no structured changes found but content exists, return a generic entry
            return [{
                "type": "code_change",
                "summary": "Detected code diff but no specific file changes parsed (potentially malformed or only metadata changes).",
                "details": diff_content,
                "raw_snippet": diff_content
            }]
            
        return changes

    def _extract_filename_from_diff_header(self, header_line: str) -> str:
        """
        Extracts the filename from a 'diff --git' header.
        Prioritizes the 'b/' path as it represents the new file.
        """
        match = re.search(r"diff --git a/([^ ]+) b/([^ ]+)", header_line)
        if match:
            # Use the 'new' file name (b path)
            return match.group(2) 
        return "unknown_file"


    def _format_code_change(self, filename: str, parsed_summary_lines: List[str], raw_diff_snippet: str) -> Dict[str, Any]:
        """
        Helper to format a code change entry.
        """
        details_str = "\n".join(parsed_summary_lines) if parsed_summary_lines else "No detailed changes parsed."
        return {
            "type": "code_change",
            "summary": f"Changes in file: {filename}",
            "details": details_str,
            "filename": filename,
            "raw_snippet": raw_diff_snippet
        }

    def _parse_json_changelog(self, json_content: str) -> List[Dict[str, Any]]:
        """
        Parses a JSON formatted changelog.
        Assumes the JSON is either a list of change objects or a single change object.
        """
        try:
            data = json.loads(json_content)
            parsed_changes: List[Dict[str, Any]] = []

            if isinstance(data, list):
                # Assume each item in the list is a distinct change
                for item in data:
                    if isinstance(item, dict):
                        # Ensure basic required fields or provide defaults
                        item.setdefault('type', 'json_update')
                        item.setdefault('summary', item.get('description', item.get('message', 'No summary provided.')))
                        # Store the entire dictionary as details for rich context
                        parsed_changes.append({
                            "type": item['type'],
                            "summary": item['summary'],
                            "details": item,
                            "raw_snippet": json.dumps(item)
                        })
                    else:
                        parsed_changes.append({
                            "type": "json_parse_warning",
                            "summary": "Non-dictionary item found in JSON changelog list.",
                            "details": str(item),
                            "raw_snippet": json.dumps(item) # Attempt to serialize non-dict item for raw_snippet
                        })
                return parsed_changes
            elif isinstance(data, dict):
                # Assume it's a single change object
                data.setdefault('type', 'json_update')
                data.setdefault('summary', data.get('description', data.get('message', 'No summary provided.')))
                return [{
                    "type": data['type'],
                    "summary": data['summary'],
                    "details": data,
                    "raw_snippet": json_content
                }]
            else:
                return [{
                    "type": "json_parse_error",
                    "summary": "JSON content is neither an object nor a list of objects.",
                    "details": str(data),
                    "raw_snippet": json_content
                }]
        except json.JSONDecodeError as e:
            return [{
                "type": "parsing_error",
                "summary": f"Invalid JSON format: {e}",
                "details": str(e),
                "raw_snippet": json_content
            }]

    def _parse_simple_text_changelog(self, text_content: str, inferred_type: str) -> List[Dict[str, Any]]:
        """
        Parses simple text-based changelogs (e.g., task lists, documentation updates).
        Breaks down by lines, identifying bullet points, numbered lists, or treating
        each significant line as an update.
        """
        changes: List[Dict[str, Any]] = []
        lines = text_content.splitlines()
        
        for line in lines:
            stripped_line = line.strip()
            if not stripped_line:
                continue

            # Identify common list markers (bullet points, numbered lists)
            match_bullet = re.match(r"^[*-]\s*(.*)", stripped_line)
            match_numbered = re.match(r"^\d+\.\s*(.*)", stripped_line)

            if match_bullet:
                summary = match_bullet.group(1).strip()
                changes.append({
                    "type": inferred_type,
                    "summary": summary,
                    "details": stripped_line, # The entire stripped line with marker
                    "raw_snippet": line # The original line
                })
            elif match_numbered:
                summary = match_numbered.group(1).strip()
                changes.append({
                    "type": inferred_type,
                    "summary": summary,
                    "details": stripped_line,
                    "raw_snippet": line
                })
            else:
                # If no specific formatting, treat each non-empty line as a general update.
                # This ensures all content is captured.
                changes.append({
                    "type": inferred_type,
                    "summary": stripped_line,
                    "details": stripped_line,
                    "raw_snippet": line
                })
        
        if not changes and text_content.strip():
            # If no line-by-line structured changes, but there was content,
            # treat the entire content as one large generic update.
            return [{
                "type": inferred_type,
                "summary": "General text update or unformatted content.",
                "details": text_content,
                "raw_snippet": text_content
            }]
        
        return changes
```