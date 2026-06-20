import os
import sys
import google.generativeai as genai
from config import Config

def main():
    """
    Main function to run the Agentic Code Review Assistant.
    Reads code from a specified file, sends it to the Gemini model for review,
    and prints the review.
    """
    # Load configuration settings
    settings = Config()

    # Ensure API key is available
    if not settings.api_key:
        print("Error: GEMINI_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    # Configure the generative AI model
    genai.configure(api_key=settings.api_key)

    # Initialize the model with configured parameters
    generation_config = genai.types.GenerationConfig(
        temperature=settings.temperature,
        max_output_tokens=settings.max_tokens,
    )
    model = genai.GenerativeModel(
        model_name=settings.model_name,
        generation_config=generation_config
    )

    # Check for file path argument
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_code_file>", file=sys.stderr)
        sys.exit(1)

    code_file_path = sys.argv[1]

    # Read the code content from the specified file
    try:
        with open(code_file_path, 'r', encoding='utf-8') as f:
            code_content = f.read()
    except FileNotFoundError:
        print(f"Error: File not found at '{code_file_path}'", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{code_file_path}': {e}", file=sys.stderr)
        sys.exit(1)

    if not code_content.strip():
        print("Warning: The provided code file is empty or contains only whitespace. No review will be generated.")
        sys.exit(0)

    # Construct the prompt for the AI
    prompt = f"""You are an expert code reviewer. Analyze the following Python code for potential bugs,
security vulnerabilities, performance issues, style violations (e.g., PEP 8), and provide
suggestions for improvement. Be concise, actionable, and constructive.
Focus on critical and high-impact feedback first.

---
Code to review:
```python
{code_content}
```
---
Please provide your review in a structured format, e.g., using bullet points or sections for different categories."""

    print(f"Reviewing code from: {code_file_path}...\n")

    try:
        # Generate content from the model
        response = model.generate_content(prompt)
        print("--- AI Code Review ---")
        print(response.text)
        print("---------------------")
    except Exception as e:
        print(f"An error occurred during AI generation: {e}", file=sys.stderr)
        if "Blocked due to safety" in str(e):
            print("The content was blocked due to safety concerns. Please adjust the input.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
