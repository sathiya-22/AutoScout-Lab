```python
import os
import json
import google.generativeai as genai
from config import Settings

def main():
    """
    Main function to demonstrate the LLM-powered data cleaning pipeline.
    It loads configuration, defines sample dirty data, constructs a cleaning prompt,
    sends it to the Gemini LLM, and prints the original and cleaned data.
    """
    settings = Settings()

    if not settings.api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        print("Please set it (e.g., `export GEMINI_API_KEY='your_key'`) or create a .env file.")
        return

    genai.configure(api_key=settings.api_key)

    # Note: Using 'gemini-1.5-flash' as 'gemini-2.5-flash' is not a publicly available model name.
    model = genai.GenerativeModel(
        model_name=settings.model_name,
        generation_config={
            "temperature": settings.temperature,
            "max_output_tokens": settings.max_tokens,
            "response_mime_type": "application/json", # Request JSON output directly
        }
    )

    # Sample dirty data requiring cleaning
    dirty_data = [
        {"Name": "john doe", "age": "30", "email": "JOHN.DOE@example.com"},
        {"Name": "Jane Smith", "age": "twenty five", "email": "jane@example.com"},
        {"Name": "bOB", "age": "42", "email": "bob@domain"},  # Invalid email
        {"Name": "Alice", "age": None, "email": "alice@example.com"},
        {"Name": "charlie brown", "age": "35", "email": "charlie.brown@peanuts"}, # Invalid domain
        {"Name": "diana prince", "age": "forty", "email": "diana@amazon.com"}
    ]

    # Define the cleaning instructions for the LLM
    cleaning_prompt = f"""
    You are an expert data cleaning assistant. Your task is to meticulously clean the provided raw data according to the specified rules.
    The output MUST be a JSON array of objects, strictly following the cleaned structure.
    Do not include any additional text or explanations outside the JSON.

    Cleaning Rules:
    1.  'Name': Standardize to title case (e.g., 'john doe' -> 'John Doe').
    2.  'age': Convert to an integer. If the value cannot be reliably converted to a number (e.g., 'twenty five', None, empty string), set it to null.
    3.  'email': Convert to lowercase. Validate the email format to ensure it contains exactly one '@' symbol and at least one '.' after the '@' for a valid domain. If the format is invalid, set the email to null.

    Raw Data to Clean:
    {json.dumps(dirty_data, indent=2)}

    Cleaned Data (JSON array):
    """

    print("--- Original Data ---")
    print(json.dumps(dirty_data, indent=2))
    print("\n--- Sending to LLM for Cleaning ---")

    try:
        # Generate content using the LLM
        response = model.generate_content(cleaning_prompt)
        response.resolve() # Ensure content is fully loaded

        cleaned_json_str = response.text
        cleaned_data = json.loads(cleaned_json_str)

        print("\n--- Cleaned Data ---")
        print(json.dumps(cleaned_data, indent=2))

    except Exception as e:
        print(f"\nAn error occurred during LLM interaction: {e}")
        print("Please check your API key, model name, and prompt formatting.")
        if hasattr(response, 'text'):
            print(f"Raw LLM response (if available):\n{response.text}")


if __name__ == "__main__":
    main()
```
