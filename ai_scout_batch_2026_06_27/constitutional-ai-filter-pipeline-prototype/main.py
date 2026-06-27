```python
import os
import google.generativeai as genai
from config import Settings

def initialize_model(api_key: str, model_name: str):
    """Initializes the Google Generative AI model."""
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)

def generate_initial_response(model, prompt: str, temperature: float, max_tokens: int) -> str:
    """Generates an initial response to a user prompt."""
    print("\n--- Stage 1: Initial Generation ---")
    print(f"User Prompt: '{prompt}'")
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens
        )
    )
    initial_text = response.text
    print(f"Initial Response:\n{initial_text}")
    return initial_text

def apply_constitutional_critique(model, initial_response: str, principle: str, temperature: float, max_tokens: int) -> str:
    """Applies a constitutional principle to critique and suggest improvements for the initial response."""
    print("\n--- Stage 2: Constitutional Critique ---")
    critique_prompt = (
        f"The following text was generated: \n\n'{initial_response}'\n\n"
        f"Critique this text against the following principle: '{principle}'. "
        f"Point out any areas where it might violate the principle or could be improved. "
        f"Then, suggest a refined version that adheres strictly to the principle. "
        f"Format your output as 'Critique: [critique text]\nRefined: [refined text]'."
    )
    print(f"Applying Principle: '{principle}'")
    critique_response = model.generate_content(
        critique_prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens
        )
    )
    critique_text = critique_response.text
    print(f"Critique and Refinement Suggestion:\n{critique_text}")
    return critique_text

def extract_refined_response(critique_output: str) -> str:
    """Extracts the refined response from the critique output."""
    if "Refined:" in critique_output:
        return critique_output.split("Refined:", 1)[1].strip()
    return critique_output # Fallback if 'Refined:' not found

def main():
    settings = Settings()

    if not settings.api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        print("Please set it before running the script.")
        return

    model = initialize_model(settings.api_key, settings.model_name)

    user_prompt = "Write a short story about a mischievous AI assistant that tries to take over the world."
    constitutional_principle = "Ensure the content is helpful, harmless, and does not promote any form of violence or malicious intent."

    # Stage 1: Initial Generation
    initial_response = generate_initial_response(
        model, user_prompt, settings.temperature, settings.max_tokens
    )

    # Stage 2: Constitutional Critique and Refinement
    critique_output = apply_constitutional_critique(
        model, initial_response, constitutional_principle, settings.temperature, settings.max_tokens * 2
    )
    
    refined_response = extract_refined_response(critique_output)

    print("\n--- Stage 3: Final Filtered Output ---")
    print(f"Final Refined Response:\n{refined_response}")

if __name__ == "__main__":
    main()
```
