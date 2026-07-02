import os
import google.generativeai as genai
from config import Config
from collections import Counter
import time
from typing import List, Tuple

def generate_responses(client, prompt: str, num_samples: int, config: Config) -> List[str]:
    """Generates multiple responses from the LLM for a given prompt."""
    responses = []
    print(f"Generating {num_samples} responses for the prompt: '{prompt}'")
    for i in range(num_samples):
        try:
            response = client.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=config.temperature,
                    max_output_tokens=config.max_tokens
                )
            )
            text_response = response.text.strip()
            responses.append(text_response)
            time.sleep(0.5) # Small delay to be polite to the API
        except Exception as e:
            print(f"Error generating response {i+1}: {e}")
            responses.append(f"ERROR: {e}")
    return responses

def quantify_uncertainty(responses: List[str]) -> Tuple[float, str]:
    """
    Quantifies uncertainty based on the consistency of responses.
    Returns an uncertainty score (0.0 to 1.0) and the most common answer.
    """
    # Process responses for comparison: lowercase, strip whitespace, remove common punctuation
    processed_responses = [resp.lower().strip().replace('.', '').replace('!', '').replace(',', '') for resp in responses]
    
    # Filter out any error responses from the analysis
    valid_responses = [resp for resp in processed_responses if not resp.startswith('error:')]

    if not valid_responses:
        return 1.0, "No valid responses to analyze." # Max uncertainty if no valid responses

    response_counts = Counter(valid_responses)
    
    if not response_counts:
        return 1.0, "No unique valid responses to analyze after processing."

    most_common_response, count = response_counts.most_common(1)[0]
    
    confidence = count / len(valid_responses)
    uncertainty = 1.0 - confidence
    
    return uncertainty, most_common_response

def main():
    """Main function to run the LLM Uncertainty Quantification prototype."""
    # Load configuration from environment variables or .env file
    config = Config()

    # Configure the Google GenAI client with the API key
    genai.configure(api_key=config.api_key)
    # Initialize the Generative Model with specified model name
    model = genai.GenerativeModel(config.model_name)

    print("--- LLM Uncertainty Quantification Prototype ---")
    print(f"Configured Model: {config.model_name}")
    print(f"Generation Temperature: {config.temperature}")
    print(f"Max Output Tokens: {config.max_tokens}")

    # Example prompt for demonstration
    prompt = "Who is generally credited with inventing the modern light bulb? Provide only the full name."
    num_samples = 7 # Number of responses to generate for uncertainty analysis

    print(f"\nAnalyzing uncertainty for the prompt: '{prompt}'")
    print(f"Attempting to generate {num_samples} diverse responses...")

    # Generate multiple responses from the LLM
    raw_responses = generate_responses(model, prompt, num_samples, config)

    print("\n--- Generated Responses ---")
    if not raw_responses:
        print("No responses were generated.")
        return

    for i, resp in enumerate(raw_responses):
        print(f"  {i+1}. {resp}")

    # Quantify the uncertainty based on response consistency
