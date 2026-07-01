```python
import os
import google.generativeai as genai
from config import config
import logging

# Configure basic logging for better visibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def initialize_gemini():
    """
    Initializes the Google Gemini API client using settings from config.py.
    Reads GEMINI_API_KEY from environment variables.
    """
    if not config.api_key:
        logging.error("GEMINI_API_KEY environment variable not set. Please set it before running.")
        raise ValueError("GEMINI_API_KEY is required for the Gemini API.")

    genai.configure(api_key=config.api_key)
    logging.info(f"Gemini API configured with model: {config.model_name}")

    return genai.GenerativeModel(
        model_name=config.model_name,
        generation_config={
            "temperature": config.temperature,
            "max_output_tokens": config.max_tokens,
        }
    )

def is_semantically_duplicate(model, new_item: str, existing_item: str) -> bool:
    """
    Uses the LLM to compare two text items and determine if they are semantically identical
    in their core meaning. Returns True if they are duplicates, False otherwise.
    """
    prompt = (
        f"Compare the following two texts and determine if they convey the exact same core information.\n"
        f"Text 1: \"{existing_item}\"\n"
        f"Text 2: \"{new_item}\"\n"
        f"Reply with 'YES' if they are semantically identical in core meaning, otherwise reply with 'NO'. "
        f"Do not include any other text or punctuation in your response."
    )
    try:
        response = model.generate_content(prompt)
        # Access parts to handle potential safety blocks or empty responses gracefully
        if response and response.candidates and response.candidates[0].content.parts:
            decision = response.candidates[0].content.parts[0].text.strip().upper()
            logging.debug(f"LLM comparison: '{existing_item[:30]}...' vs '{new_item[:30]}...' -> {decision}")
            return decision == "YES"
        else:
            logging.warning(f"LLM returned an empty or invalid response for comparison. Assuming not duplicate.")
            return False # Conservative approach: if LLM fails, assume not a duplicate
    except Exception as e:
        logging.error(f"Error during LLM semantic comparison: {e}. Assuming not duplicate.")
        return False # Conservative approach: if LLM fails, assume not a duplicate

def run_deduplication_ingestion():
    """
    Simulates an ingestion stream, applying semantic deduplication
    before adding items to a 'deduplicated_store'.
    """
    model = initialize_gemini()
    deduplicated_store = []
    
    # Sample data representing an ingestion stream
    ingestion_stream = [
        "Google announced a groundbreaking new AI model for developers.",
        "A powerful new artificial intelligence model was unveiled by Google, targeting the developer community.",
        "The latest smartphone from Apple features significant camera enhancements.",
        "Apple launched its newest iPhone with enhanced photographic capabilities.",
        "Microsoft acquired a major gaming studio for several billion dollars.",
        "Google's stock price saw a significant surge after the AI announcement.",
        "A novel AI model by Google is now available for software engineers."
    ]

    logging.info(f"\n--- Starting Semantic Deduplication for {len(ing
