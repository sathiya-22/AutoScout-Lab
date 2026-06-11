import os
import json
import requests
import google.generativeai as genai
from config import Config
from typing import Dict, Any, Optional

# Load configuration
config = Config()

# Configure the Generative AI model
genai.configure(api_key=config.gemini_api_key)
model = genai.GenerativeModel(
    model_name=config.model_name,
    generation_config={
        "temperature": config.temperature,
        "max_output_tokens": config.max_tokens,
    }
)

def fetch_page_content(url: str) -> str:
    """Fetches the text content of a given URL."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, timeout=15, headers=headers)
        response.raise_for_status() # Raise an exception for HTTP errors
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return ""

def extract_with_llm(html_content: str, extraction_task: str) -> Optional[Dict[str, Any]]:
    """
    Uses an LLM to extract structured data from HTML content based on a task.
    The LLM is prompted to output JSON.
    """
    if not html_content:
        return None

    # Truncate HTML to avoid exceeding context window limits
    # 8000 characters is a safe bet for many models, adjust if needed
    truncated_html = html_content[:8000]

    prompt = f"""
    You are an intelligent web scraping agent. Your goal is to extract specific information
    from the provided HTML content based on the user's request.
    Always output the extracted information as a JSON object.

    HTML Content:
    ---
    {truncated_html}
    ---

    Extraction Task: {extraction_task}

    Ensure your output is a valid JSON object. If a piece of information cannot be found, use null for its value.
    """

    try:
        response = model.generate_content(prompt)
        text_response = response.text.strip()

        # Attempt to parse JSON. Sometimes LLMs include conversational text.
        json_start = text_response.find('{')
        json_end = text_response.rfind('}')
        if json_start != -1 and json_end != -1 and json_end > json_start:
            json_str = text_response[json_start : json_end + 1]
            return json.
