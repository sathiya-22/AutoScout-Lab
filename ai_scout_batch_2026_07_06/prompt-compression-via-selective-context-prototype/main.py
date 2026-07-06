```python
import os
import google.generativeai as genai
from config import Config
import logging

# Configure logging for better visibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_gemini_model(config: Config):
    """
    Configures the Google Generative AI client and initializes the model.
    Raises an exception if API key configuration fails.
    """
    try:
        if not config.api_key:
            raise ValueError("GEMINI_API_KEY is not set. Please set the environment variable.")
        genai.configure(api_key=config.api_key)
        model = genai.GenerativeModel(
            model_name=config.model_name,
            generation_config=genai.GenerationConfig(
                temperature=config.temperature,
                max_output_tokens=config.max_tokens,
            )
        )
        logging.info(f"Initialized Gemini model: {config.model_name}")
        return model
    except ValueError as ve:
        logging.critical(f"Configuration error: {ve}")
        raise
    except Exception as e:
        logging.critical(f"Failed to configure Gemini model or API client: {e}")
        raise

def compress_context(llm_model, long_text: str, user_query: str) -> str:
    """
    Uses the LLM to extract or summarize key information from a long text
    relevant to a specific user query, creating a compressed context.
    """
    compression_prompt = f"""
Given the following extensive document and a specific user question, your task is to extract or summarize only the most crucial and directly relevant information from the document that is essential for answering the user's question.

Focus on key facts, entities, relationships, and concepts. Do NOT answer the question yet. Simply provide the condensed, relevant context. If no information is directly relevant, state "No relevant information found."

---
Document:
{long_text}

---
User Question:
{user_query}

---
Relevant Information (condensed for the user's question):
"""
    logging.info("Attempting to compress context...")
    try:
        response = llm_model.generate_content(compression_prompt)
        compressed_text = response.text.strip()
        logging.info(f"Context compressed. Original length: {len(long_text)} chars, Compressed length: {len(compressed_text)} chars.")
        return compressed_text
    except Exception as e:
        logging.error(f"Error during context compression: {e}")
        return "Error during context compression."

def answer_question_with_context(llm_model, context: str, user_query: str) -> str:
    """
    Answers a user's question using only the provided context.
    """
    answer_prompt = f"""
Using ONLY the following provided context, answer the user's question.
If the answer cannot be found or directly inferred from the provided context, state clearly: "The answer cannot be found in the provided context."
Do not use any outside knowledge.

---
Context:
{context}

---
User Question:
{user_query}

---
Answer:
"""
    logging.info("Attempting to answer question with provided context...")
    try:
        response = llm_model.generate_content(answer_prompt)
        answer = response.text.strip()
        logging.info("Question answered.")
        return answer
    except Exception as e:
        logging.error(f"Error during question answering: {e}")
        return "Error during question answering."

def main():
    """Main function to run the prompt compression and answering demo."""
    logging.info("Starting Prompt Compression via Selective Context demo.")
    try:
        config = Config()
        gemini_model = setup_gemini_model(config)
    except Exception:
        # setup_gemini_model already logs critical errors
        return

    # --- Example Scenario ---
    long_document = """
The Amazon rainforest is the largest tropical rainforest in the world, covering much of northwestern Brazil and extending into Peru, Colombia, Ecuador, Bolivia, Guyana, Suriname, and French Guiana. It is characterized by immense biodiversity, housing millions of species of insects, plants, birds, and other animals. The rainforest plays a crucial role in regulating the Earth's climate by absorbing vast amounts of carbon dioxide. Deforestation, primarily due to cattle ranching and agriculture, poses a significant threat to the Amazon, leading to habitat loss and increased carbon emissions. Efforts are underway by various organizations and governments to protect this vital ecosystem. One such effort involves sustainable land management practices and the creation of protected areas. The indigenous communities living in the Amazon have a deep
