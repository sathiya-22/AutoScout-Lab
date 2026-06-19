import os
import google.generativeai as genai
from config import Settings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Main function to run the Retrieval Confidence Calibration Module demo.
    It simulates different retrieval scenarios and uses a Gemini model
    to assess the confidence of the retrieved document in answering a query.
    """
    settings = Settings()

    if not settings.api_key.get_secret_value():
        logging.error("GEMINI_API_KEY is not set. Please set the environment variable.")
        return

    genai.configure(api_key=settings.api_key.get_secret_value())
    logging.info(f"Google Generative AI configured with model: {settings.model_name}")

    model = genai.GenerativeModel(
        model_name=settings.model_name,
        generation_config={
            "temperature": settings.temperature,
            "max_output_tokens": settings.max_tokens,
        }
    )

    def assess_confidence(current_query: str, document: str) -> str:
        """
        Uses the configured LLM to assess the confidence of a document
        in answering a given query.
        """
        prompt = f"""
        You are an expert retrieval confidence calibration module. Your task is to assess
        how confidently the provided 'Retrieved Document' answers the 'User Query'.

        Evaluate the document based on:
        1.  **Directness & Completeness:** Does the document directly and fully answer the query?
        2.  **Specificity:** Is the information specific enough?
        3.  **Ambiguity:** Is there any ambiguity or missing crucial information?

        Provide a confidence score from 1 (very low confidence) to 5 (very high confidence)
        and a brief explanation for your score.

        ---
        User Query: {current_query}

        Retrieved Document:
        {document}
        ---

        Confidence Assessment:
        Confidence Score (1-5):
        Explanation:
        """
        logging.info(f"Assessing confidence for query: '{current_query}' with document snippet.")
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            logging.error(f"Error generating content: {e}")
            return f"Error: {e}"

    # --- Demo Scenarios ---

    # Scenario 1: High Confidence
    query_high = "What is the capital of France?"
    doc_high = """Paris is the capital and most populous city of France.
    It is located on the River Seine, in the north-central part of the country."""
    logging.info("\n--- High Confidence Scenario ---")
    confidence_output_high = assess_confidence(query_high, doc_high)
    print(f"Query: {query_high}\nRetrieved Document:\n{doc_high}\nConfidence Assessment:\n{confidence_output_high}\n")

    # Scenario 2: Low Confidence
    query_low = "What is the main export of Brazil?"
    doc_low = """Brazil is a large South American country known for its vibrant culture,
    Amazon rainforest, and extensive coastline. It has a significant agricultural sector."""
    logging.info("\n--- Low Confidence Scenario ---")
    confidence_output_low = assess_confidence(query_low, doc_low)
    print(f"Query: {query_low}\nRetrieved Document:\n{doc_low}\nConfidence Assessment:\n{confidence_output_low}\n")

    # Scenario 3: Ambiguous/Partial Confidence
    query_ambiguous = "What is the primary function of the human appendix?"
    doc_ambiguous = """The human appendix is a small, finger-shaped organ that projects
    from the large intestine. Its exact function has long been a subject of debate
    among scientists, with some theories suggesting it plays a role in immunity."""
    logging.info("\n--- Ambiguous/Partial Confidence Scenario ---")
    confidence_output_ambiguous = assess_confidence(query_ambiguous, doc_ambiguous)
    print(f"Query: {query_ambiguous}\nRetrieved Document:\n{doc_ambiguous}\nConfidence Assessment:\n{confidence_output_ambiguous}\n")

if __name__ == "__main__":
    main()
