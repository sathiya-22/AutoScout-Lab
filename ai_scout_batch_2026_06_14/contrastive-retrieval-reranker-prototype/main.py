```python
import os
import google.generativeai as genai
from config import Settings
import logging

# Configure basic logging for better feedback
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def rerank_documents(query: str, documents: list[str], settings: Settings) -> list[str]:
    """
    Reranks a list of documents based on relevance to a query using a generative AI model.
    """
    if not settings.gemini_api_key:
        logging.error("GEMINI_API_KEY is not set. Cannot perform reranking.")
        return []

    genai.configure(api_key=settings.gemini_api_key)
    model = genai.GenerativeModel(settings.model_name)

    # Construct the prompt for reranking
    document_list_str = "\n".join([f"{i+1}. {doc}" for i, doc in enumerate(documents)])
    prompt = (
        f"Query: {query}\n\n"
        f"Documents to re-rank:\n{document_list_str}\n\n"
        f"Please re-rank these documents from most to least relevant to the query. "
        f"Output only the re-ordered numbered list of the document *titles* (the original text provided), "
        f"without any additional text or explanations. Each document title should be on a new line, prefixed with its new rank number.\n"
        f"Example:\n1. Document Title C\n2. Document Title A\n3. Document Title B"
    )

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=settings.temperature,
                max_output_tokens=settings.max_tokens,
            )
        )
        if response.candidates:
            reranked_text = response.candidates[0].content.parts[0].text
            # Parse the re-ranked text into a list of strings, removing numbering
            reranked_docs = [line.strip().lstrip('0123456789.- ').strip() for line in reranked_text.split('\n') if line.strip()]
            return reranked_docs
        else:
            logging.warning("No candidates found in the model response.")
            return []
    except Exception as e:
        logging.error(f"Error during reranking: {e}")
        return []

def main():
    settings = Settings() # Pydantic will read GEMINI_API_KEY from environment variables

    if not settings.gemini_api_key:
        logging.error("GEMINI_API_KEY environment variable not set. Please set it to run the prototype.")
        return

    # Sample query and documents
    query = "latest advancements in artificial intelligence"
    documents = [
        "The history of computing from mainframes to microprocessors.",
        "Recent breakthroughs in large language models and generative AI.",
        "The impact of AI on the healthcare industry and drug discovery.",
        "Understanding neural networks: A beginner's guide to deep learning.",
        "Quantum computing challenges and future prospects.",
        "Ethical considerations in the development of artificial intelligence.",
    ]

    logging.info("--- Original Documents ---")
    for i, doc in enumerate(documents):
        logging.info(f"{i+1}. {doc}")
    logging.info("-" * 30)

    logging.info("Reranking documents...")
    reranked_documents = rerank_documents(query, documents, settings)

    if reranked_documents:
        logging.info("\n--- Reranked Documents ---")
        for i, doc in enumerate(reranked_documents):
            logging.info(f"{i+1}. {doc}")
    else:
        logging.warning("Failed to rerank documents or no reranked documents returned.")

if __name__ == "__main__":
    main()
```
