Retrieval systems often face the challenge of presenting the most relevant information first, even after an initial search. While a retriever can fetch a broad set of relevant documents, their internal ranking might not be perfectly optimized for a specific query, leading to potentially less effective information consumption.

This prototype addresses this by implementing a Contrastive Retrieval Reranker. It uses a large language model (LLM) to refine the ranking of an initially retrieved set of documents. Given a query and a list of documents, the LLM analyzes each document's content against the query, contrasting their relevance to produce a new, more precise order. The `gemini-2.5-flash` model is utilized for its efficiency and strong contextual understanding, ensuring that the most pertinent documents are elevated to the top.

To use this prototype:
1. Ensure Python 3.9+ is installed.
2. Install the required packages: `pip install -r requirements.txt`
3. Set your Google Gemini API key as an environment variable: `export GEMINI_API_KEY='your_api_key_here'` (or create a `.env` file in the project root with `GEMINI_API_KEY=your_api_key_here`).
4. Run the main script: `python main.py`
The console will display the original list of documents followed by their reranked order.
