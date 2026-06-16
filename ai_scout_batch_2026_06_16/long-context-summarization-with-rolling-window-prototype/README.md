This project provides a Python prototype for long-context summarization using a rolling window approach. Large Language Models (LLMs) have inherent context window limitations, making it challenging to summarize extremely long documents directly. This prototype addresses this by breaking down the text into manageable chunks.

The core idea is to iteratively summarize segments of the document. First, the input text is split into overlapping chunks. Each chunk is then summarized independently using a powerful LLM. These "partial summaries" are concatenated to form a condensed representation of the original document. Finally, this combined text of summaries is itself summarized by the LLM to produce a single, comprehensive overview of the entire original document. This hierarchical strategy allows for summarizing texts of virtually any length.

To use the prototype:
1. Ensure you have a Google Cloud Project with the Gemini API enabled.
2. Set your `GEMINI_API_KEY` as an environment variable.
3. Install the required dependencies:
