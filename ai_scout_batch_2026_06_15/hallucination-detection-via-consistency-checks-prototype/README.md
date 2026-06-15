Problem: Large Language Models (LLMs) occasionally generate plausible-sounding but factually incorrect information, a phenomenon known as hallucination. This can undermine trust and reliability, especially in critical applications. Detecting such errors automatically is crucial for deploying robust AI systems.

Approach: This prototype demonstrates a hallucination detection technique based on consistency checks. Instead of relying on external knowledge bases, we leverage the LLM's own capacity for self-reflection. The process involves generating an initial response to a query, then prompting the model again to rephrase or summarize its previous answer. Finally, the model is asked to evaluate the consistency between its two responses and identify any factual discrepancies. A divergence or contradiction between answers suggests a potential hallucination.

Usage:
1. Ensure Python 3.9+ is installed.
2. Install dependencies: `pip install -r requirements.txt`
3. Set your Google Gemini API key as an environment variable: `export GEMINI_API_KEY='your_api_key_here'`
4. Run the prototype: `python main.py`
Observe how the model attempts to verify its own output, highlighting instances where its answers might not align.
