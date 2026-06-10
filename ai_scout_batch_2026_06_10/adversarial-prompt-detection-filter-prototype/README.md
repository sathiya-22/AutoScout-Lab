This project demonstrates an "Adversarial Prompt Detection Filter" using Google's Gemini LLM. The core problem is to prevent malicious or unwanted prompts from reaching a primary language model, which could lead to undesirable outputs, security vulnerabilities, or misuse.

Our approach employs a two-stage filtering mechanism. First, an input prompt is sent to a dedicated "filter" LLM. This filter LLM is specifically instructed to classify the prompt's intent as either "SAFE" or "ADVERSARIAL". Only if the filter model deems the prompt "SAFE" is it then forwarded to the "main" LLM for processing and generating a response. If classified as "ADVERSARIAL", the prompt is blocked, and a warning is issued to the user. This acts as a robust front-line defense against prompt injection, harmful content requests, or other adversarial inputs.

To use the prototype:
1.  Set your `GEMINI_API_KEY` environment variable.
2.  Install dependencies: `pip install -r requirements.txt`.
3.  Run `python main.py`. You can then interact with the filtered LLM in your terminal.
