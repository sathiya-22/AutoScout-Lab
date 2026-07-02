LLMs can produce text with varying levels of certainty, which isn't always apparent to the user. This lack of transparency can lead to misinterpretations or over-reliance on potentially uncertain information. This prototype addresses this by quantifying the inherent uncertainty in LLM responses.

Our approach involves generating multiple responses to the same prompt, leveraging the model's temperature setting to encourage diversity. By analyzing the consistency and divergence across these generated outputs, we derive an "uncertainty score." A higher score indicates greater variability in the LLM's answers, suggesting less confidence in any single definitive response. The system identifies the most frequent answer and calculates a confidence score based on its prevalence among the samples.

To use this prototype:
1. Ensure you have a Google Gemini API key. Set it as an environment variable: `export GEMINI_API_KEY='YOUR_API_KEY'` or create a `.env` file in the project root with `GEMINI_API_KEY='YOUR_API_KEY'`.
2. Install the necessary dependencies: `pip install -r requirements.txt`
3. Run the main script: `python main.py`
The output will display the individual responses, the most common answer found, and the calculated uncertainty score.
