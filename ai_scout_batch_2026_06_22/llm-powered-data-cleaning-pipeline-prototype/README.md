Data often arrives in inconsistent, incomplete, or incorrectly formatted states, posing a significant challenge for analysis, reporting, and machine learning. Manual data cleaning is a time-consuming, error-prone, and unscalable process.

This project presents a Python prototype for an LLM-powered data cleaning pipeline. It leverages the advanced natural language understanding and generation capabilities of Google's Gemini 1.5 Flash model to intelligently identify and correct common data quality issues. The approach involves defining clear cleaning rules within a prompt, allowing the LLM to process raw, messy data, and output a structured, cleaned dataset in JSON format. This automation dramatically reduces the effort and potential for human error in data preparation.

To use this prototype:
1.  **Set your API Key**: Obtain a Google Gemini API key and set it as an environment variable: `export GEMINI_API_KEY='your_api_key_here'`. Alternatively, create a `.env` file in the project root with `GEMINI_API_KEY='your_api_key_here'`.
2.  **Install Dependencies**: Run `pip install -r requirements.txt`.
3.  **Execute**: Run `python main.py` to see a demonstration of sample dirty data being transformed into a clean, structured output by the LLM.
