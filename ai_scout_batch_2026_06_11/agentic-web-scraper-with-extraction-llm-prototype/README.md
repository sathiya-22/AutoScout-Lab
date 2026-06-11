This project provides a prototype for an agentic web scraper powered by a Large Language Model (LLM). Traditional web scraping often relies on brittle CSS selectors or XPath expressions, which require constant updates as website layouts evolve. This approach introduces significant maintenance overhead and limits flexibility.

Our solution leverages an LLM (Google's Gemini-2.5-Flash) to act as an intelligent agent. Instead of predefined rules, the LLM processes the raw HTML content and extracts structured information based on natural language instructions and a desired JSON schema. This makes the scraper highly resilient to minor layout changes and adaptable to a wide range of extraction tasks, as the LLM intelligently interprets the page content. The "agentic" aspect comes from the LLM's ability to understand the page context and fulfill complex data extraction goals.

To use this prototype:
1.  **Setup Environment**: Create a virtual environment and install dependencies:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    pip install -r requirements.txt
    ```
2.  **API Key**: Obtain a Google Gemini API key and create a `.env` file in the project root:
    ```
    GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
    ```
3.  **Run**: Execute the main script:
    ```bash
    python main.py
    ```
4.  **Customize**: Modify `main.py` to target different URLs and define specific extraction tasks in natural language.
*Note: The model "gemini-2.5-flash" is specified as per requirements. If you encounter issues (e.g., model not found), consider adjusting to an available model like "gemini-1.5-flash".*
