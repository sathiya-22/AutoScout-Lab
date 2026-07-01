Data ingestion processes frequently encounter the challenge of duplicate content. Traditional exact-match deduplication, relying on hashes or simple string comparisons, fails when information is semantically similar but syntactically distinct—think rephrased articles or slightly varied product descriptions. This leads to inefficient storage, skewed analytics, and a degraded user experience, as redundant data clutters systems and obscures unique insights.

This prototype addresses "Semantic Deduplication at Ingestion" by leveraging the advanced understanding capabilities of Large Language Models (LLMs). Instead of merely comparing text literally, the system uses Google's Gemini-1.5-Flash model to grasp the core meaning of incoming data. Each new item is semantically analyzed against an existing store of unique documents. If the LLM determines that a new item conveys essentially the same core information as an item already present, it is identified as a semantic duplicate and discarded. Otherwise, it is added to the unique data store, ensuring high data quality.

To run this prototype:
1.  Obtain a Google Gemini API key.
2.  Set your API key as an environment variable: `export GEMINI_API_KEY="your_api_key_here"`.
3.  Install the required dependencies: `pip install -r requirements.txt`.
4.  Execute the main script: `python main.py`.
The script will simulate an ingestion stream, demonstrating how items are processed and identified as unique or semantic duplicates.
