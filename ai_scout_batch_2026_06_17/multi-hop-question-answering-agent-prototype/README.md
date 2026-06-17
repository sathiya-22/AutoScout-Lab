This project demonstrates a Multi-Hop Question Answering Agent using the Google Gemini Pro model. Complex questions often require information synthesis from multiple sources or across several logical steps. A "multi-hop" approach breaks down such questions into simpler, sequential sub-questions, answers each one, and then synthesizes a final, comprehensive answer.

The core problem addressed is enabling an AI to perform multi-step reasoning. Traditional QA might struggle with questions that require combining facts from different "hops" of information retrieval. Our approach leverages the large language model's ability to plan, execute sub-tasks, and synthesize.

**Approach:**
The agent is designed to take a complex question, prompt the LLM to first decompose it into a series of intermediate steps or sub-questions, answer each of these steps, and finally consolidate the findings into a coherent final answer. This simulates a chain-of-thought process, making the reasoning transparent and robust.

**Usage:**
1.  **Set up your Google Gemini API Key:**
    Obtain a `GEMINI_API_KEY` from the Google AI Studio.
    Set it as an environment variable: `export GEMINI_API_KEY='your_api_key_here'`
2.  **Install dependencies:**
    `pip install -r requirements.txt`
3.  **Run the agent:**
    `python main.py`
The script will output the multi-hop reasoning process and the final answer for a predefined complex question.
