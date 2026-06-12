This project provides a Python prototype for a Temporal Reasoning Module designed to enhance Question Answering (QA) systems.

**Problem:** Traditional QA systems often struggle with questions that require understanding temporal relationships, sequences, durations, and causality (e.g., "What happened before X?", "How long did Y last?", "When did Z occur relative to A and B?"). This limitation can lead to incomplete or inaccurate answers for time-sensitive queries, reducing the overall utility of a QA system in dynamic environments.

**Approach:** Our module leverages the `gemini-2.5-flash` large language model from Google AI as its core reasoning engine. By crafting a detailed prompt, the LLM is instructed to act as an expert temporal reasoner. It analyzes user questions and optional contextual information, identifies all relevant temporal entities and relationships, performs necessary temporal calculations (e.g., duration, sequencing), and then synthesizes a precise, temporally accurate answer. This strategy offloads complex temporal logic to a powerful LLM, making the system highly adaptable and capable of handling diverse temporal queries.

**Usage:**
1.  **Prerequisites:** Ensure Python 3.8+ is installed.
2.  **Setup Environment:**
    *   Create a virtual environment: `python -m venv .venv`
    *   Activate it:
        *   Windows: `.venv\Scripts\activate`
        *   Linux/macOS: `source .venv/bin/activate`
    *   Install dependencies: `pip install -r requirements.txt`
    *   Obtain a `GEMINI_API_KEY` from Google AI Studio.
    *   Set the API key as an environment variable (e.g., `export GEMINI_API_KEY='YOUR_API_KEY'` on Linux/macOS, or create a `.env` file in the project root with `GEMINI_API_KEY=YOUR_API_KEY`).
3.  **Run the Demo:** Execute `python main.py` to see the temporal reasoning module in action with several example questions.
