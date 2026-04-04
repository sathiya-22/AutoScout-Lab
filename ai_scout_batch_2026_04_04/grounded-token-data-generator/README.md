# Grounded Token Data Generator

## Project Overview

The 'Grounded Token Initialization' (GTI) method and similar approaches for extending an LLM's vocabulary critically depend on 'paired linguistic supervision'—natural language descriptions linked to new vocabulary tokens—to semantically ground these novel tokens. However, for large-scale vocabulary extensions, especially in dynamic or rapidly expanding domains, the manual generation, curation, and maintenance of high-quality descriptive text for thousands or millions of new tokens presents an enormous and expensive data engineering bottleneck.

The **Grounded Token Data Generator** is a semi-automated tool designed to alleviate this bottleneck. It leverages existing knowledge bases, structured data, and advanced Large Language Models (LLMs) to generate candidate descriptions for new tokens. The tool incorporates an active learning framework and a human-in-the-loop (HIL) interface, enabling efficient validation and refinement of generated data, alongside quality metrics to assess linguistic diversity and semantic coverage.

## Features

*   **Automated Description Generation**: Synthesize initial linguistic supervision candidates using LLMs and integrated knowledge sources.
*   **Knowledge Base Integration**: Connects to various external knowledge sources (Wikidata, domain-specific KBs, structured data) to enrich descriptions.
*   **LLM Agnostic**: Modular design allows integration with different LLM providers (e.g., OpenAI, Hugging Face models) via a unified interface.
*   **Prompt Management**: Centralized management and parameterization of LLM prompts for optimized description generation.
*   **Quality Assessment**: Built-in metrics to evaluate linguistic diversity, semantic coverage, fluency, and relevance of generated descriptions.
*   **Human-in-the-Loop (HIL) Interface**: A user-friendly web UI for human annotators to review, edit, and validate generated descriptions efficiently.
*   **Active Learning**: Strategies to intelligently prioritize descriptions for human review, maximizing annotation efficiency and minimizing manual effort.
*   **Structured Data Output**: Generates high-quality, semantically grounded paired linguistic supervision data in a structured format suitable for downstream LLM training.

## Architecture

The Grounded Token Data Generator is a modular, semi-automated system focusing on core generation, quality assessment, and human-in-the-loop (HIL) validation.

```
+---------------------------------------------------------------------------------------------------------------------------------------+
|                                                      Grounded Token Data Generator                                                    |
|                                                                                                                                       |
|  +---------------------+        +---------------------+                                       +-----------------------------------+   |
|  |     Data Input      |        | Knowledge Sources   |                                       |           HIL Frontend            |   |
|  | (new_tokens.csv)    |        | (Wikidata, DBs,    |                                       | (Review, Edit, Validate UI)       |   |
|  +----------+----------+        |  Web, etc.)         |                                       +-----------------+-----------------+   |
|             |                             |                                                                     |                     |
|             |                             |                                                                     |                     |
|  +----------v----------+                  |                                                                     |                     |
|  | Data Management     |                  |                                                                     |                     |
|  | (`src/data/input/`) |                  |                                                                     |                     |
|  +----------+----------+                  |                                                                     |                     |
|             |                             |                                                                     |                     |
|             |                             |                                                                     |                     |
|  +----------v----------+  +--------------v--------------+                                  +------------------v------------------+ |
|  | Generation Engine   |  | Knowledge Integration Layer |                                  | API Layer (`src/api/routes.py`)  | |
|  | (`src/core/generator.py`)<--> (`src/core/knowledge_integrator.py`)                     | (Tokens, Candidates, Feedback)    | |
|  +----------+----------+  +-----------------------------+                                  +------------------+------------------+ |
|             |                                                                                                   |                     |
|             |                                                                                                   |                     |
|  +----------v----------+                                                                                        |                     |
|  | LLM Integration     |                                                                                        |                     |
|  | (`src/core/llm_interface.py`, `src/core/prompt_manager.py`)                                                  |                     |
|  +----------+----------+                                                                                        |                     |
|             |                                                                                                   |                     |
|             |                                                                                                   |                     |
|  +----------v----------+        +-------------------------------------+      +--------------------------------+                     |
|  | Quality Assessment  |        | Active Learning Strategy            |      | Data Output                    |                     |
|  | (`src/core/quality_metrics.py`)<--> (`src/active_learning/strategy.py`)<----| (`src/data/output/generated_supervision.json`) | |
|  +----------+----------+        +-------------------------------------+      +--------------------------------+                     |
|             ^                                                                                                   |                     |
|             |                                                                                                   |                     |
|  +----------+----------+                                                                                        |                     |
|  | Configuration &     |                                                                                        |                     |
|  | Environment         |                                                                                        |                     |
|  | (`config.py`, `.env`)|                                                                                        |                     |
|  +---------------------+-----------------------------------------------------------------------------------------+---------------------+
```

### Key Components:

1.  **Configuration & Environment (`config.py`, `.env`)**: Centralized settings for API keys, database connections, LLM parameters, and system thresholds. `.env` handles sensitive credentials.
2.  **Knowledge Integration Layer (`src/core/knowledge_integrator.py`)**: Abstracts access to various external knowledge sources (structured databases, knowledge graphs like Wikidata, domain-specific KBs, web data) via a unified, extensible interface.
3.  **LLM Integration Layer (`src/core/llm_interface.py`, `src/core/prompt_manager.py`)**: Manages interactions with different LLMs for generating descriptions. `llm_interface.py` handles API calls, while `prompt_manager.py` centralizes and parameterizes prompts.
4.  **Generation Engine (`src/core/generator.py`)**: Orchestrates the creation of initial candidate descriptions. It queries the `Knowledge Integration Layer` and leverages the `LLM Integration Layer` to synthesize descriptions from various sources.
5.  **Quality Assessment Module (`src/core/quality_metrics.py`)**: Evaluates the linguistic diversity, semantic coverage, fluency, and relevance of generated descriptions using quantitative and qualitative metrics (e.g., embedding similarity, n-gram diversity, readability, factuality checks).
6.  **Human-in-the-Loop (HIL) & Active Learning (`src/api/`, `src/ui/`, `src/active_learning/strategy.py`)**:
    *   **API (`src/api/routes.py`)**: A lightweight RESTful API (e.g., Flask/FastAPI) for submitting tokens, retrieving candidates, and capturing human validation feedback.
    *   **UI (`src/ui/`)**: A simple web-based frontend for annotators to review, edit, and validate descriptions.
    *   **Active Learning Strategy (`src/active_learning/strategy.py`)**: Employs strategies (e.g., uncertainty or diversity sampling) to present the most informative examples to humans, maximizing efficiency.
7.  **Data Management (`src/data/`)**:
    *   **Input**: `src/data/input/new_tokens.csv` (or similar) holds novel tokens and optional metadata.
    *   **Output**: `src/data/output/generated_supervision.json` stores the final, validated paired linguistic supervision.
8.  **Utilities (`src/utils/`)**: Common helper functions for data loading (`data_loaders.py`), text preprocessing (`text_processors.py`), and other reusable logic.

## Data Flow

1.  **Ingestion**: New tokens (and optional initial context) are ingested from `src/data/input/new_tokens.csv`.
2.  **Generation**: The `Generation Engine` orchestrates queries to the `Knowledge Integration Layer` and calls the `LLM Integration Layer` (using prompts from `prompt_manager.py`) to produce a set of candidate descriptions.
3.  **Assessment**: These candidates are evaluated by the `Quality Assessment Module` to assign quality scores.
4.  **Active Learning**: `Active Learning` strategies select the most critical, uncertain, or diverse descriptions for human review based on quality metrics.
5.  **Human Review**: Selected descriptions are presented to human annotators via the `HIL UI` (served by the `API`). Annotators review, edit, and validate the descriptions.
6.  **Feedback Loop**: Human feedback (approve, reject, edit) is captured by the `API`, which informs the active learning loop and refines subsequent generations.
7.  **Output**: Finally, the validated and refined paired linguistic supervision data is stored in `src/data/output/generated_supervision.json`, ready for downstream LLM training or fine-tuning.

## Getting Started

### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)
*   Access to an LLM API (e.g., OpenAI API key) or a locally running LLM.
*   (Optional) Access to external knowledge bases (e.g., Wikidata API keys if needed for specific integrations).

### Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-org/grounded-token-data-generator.git
    cd grounded-token-data-generator
    ```

2.  **Create a virtual environment and activate it:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure environment variables:**
    Create a `.env` file in the project root and add your sensitive credentials and configurations:
    ```
    # .env example
    OPENAI_API_KEY="sk-your_openai_api_key_here"
    KNOWLEDGE_BASE_API_KEY="your_kb_api_key_if_any"
    LLM_PROVIDER="openai" # or "huggingface", "local"
    HUGGINGFACE_MODEL_PATH="path/to/local/model" # if LLM_PROVIDER is huggingface/local
    DATABASE_URL="sqlite:///./data/app.db" # for the API/HIL
    ```
    Refer to `config.py` for all configurable parameters.

### Running the Prototype

The prototype involves several steps:

1.  **Prepare Input Tokens**: Place your new tokens in `src/data/input/new_tokens.csv`. This file should have at least a `token` column and can include an optional `context` or `entity_type` column.

    Example `new_tokens.csv`:
    ```csv
    token,context
    QuantFusion,"a novel algorithm combining quantum annealing and deep learning for predictive analytics"
    EcoNexus, "a startup specializing in sustainable urban farming technologies"
    HyperSense,"a new generation of IoT sensors with enhanced predictive capabilities"
    ```

2.  **Start the API Server**:
    ```bash
    python -m src.api.main
    ```
    This will typically run on `http://127.0.0.1:8000` (if using FastAPI) or `http://127.0.0.1:5000` (if using Flask).

3.  **Trigger Generation (via API or a script)**:
    You would typically have a script or an API endpoint to initiate the generation process for new tokens. For a direct run, you might call the generator explicitly:
    ```bash
    python -m src.run_generation
    ```
    (Note: `src/run_generation.py` is an example script you might create for batch processing.)

4.  **Access the HIL UI**:
    Open your web browser and navigate to the UI endpoint (e.g., `http://127.0.0.1:8000/ui` or similar, depending on your API implementation). Here, you can review, edit, and approve the generated descriptions.

5.  **Review and Validate**:
    *   The UI will display candidate descriptions, possibly prioritized by the active learning module.
    *   For each token, review its generated descriptions.
    *   Edit descriptions for accuracy, clarity, or improved grounding.
    *   Approve or reject descriptions. Approved descriptions will be saved to `src/data/output/generated_supervision.json`.

## Project Structure

```
.
├── .env                  # Environment variables for sensitive data
├── config.py             # Centralized system configurations
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── src/
│   ├── active_learning/
│   │   └── strategy.py   # Implementations of active learning strategies
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py       # Entry point for the API server (e.g., FastAPI/Flask app)
│   │   └── routes.py     # API endpoints for tokens, generation, validation
│   ├── core/
│   │   ├── __init__.py
│   │   ├── generator.py  # Orchestrates description generation
│   │   ├── knowledge_integrator.py # Unified interface for knowledge sources
│   │   ├── llm_interface.py # Abstracted LLM API calls
│   │   ├── prompt_manager.py # Manages LLM prompts
│   │   └── quality_metrics.py # Computes quality scores for descriptions
│   ├── data/
│   │   ├── input/
│   │   │   └── new_tokens.csv # Input file for new tokens
│   │   ├── output/
│   │   │   └── generated_supervision.json # Output for validated data
│   │   └── database.db # Placeholder for SQLite DB if used by API/HIL
│   ├── ui/
│   │   ├── index.html    # Main HIL user interface
│   │   ├── static/       # CSS, JS, images for the UI
│   │   └── ...
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_loaders.py # Helper functions for loading data
│   │   └── text_processors.py # Text preprocessing utilities
│   └── run_generation.py # Example script to trigger generation process
```

## Contributing

We welcome contributions to the Grounded Token Data Generator! Please see `CONTRIBUTING.md` (to be created) for guidelines on how to submit issues, propose features, and contribute code.

## License

This project is licensed under the MIT License - see the `LICENSE` (to be created) file for details.