```python
import os
import pandas as pd
import json
from datetime import datetime

# Import core components and utilities
from config import settings
from src.core.knowledge_integrator import KnowledgeIntegrator
from src.core.llm_interface import LLMInterface
from src.core.prompt_manager import PromptManager
from src.core.generator import Generator
from src.core.quality_metrics import QualityMetrics
from src.active_learning.strategy import ActiveLearningStrategy
from src.utils.data_loaders import load_tokens_from_csv
# from src.utils.text_processors import preprocess_text # Example utility, not directly orchestrated in main

def setup_directories():
    """Ensures the output directory exists."""
    output_dir = os.path.dirname(settings.OUTPUT_SUPERVISION_PATH)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except OSError as e:
            print(f"Error creating output directory {output_dir}: {e}")
            raise

def main():
    """
    Orchestrates the Grounded Token Data Generator pipeline:
    1. Loads new tokens from a CSV.
    2. Initializes core components (Knowledge Integrator, LLM Interface, Prompt Manager, Generator, Quality Metrics, Active Learning Strategy).
    3. For each token:
        a. Generates candidate linguistic descriptions using the Generator.
        b. Evaluates the quality of these descriptions using Quality Metrics.
        c. Prioritizes descriptions for human review using Active Learning (optional for this main.py, but stored).
    4. Saves the generated (and scored) supervision data to a JSON file, ready for HIL.
    """
    print("--- Starting Grounded Token Data Generator Prototype ---")
    setup_directories()

    # 1. Initialize core components
    try:
        print("Initializing core components...")
        knowledge_integrator = KnowledgeIntegrator(
            wikipedia_base_url=settings.WIKIPEDIA_BASE_URL,
            wikidata_query_endpoint=settings.WIKIDATA_QUERY_ENDPOINT
        )
        llm_interface = LLMInterface(
            api_key=settings.OPENAI_API_KEY,
            model_name=settings.LLM_MODEL_NAME,
            base_url=settings.OPENAI_API_BASE_URL
        )
        prompt_manager = PromptManager(
            default_prompt_template=settings.DEFAULT_PROMPT_TEMPLATE,
            system_message_template=settings.SYSTEM_MESSAGE_TEMPLATE
        )
        generator = Generator(
            knowledge_integrator=knowledge_integrator,
            llm_interface=llm_interface,
            prompt_manager=prompt_manager
        )
        quality_metrics = QualityMetrics(
            llm_interface=llm_interface # Pass LLM for embedding-based metrics, etc.
        )
        active_learning_strategy = ActiveLearningStrategy() # Initialized, but its full loop is external to main.py

        print("Components initialized successfully.")
    except Exception as e:
        print(f"ERROR: Failed to initialize core components: {e}")
        return

    # 2. Load new tokens
    print(f"Loading new tokens from {settings.INPUT_TOKENS_PATH}...")
    try:
        new_tokens_df = load_tokens_from_csv(settings.INPUT_TOKENS_PATH)
        if new_tokens_df.empty:
            print("No new tokens found in the input file. Exiting.")
            return
        # Ensure 'token' column exists
        if 'token' not in new_tokens_df.columns:
            raise ValueError("Input CSV must contain a 'token' column.")

        print(f"Successfully loaded {len(new_tokens_df)} new tokens.")
    except FileNotFoundError:
        print(f"ERROR: Input file not found at {settings.INPUT_TOKENS_PATH}")
        return
    except Exception as e:
        print(f"ERROR: Failed to load tokens from CSV: {e}")
        return

    generated_supervision_data = []

    # 3. Process each token
    print("\n--- Starting token processing ---")
    for index, row in new_tokens_df.iterrows():
        token_id = row.get('id', f"token_{index}") # Use 'id' column or generate a unique one
        token_text = row['token'].strip()
        initial_context = row.get('context', '').strip() # Optional context from CSV

        if not token_text:
            print(f"WARNING: Skipping row {index} due to empty token text.")
            continue

        print(f"\nProcessing token: '{token_text}' (ID: {token_id})")

        try:
            # 3a. Generate candidate descriptions
            print("  Generating candidate descriptions...")
            candidate_descriptions = generator.generate_descriptions(
                token=token_text,
                initial_context=initial_context,
                num_candidates=settings.NUM_GENERATION_CANDIDATES
            )

            if not candidate_descriptions:
                print(f"  No descriptions generated for '{token_text}'. Skipping to next token.")
                token_entry = {
                    'token_id': token_id,
                    'token_text': token_text,
                    'initial_context': initial_context,
                    'generation_timestamp': datetime.now().isoformat(),
                    'error': "No descriptions could be generated.",
                    'candidates': [],
                    'final_description': None,
                    'validation_status': 'failed_generation'
                }
                generated_supervision_data.append(token_entry)
                continue

            # 3b. Evaluate quality of candidates
            print("  Evaluating candidate descriptions...")
            scored_candidates = []
            for desc_text in candidate_descriptions:
                scores = quality_metrics.evaluate_description(
                    token=token_text,
                    description=desc_text,
                    # Potentially pass additional context or existing knowledge from KB here
                    existing_knowledge=knowledge_integrator.get_knowledge(token_text)
                )
                scored_candidates.append({
                    'description': desc_text,
                    'scores': scores,
                    'status': 'generated' # Initial status before human review
                })

            # 3c. Prioritize descriptions for human review (pre-sorting for HIL)
            # This step primarily informs how `candidates` are ordered for display in the UI.
            prioritized_candidates = active_learning_strategy.prioritize_for_review(scored_candidates)
            # The full active learning loop (feedback and re-prioritization) happens within the API/UI.

            # Prepare data for output, ready for HIL review
            token_entry = {
                'token_id': token_id,
                'token_text': token_text,
                'initial_context': initial_context,
                'generation_timestamp': datetime.now().isoformat(),
                'candidates': prioritized_candidates, # Store all scored and prioritized candidates
                'final_description': None, # Placeholder for human-validated description
                'validation_status': 'pending_review' # Status for the HIL interface
            }
            generated_supervision_data.append(token_entry)

        except Exception as e:
            print(f"  ERROR: Failed to process token '{token_text}': {e}")
            # Log the error and continue with the next token
            generated_supervision_data.append({
                'token_id': token_id,
                'token_text': token_text,
                'initial_context': initial_context,
                'generation_timestamp': datetime.now().isoformat(),
                'error': str(e),
                'candidates': [],
                'final_description': None,
                'validation_status': 'failed_processing'
            })

    # 4. Save generated (and scored) supervision data
    print(f"\n--- Saving generated supervision data to {settings.OUTPUT_SUPERVISION_PATH} ---")
    try:
        with open(settings.OUTPUT_SUPERVISION_PATH, 'w', encoding='utf-8') as f:
            json.dump(generated_supervision_data, f, indent=4, ensure_ascii=False)
        print("Generation complete. Data saved successfully for human-in-the-loop validation.")
    except Exception as e:
        print(f"ERROR: Failed to save generated data: {e}")

    print("\n--- Grounded Token Data Generator Finished ---")

if __name__ == "__main__":
    main()
```