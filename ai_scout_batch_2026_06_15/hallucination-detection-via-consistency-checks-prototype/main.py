import os
import google.generativeai as genai
from config import Config

def main():
    """
    Main function to run the hallucination detection prototype.
    It performs consistency checks on LLM responses for several queries.
    """
    config = Config()

    # Configure the Google Generative AI client
    try:
        genai.configure(api_key=config.api_key)
    except Exception as e:
        print(f"Error configuring Google GenAI. Ensure GEMINI_API_KEY is set correctly: {e}")
        return

    # Initialize the generative model
    model = genai.GenerativeModel(
        model_name=config.model_name,
        generation_config={
            "temperature": config.temperature,
            "max_output_tokens": config.max_tokens,
        }
    )

    queries = [
        "What is the capital of France?",
        "Who was the first person to walk on the moon, and in what year?",
        "Explain the theory of relativity in simple terms, focusing on time dilation.",
        "List three benefits of eating apples. (Ensure one benefit is completely fabricated)",
        "What is the boiling point of water at sea level in Celsius?",
    ]

    for i, original_query in enumerate(queries):
        print(f"\n--- Query {i+1}: {original_query} ---")

        # Step 1: Get initial response from the model
        try:
            initial_response_obj = model.generate_content(original_query)
            initial_response = initial_response_obj.text
            print(f"Initial Response:\n{initial_response}\n")
        except Exception as e:
            print(f"Error generating initial response: {e}")
            continue

        # Step 2: Formulate a consistency check prompt
        # Ask the model to rephrase or re-explain the core facts of its initial answer.
        consistency_prompt = (
            f"Based on your previous answer to the question '{original_query}', "
            f"can you rephrase the core facts or provide a brief summary of them? "
            f"Do not introduce new information not present in your first answer."
        )

        try:
            consistency_response_obj = model.generate_content(consistency_prompt)
            consistency_response = consistency_response_obj.text
            print(f"Consistency Check Response:\n{consistency_response}\n")
        except Exception as e:
            print(f"Error generating consistency check response: {e}")
            continue

        # Step 3: Ask the model to evaluate its own consistency
        evaluation_prompt = (
            f"You were asked: '{original_query}'\n"
            f"Your first answer was:\n'{initial_response}'\n"
            f"Your second answer (rephrased/summarized from the first) was:\n'{consistency_response}'\n\n"
            f"Are these two answers consistent with each other regarding the core facts? "
            f"Is there any factual discrepancy or contradiction? "
            f"State 'CONSISTENT' or 'POTENTIAL HALLUCINATION' and explain your reasoning concisely."
        )

        try:
            evaluation_response_obj = model.generate_content(evaluation_prompt)
            evaluation_result = evaluation_response_obj.text
            print(f"Consistency Evaluation:\n{evaluation_result}\n")
        except Exception as e:
            print(f"Error generating consistency evaluation: {e}")
            continue

if __name__ == "__main__":
    main()
