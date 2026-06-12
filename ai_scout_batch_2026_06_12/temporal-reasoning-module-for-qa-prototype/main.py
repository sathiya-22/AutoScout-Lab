import os
import google.generativeai as genai
from config import Config

def temporal_reasoning_qa(question: str, context: str = "") -> str:
    """
    Analyzes a question and optional context to provide a temporally accurate answer
    using a generative AI model configured for temporal reasoning.
    """
    cfg = Config()

    if not cfg.api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set or loaded.")

    genai.configure(api_key=cfg.api_key)
    model = genai.GenerativeModel(model_name=cfg.model_name)

    # Craft a detailed prompt for temporal reasoning
    prompt_parts = [
        "You are an expert temporal reasoning module for a Question Answering system.",
        "Your primary task is to analyze questions and provided context to extract, infer, or calculate temporal information.",
        "Focus on dates, times, durations, sequences, and relationships like 'before', 'after', 'during', 'simultaneously'.",
        "If the question involves a timeline, duration, or sequence, break down the temporal logic and provide a clear, concise answer.",
        "---",
        f"Context: {context if context else 'No specific context provided.'}",
        f"Question: {question}",
        "---",
        "Based on the above, provide a precise answer that addresses the temporal aspects of the question.",
        "If the question does not involve temporal reasoning, or if information is insufficient to answer temporally, state that clearly."
    ]
    prompt = "\n".join(prompt_parts)

    generation_config = genai.types.GenerationConfig(
        temperature=cfg.temperature,
        max_output_tokens=cfg.max_tokens,
    )

    try:
        response = model.generate_content(prompt, generation_config=generation_config)
        return response.text.strip()
    except Exception as e:
        return f"An error occurred during AI generation: {e}"

if __name__ == "__main__":
    print("Initializing Temporal Reasoning Module Demo...\n")

    # Example 1: Basic temporal event recall
    q1 = "When did the first human land on the Moon?"
    print(f"Question: {q1}")
    ans1 = temporal_reasoning_qa(q1)
    print(f"Answer: {ans1}\n")

    # Example 2: Sequence and duration calculation
    q2 = "If a meeting started at 10:00 AM and lasted 1 hour and 30 minutes, and a follow-up discussion began 15 minutes after the meeting ended, when did the follow-up discussion start?"
    print(f"Question: {q2}")
    ans2 = temporal_reasoning_qa(q2)
    print(f"Answer: {ans2}\n")

    # Example 3: Contextual temporal ordering
    context3 = "The construction of the Eiffel Tower began on January 28, 1887. It was completed on March 31, 1889. The Universal Exposition opened on May 6, 1889."
    q3 = "What happened between the completion of the Eiffel Tower and the opening of the Universal Exposition?"
    print(f"Question: {q3}")
    ans3 = temporal_reasoning_qa(q3, context=context3)
    print(f"Answer: {ans3}\n")

    # Example 4: Duration calculation from specific dates
    q4 = "How many days passed between January 1, 2023 and January 31, 2023?"
    print(f"Question: {q4}")
    ans4 = temporal_reasoning_qa(q4)
    print(f"Answer: {ans4}\n")

    # Example 5: Non-temporal question (should be identified as such)
    q5 = "What is the capital of France?"
    print(f"Question: {q5}")
    ans5 = temporal_reasoning_qa(q5)
    print(f"Answer: {ans5}\n")
