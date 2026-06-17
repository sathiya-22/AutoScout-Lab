```python
import os
import google.generativeai as genai
from config import Settings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MultiHopAgent:
    """
    A Multi-Hop Question Answering Agent that uses a Generative AI model
    to break down complex questions, answer sub-parts, and synthesize a final answer.
    """
    def __init__(self, settings: Settings):
        self.settings = settings
        genai.configure(api_key=self.settings.api_key)
        self.model = genai.GenerativeModel(
            model_name=self.settings.model_name,
            generation_config=genai.GenerationConfig(
                temperature=self.settings.temperature,
                max_output_tokens=self.settings.max_tokens,
            )
        )
        logging.info(f"Agent initialized with model: {self.settings.model_name}")

    def _generate_response(self, prompt: str) -> str:
        """Helper to generate content from the model."""
        try:
            response = self.model.generate_content(prompt)
            if response.candidates and response.candidates[0].content.parts:
                return response.candidates[0].content.parts[0].text
            else:
                logging.warning("No content found in model response.")
                return "No answer generated."
        except Exception as e:
            logging.error(f"Error generating content: {e}")
            return f"Error: {e}"

    def answer_question(self, question: str) -> str:
        """
        Answers a multi-hop question by guiding the model through a step-by-step process.
        """
        logging.info(f"\n--- Processing Question ---")
        logging.info(f"Question: {question}")

        multi_hop_prompt = f"""
        You are a multi-hop question answering agent. Your task is to answer complex questions
        by first breaking them down into logical, sequential steps. For each step, provide
        a clear answer. Finally, synthesize all the step-by-step answers into a single, concise,
        and comprehensive final answer.

        Question: "{question}"

        Please follow this format:
        ---
        Steps:
        1. [Step 1 description]
        2. [Step 2 description]
        ...

        Step-by-step Answers:
        Step 1 Answer: [Answer to Step 1]
        Step 2 Answer: [Answer to Step 2]
        ...

        Final Answer: [Synthesized final answer]
        ---
        """
        logging.info("Sending multi-hop prompt to the model...")
        full_response = self._generate_response(multi_hop_prompt)
        logging.info("\n--- Model Response ---")
        logging.info(full_response)
        logging.info("--- End of Response ---")
        return full_response

if __name__ == "__main__":
    try:
        settings = Settings()
        agent = MultiHopAgent(settings)

        # Example multi-hop question
        complex_question = (
            "Who wrote '1984', what year was it published, and what is the author's real name?"
        )
        agent.answer_question(complex_question)

        complex_question_2 = (
            "What is the capital of France, and which river flows through it? "
            "Also, name a famous landmark located near that river in the capital."
        )
        agent.answer_question(complex_question_2)

    except Exception as e:
        logging.critical(f"An error occurred during execution: {e}")
        if "GEMINI_API_KEY" not in os.environ:
            logging.critical("Please ensure GEMINI_API_KEY environment variable is set.")

```
