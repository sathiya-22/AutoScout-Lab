```python
import os
import google.generativeai as genai
from config import Settings
import logging

# Configure logging for better user feedback
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    """
    Main function to run the Task Decomposition and Planning Agent.
    """
    settings = Settings()

    if not settings.api_key:
        logging.error("GEMINI_API_KEY environment variable not set. Please set it to proceed.")
        return

    # Configure the Google Generative AI client
    genai.configure(api_key=settings.api_key)

    # Initialize the Generative Model with specified settings
    try:
        model = genai.GenerativeModel(
            model_name=settings.model_name,
            generation_config={
                "temperature": settings.temperature,
                "max_output_tokens": settings.max_tokens,
            }
        )
    except Exception as e:
        logging.error(f"Failed to initialize the model: {e}. Check model name or API key.")
        return

    high_level_task = input("Enter a high-level task for the agent to decompose and plan: ")
    if not high_level_task.strip():
        logging.warning("Task cannot be empty. Exiting.")
        return

    logging.info(f"\n--- Decomposing Task: '{high_level_task}' ---")

    # Prompt for task decomposition
    decomposition_prompt = (
        "You are a task decomposition expert. Break down the following high-level goal into a list of concise, "
        "actionable, and sequential steps. Each step should be a distinct, manageable sub-task. "
        "Output only the numbered list of steps, without any additional text or explanations.\n\n"
        f"Goal: {high_level_task}"
    )

    try:
        # Generate decomposition
        decomposition_response = model.generate_content(decomposition_prompt)
        raw_steps = decomposition_response.text.strip()
        
        # Parse steps, handling various numbering/list formats
        steps = [step.strip() for step in raw_steps.split('\n') if step.strip()]
        steps = [step.split('.', 1)[1].strip() if '.' in step and step.split('.', 1)[0].isdigit() else step for step in steps]


        if not steps:
            logging.warning("No steps were generated for the task. Please try a different task or adjust model parameters.")
            return

        logging.info("\n--- Generated Steps ---")
        for i, step in enumerate(steps):
            print(f"Step {i+1}: {step}")

        logging.info("\n--- Generating Plans for Each Step ---")
        all_plans = []
        for i, step in enumerate(steps):
            logging.info(f"\n--- Planning for Step {i+1}: '{step}' ---")
            # Prompt for detailed planning for each step
            planning_prompt = (
                "You are a meticulous planner. For the following step from a larger task, "
                "generate a detailed, step-by-step plan to accomplish it. "
                "Focus on practical actions and considerations.\n\n"
                f"Task Step: {step}\n\n"
                "Plan:"
            )
            plan_response = model.generate_content(planning_prompt)
            plan_text = plan_response.text.strip()
            all_plans.append({"step": step, "plan": plan_text})
            print(f"\nPlan for Step {i+1} ('{step}'):\n{plan_text}\n")

        logging.info("\n--- Task Decomposition and Planning Complete ---")

    except genai.types.BlockedPromptException as e:
        logging.error(f"Prompt was blocked due to safety concerns: {e}")
    except genai.types.StopCandidateException as e:
        logging.error(f"Model stopped generating content prematurely: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during generation: {e}")

if __name__ == "__main__":
    main()
```
