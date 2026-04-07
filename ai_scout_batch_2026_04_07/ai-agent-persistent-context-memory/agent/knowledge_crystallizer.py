import os
import json
import logging
from datetime import datetime

# Assume these modules exist at the specified paths
from utils.llm_api import LLMAPI
from prompts.prompt_loader import load_prompt_template # Assuming a utility to load prompts
# from storage.models import Skill # Optional: if we want to store skill metadata in DB

logger = logging.getLogger(__name__)

class KnowledgeCrystallizer:
    """
    Manages the process of identifying, formalizing, and storing generalizable insights
    from agent interactions into reusable 'skills'.
    """

    def __init__(self, llm_api: LLMAPI, prompts_dir: str = 'prompts', skills_base_dir: str = 'skills/generated_skills'):
        """
        Initializes the KnowledgeCrystallizer.

        Args:
            llm_api (LLMAPI): An instance of the LLMAPI for interacting with the LLM.
            prompts_dir (str): Directory where prompt templates are stored.
            skills_base_dir (str): Base directory where generated skills will be saved.
        """
        self.llm_api = llm_api
        self.prompts_dir = prompts_dir
        self.skills_base_dir = skills_base_dir
        os.makedirs(self.skills_base_dir, exist_ok=True)
        self.crystallization_prompt_path = os.path.join(self.prompts_dir, 'crystallization_prompt.txt')

    def _load_crystallization_prompt(self) -> str:
        """
        Loads the prompt template for knowledge crystallization.

        Returns:
            str: The loaded prompt template.

        Raises:
            FileNotFoundError: If the crystallization prompt file does not exist.
            IOError: For other issues during file reading.
        """
        try:
            # Assuming prompt_loader.py has a function like load_prompt_template
            # If not, a simple open() read would suffice.
            from prompts.prompt_loader import load_prompt_template
            return load_prompt_template('crystallization_prompt', self.prompts_dir)
        except FileNotFoundError:
            logger.error(f"Crystallization prompt file not found at: {self.crystallization_prompt_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading crystallization prompt: {e}")
            raise

    def _save_skill_to_file(self, skill_name: str, skill_content: str, file_extension: str = '.json') -> str:
        """
        Saves the generated skill content to a file in the skills directory.

        Args:
            skill_name (str): The name of the skill, used to create a filename.
            skill_content (str): The actual content of the skill (e.g., code, template).
            file_extension (str): The file extension for the skill file.

        Returns:
            str: The path to the saved skill file.

        Raises:
            IOError: If there's an error writing the file.
        """
        sanitized_skill_name = "".join(c for c in skill_name if c.isalnum() or c in (' ', '_')).rstrip()
        if not sanitized_skill_name:
            sanitized_skill_name = f"untitled_skill_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        filename = f"{sanitized_skill_name.replace(' ', '_').lower()}{file_extension}"
        filepath = os.path.join(self.skills_base_dir, filename)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(skill_content)
            logger.info(f"Skill '{skill_name}' saved to: {filepath}")
            return filepath
        except IOError as e:
            logger.error(f"Error saving skill '{skill_name}' to file {filepath}: {e}")
            raise

    def crystallize_knowledge(self, agent_insights: str, context: dict = None) -> (str | None):
        """
        Identifies recurring patterns or generalizable insights and prompts the LLM
        to formalize them into reusable skills.

        Args:
            agent_insights (str): A summary or raw data of agent's recent performance,
                                  interactions, or problem-solving steps that might
                                  contain crystallizable knowledge.
            context (dict, optional): Additional context to pass to the LLM (e.g., current task, goal).

        Returns:
            str | None: The path to the saved skill file if successful, otherwise None.
        """
        logger.info("Attempting to crystallize knowledge...")
        try:
            crystallization_prompt_template = self._load_crystallization_prompt()
            
            # Prepare the full prompt for the LLM
            # The crystallization_prompt.txt should guide the LLM to output a structured format,
            # e.g., JSON with "skill_name", "description", "skill_content", "skill_type".
            full_prompt = crystallization_prompt_template.format(
                agent_insights=agent_insights,
                additional_context=json.dumps(context) if context else "{}"
            )

            # Call the LLM to generate the skill
            messages = [
                {"role": "system", "content": "You are a knowledge crystallization engine. Your task is to identify and formalize reusable skills from agent insights."},
                {"role": "user", "content": full_prompt}
            ]
            
            llm_response = self.llm_api.generate_response(messages)

            if not llm_response:
                logger.warning("LLM returned an empty response for knowledge crystallization.")
                return None

            # Attempt to parse the LLM's structured output
            try:
                # Assuming the LLM is guided to output a JSON object within its response
                # We need to extract the JSON string first.
                start_idx = llm_response.find('{')
                end_idx = llm_response.rfind('}')
                
                if start_idx == -1 or end_idx == -1 or start_idx > end_idx:
                    raise ValueError("No valid JSON object found in LLM response.")

                json_str = llm_response[start_idx : end_idx + 1]
                crystallized_skill_data = json.loads(json_str)

                skill_name = crystallized_skill_data.get('skill_name')
                skill_content = crystallized_skill_data.get('skill_content')
                skill_type = crystallized_skill_data.get('skill_type', 'generic') # e.g., 'python_code', 'prompt_template', 'json_config'
                skill_description = crystallized_skill_data.get('description', 'No description provided.')

                if not skill_name or not skill_content:
                    raise ValueError("LLM response missing 'skill_name' or 'skill_content'.")

                logger.info(f"LLM successfully generated a skill: '{skill_name}' (Type: {skill_type})")

                # Determine file extension based on skill type
                file_extension = '.txt'
                if skill_type == 'python_code':
                    file_extension = '.py'
                elif skill_type == 'json_config':
                    file_extension = '.json'
                elif skill_type == 'prompt_template':
                    file_extension = '.txt' # Could be .prompt or similar if desired

                saved_path = self._save_skill_to_file(skill_name, skill_content, file_extension)

                # Optional: Persist skill metadata to database if a Skill model exists
                # try:
                #     # This part would require access to a DB session manager
                #     # and the Skill model from storage.models
                #     new_skill_metadata = Skill(
                #         name=skill_name,
                #         description=skill_description,
                #         type=skill_type,
                #         file_path=saved_path,
                #         created_at=datetime.utcnow()
                #     )
                #     # Assuming a session management setup:
                #     # db_session.add(new_skill_metadata)
                #     # db_session.commit()
                #     logger.info(f"Skill metadata for '{skill_name}' recorded in database.")
                # except Exception as db_err:
                #     logger.warning(f"Failed to save skill metadata to database: {db_err}")

                return saved_path

            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode LLM response as JSON: {e}. Response: {llm_response[:500]}...")
            except ValueError as e:
                logger.error(f"Invalid structured skill data from LLM: {e}. Response: {llm_response[:500]}...")
            except Exception as e:
                logger.error(f"Unexpected error processing LLM skill response: {e}")

        except FileNotFoundError:
            logger.error("Could not crystallize knowledge because the prompt template was not found.")
        except Exception as e:
            logger.error(f"An unexpected error occurred during knowledge crystallization: {e}")

        return None

# Placeholder for prompt_loader if it doesn't exist yet
# In a real setup, this would be in prompts/prompt_loader.py
if not os.path.exists('prompts'):
    os.makedirs('prompts')
    with open('prompts/crystallization_prompt.txt', 'w') as f:
        f.write("""
You are an expert AI assistant tasked with identifying recurring patterns, successful problem-solving strategies, or generalizable insights from agent interactions and formalizing them into reusable 'skills'.

Review the provided agent insights and identify any knowledge that could be abstracted into a standalone, reusable skill. This skill should be a specific instruction, code snippet, template, or configuration that helps the agent perform better in similar future situations.

Your output MUST be a JSON object with the following keys:
- "skill_name": A concise, descriptive name for the skill (e.g., "CodeRefactoring", "SummarizeDocument", "TaskDecomposition").
- "description": A brief explanation of what the skill does and when it should be used.
- "skill_content": The actual content of the skill. This could be Python code, a detailed prompt template, a configuration snippet, or specific instructions.
- "skill_type": Categorize the skill (e.g., "python_code", "prompt_template", "json_config", "instruction_set").

Agent Insights to analyze:
---
{agent_insights}
---

Additional context (if available):
{additional_context}

Example JSON output format:
```json
{{
  "skill_name": "EfficientFunctionOptimization",
  "description": "A strategy to optimize Python functions by identifying bottlenecks and applying common optimization techniques like memoization, generator expressions, or C-extensions.",
  "skill_content": "When a Python function is identified as a bottleneck, first profile it using cProfile. If I/O bound, consider async/await or thread pools. If CPU bound, explore Numba for numerical tasks, Cython for C-extensions, or carefully consider algorithm complexity improvements. For repetitive calculations, implement memoization.",
  "skill_type": "instruction_set"
}}
```
""")

class MockLLMAPI:
    """A mock LLMAPI for testing purposes."""
    def generate_response(self, messages):
        # Simulate a simple skill extraction based on input for testing
        last_user_message = messages[-1]['content']
        if "recurrent pattern: optimize database queries" in last_user_message:
            return json.dumps({
                "skill_name": "OptimizeDatabaseQueries",
                "description": "Identifies slow SQL queries and suggests indexing, query rewriting, or caching strategies.",
                "skill_content": "Always use 'EXPLAIN ANALYZE' for slow queries. Ensure foreign keys are indexed. Avoid N+1 queries by using JOINs or select_related/prefetch_related. Implement caching for frequently accessed static data.",
                "skill_type": "instruction_set"
            })
        elif "failed to debug a complex error" in last_user_message:
            return json.dumps({
                "skill_name": "ComplexErrorDebuggingStrategy",
                "description": "A systematic approach to debugging complex, multi-component errors.",
                "skill_content": "1. Isolate the component: Try to reproduce the error in a minimal environment. 2. Verify inputs/outputs: Check data flow between components. 3. Logs & Metrics: Scrutinize all relevant logs and monitoring metrics for anomalies. 4. Binary search: Comment out/disable parts of the system until the error disappears to pinpoint the source. 5. Consult documentation/community: Search for similar issues.",
                "skill_type": "instruction_set"
            })
        else:
            return json.dumps({
                "skill_name": "GenericInsight",
                "description": "A generic insight crystallized from unspecific patterns.",
                "skill_content": "Always break down large tasks into smaller, manageable sub-tasks. Prioritize sub-tasks based on dependencies and impact. Document decisions and assumptions clearly.",
                "skill_type": "instruction_set"
            })

# Assuming this helper function resides in prompts/prompt_loader.py
# Adding it here for standalone execution/testing during development of this file.
if 'load_prompt_template' not in globals():
    def load_prompt_template(template_name: str, base_dir: str = 'prompts') -> str:
        """Loads a prompt template from a specified directory."""
        file_path = os.path.join(base_dir, f"{template_name}.txt")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Prompt template '{template_name}.txt' not found at {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

# Example usage (for testing purposes during development)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    mock_llm_api = MockLLMAPI()
    crystallizer = KnowledgeCrystallizer(llm_api=mock_llm_api)

    print("\n--- Testing Knowledge Crystallization (Scenario 1: DB Optimization) ---")
    insights_1 = "Agent repeatedly encountered performance issues due to slow database queries. It learned to check indexes and optimize joins."
    skill_path_1 = crystallizer.crystallize_knowledge(insights_1)
    if skill_path_1:
        print(f"Crystallized skill saved to: {skill_path_1}")
        with open(skill_path_1, 'r') as f:
            print("Content:\n", f.read())

    print("\n--- Testing Knowledge Crystallization (Scenario 2: Debugging Strategy) ---")
    insights_2 = "Agent struggled to debug a complex, multi-component error across different services. It eventually found a systematic way to isolate issues."
    skill_path_2 = crystallizer.crystallize_knowledge(insights_2)
    if skill_path_2:
        print(f"Crystallized skill saved to: {skill_path_2}")
        with open(skill_path_2, 'r') as f:
            print("Content:\n", f.read())

    print("\n--- Testing Knowledge Crystallization (Scenario 3: Generic Insight) ---")
    insights_3 = "Agent completed several tasks without specific patterns emerging, but generally improved its task decomposition abilities."
    skill_path_3 = crystallizer.crystallize_knowledge(insights_3, context={"current_task_type": "general_coding"})
    if skill_path_3:
        print(f"Crystallized skill saved to: {skill_path_3}")
        with open(skill_path_3, 'r') as f:
            print("Content:\n", f.read())

    print("\n--- Testing Knowledge Crystallization (Scenario 4: LLM output malformed) ---")
    class MalformedLLMAPI(LLMAPI):
        def generate_response(self, messages):
            return "This is not a JSON { malformed_output } here."
    
    malformed_crystallizer = KnowledgeCrystallizer(llm_api=MalformedLLMAPI())
    skill_path_4 = malformed_crystallizer.crystallize_knowledge("Some insights, expecting malformed output.")
    if not skill_path_4:
        print("Successfully handled malformed LLM response.")
    else:
        print(f"ERROR: Skill was created for malformed response: {skill_path_4}")

    print("\n--- Testing Knowledge Crystallization (Scenario 5: Empty LLM output) ---")
    class EmptyLLMAPI(LLMAPI):
        def generate_response(self, messages):
            return ""
    
    empty_crystallizer = KnowledgeCrystallizer(llm_api=EmptyLLMAPI())
    skill_path_5 = empty_crystallizer.crystallize_knowledge("Some insights, expecting empty output.")
    if not skill_path_5:
        print("Successfully handled empty LLM response.")
    else:
        print(f"ERROR: Skill was created for empty response: {skill_path_5}")