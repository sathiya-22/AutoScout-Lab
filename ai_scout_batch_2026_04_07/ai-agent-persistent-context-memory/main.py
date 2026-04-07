import os
import sys
from dotenv import load_dotenv

# Add project root to sys.path to enable absolute imports from subdirectories
# This assumes main.py is in the project root.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# Database path for SQLAlchemy (SQLite for prototyping)
DB_PATH = os.getenv("DATABASE_PATH", "sqlite:///./agent_memory.db")
# Vector store path (e.g., for local file-based vector stores like ChromaDB, FAISS)
VECTOR_DB_PATH = os.getenv("VECTOR_DATABASE_PATH", "./agent_vector_store")
# Directory where crystallized skills will be stored
SKILLS_DIR = os.getenv("SKILLS_DIRECTORY", os.path.join(PROJECT_ROOT, "skills", "generated_skills"))
# Directory where prompt templates are located
PROMPTS_DIR = os.getenv("PROMPTS_DIRECTORY", os.path.join(PROJECT_ROOT, "prompts"))
# LLM API Key (e.g., OPENAI_API_KEY, ANTHROPIC_API_KEY)
LLM_API_KEY = os.getenv("OPENAI_API_KEY") # Adjust based on your LLM provider

# Ensure necessary directories exist for skills and prompts
os.makedirs(SKILLS_DIR, exist_ok=True)
os.makedirs(PROMPTS_DIR, exist_ok=True)

# Helper function to create dummy prompt files if they don't exist
# This ensures the prototype can run without manual prompt file creation.
def create_dummy_prompt_file(filepath, content):
    if not os.path.exists(filepath):
        try:
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"Created dummy prompt file: {filepath}")
        except IOError as e:
            print(f"Error creating dummy prompt file {filepath}: {e}", file=sys.stderr)

# Define paths for specific prompt files
CHECKPOINT_PROMPT_PATH = os.path.join(PROMPTS_DIR, "checkpoint_prompt.txt")
CRYSTALLIZATION_PROMPT_PATH = os.path.join(PROMPTS_DIR, "crystallization_prompt.txt")

# Create dummy content for prompt files
create_dummy_prompt_file(CHECKPOINT_PROMPT_PATH, "Summarize the agent's current state, understanding, and proposed next steps for human review, focusing on key decisions and progress towards the main task.")
create_dummy_prompt_file(CRYSTALLIZATION_PROMPT_PATH, "Analyze the following interaction log/context. Identify any recurring patterns, successful problem-solving strategies, or generalizable insights that could be formalized into a reusable skill. Output the skill in a concise, executable format (e.g., a function, a specialized prompt template, or a codified rule).")


# --- Import Modules ---
# Centralized error handling for module imports to provide clearer startup diagnostics
try:
    from storage import database, models
    from storage.vector_store_interface import VectorStoreInterface
    from utils.llm_api import LLMApi

    from agent.state_manager import StateManager
    from agent.memory_subsystem import MemorySubsystem
    from agent.checkpoint_manager import CheckpointManager
    from agent.knowledge_crystallizer import KnowledgeCrystallizer
    from agent.context_optimizer import ContextOptimizer
    from agent.core_agent import CoreAgent

    from context.semantic_context_retriever import SemanticContextRetriever
    from context.changelog_processor import ChangelogProcessor
    from context.tree_context_model import TreeContextModel
    from context.context_aggregator import ContextAggregator
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import a required module: {e}", file=sys.stderr)
    print("Please ensure all project files are correctly structured and available in their respective directories.", file=sys.stderr)
    sys.exit(1)


def main():
    print("--- Initializing AI Agent Prototype ---")

    # 1. Initialize Persistence Layer
    engine = None
    db_session = None
    try:
        engine, SessionLocal = database.init_db(DB_PATH)
        models.Base.metadata.create_all(bind=engine) # Create tables if they don't exist
        db_session = SessionLocal()
        print(f"Database initialized and connected at {DB_PATH}")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to initialize database: {e}", file=sys.stderr)
        sys.exit(1)

    vector_store = None
    try:
        vector_store = VectorStoreInterface(VECTOR_DB_PATH)
        print(f"Vector store interface initialized (using path: {VECTOR_DB_PATH})")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to initialize vector store: {e}", file=sys.stderr)
        if db_session:
            db_session.close()
        sys.exit(1)

    # 2. Initialize LLM Interface
    llm_api = None
    if not LLM_API_KEY:
        print("CRITICAL ERROR: LLM_API_KEY not found in environment variables.", file=sys.stderr)
        print("Please set OPENAI_API_KEY or your preferred LLM provider's API key.", file=sys.stderr)
        if db_session:
            db_session.close()
        sys.exit(1)

    try:
        llm_api = LLMApi(LLM_API_KEY)
        print("LLM API initialized.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to initialize LLM API: {e}", file=sys.stderr)
        if db_session:
            db_session.close()
        sys.exit(1)

    # 3. Initialize Core Agent Components with their dependencies
    try:
        state_manager = StateManager(db_session)
        print("State Manager initialized.")

        # Context Aggregation components
        semantic_retriever = SemanticContextRetriever(vector_store, llm_api)
        changelog_processor = ChangelogProcessor()
        tree_context_model = TreeContextModel()
        print("Context Sub-components (Semantic Retriever, Changelog Processor, Tree Context Model) initialized.")

        context_aggregator = ContextAggregator(
            semantic_retriever=semantic_retriever,
            changelog_processor=changelog_processor,
            tree_context_model=tree_context_model,
            llm_api=llm_api
        )
        print("Context Aggregator initialized.")

        memory_subsystem = MemorySubsystem(
            state_manager=state_manager,
            context_aggregator=context_aggregator,
            skills_dir=SKILLS_DIR
        )
        print("Memory Subsystem initialized.")

        checkpoint_manager = CheckpointManager(
            state_manager=state_manager,
            memory_subsystem=memory_subsystem,
            llm_api=llm_api,
            prompt_filepath=CHECKPOINT_PROMPT_PATH
        )
        print("Checkpoint Manager initialized.")

        knowledge_crystallizer = KnowledgeCrystallizer(
            llm_api=llm_api,
            prompt_filepath=CRYSTALLIZATION_PROMPT_PATH,
            skills_dir=SKILLS_DIR
        )
        print("Knowledge Crystallizer initialized.")

        context_optimizer = ContextOptimizer(
            llm_api=llm_api,
            context_aggregator=context_aggregator
        )
        print("Context Optimizer initialized.")

        # Core Agent orchestrator
        agent = CoreAgent(
            state_manager=state_manager,
            memory_subsystem=memory_subsystem,
            context_aggregator=context_aggregator,
            checkpoint_manager=checkpoint_manager,
            knowledge_crystallizer=knowledge_crystallizer,
            context_optimizer=context_optimizer,
            llm_api=llm_api
        )
        print("Core Agent initialized successfully.")

    except Exception as e:
        print(f"CRITICAL ERROR: Failed to initialize one or more agent components: {e}", file=sys.stderr)
        if db_session:
            db_session.close()
        sys.exit(1)

    print("\n--- Starting Agent Session Example ---")

    # Example Agent Workflow Simulation
    agent_id = "prototype_agent_001"
    initial_task = "Develop a simple web application using Python Flask for managing user tasks. Focus on creating an API for CRUD operations."
    print(f"Agent '{agent_id}' starting with task: '{initial_task}'")

    try:
        # Load or create agent state for the session
        current_agent_state = state_manager.get_agent_state(agent_id)
        if current_agent_state is None:
            # If no state exists, create a new one.
            state_manager.create_agent_state(agent_id, {"task_goal": initial_task, "progress": "started", "turn": 0})
            current_agent_state = state_manager.get_agent_state(agent_id)
            print(f"Created new agent state for {agent_id}.")
        else:
            # If state exists, resume.
            print(f"Loaded existing agent state for {agent_id}. Current task: {current_agent_state.data.get('task_goal', 'N/A')}")
            print(f"Current progress: {current_agent_state.data.get('progress', 'N/A')}")

        # Simulate a few turns of interaction/work
        num_simulated_turns = 3
        for i in range(1, num_simulated_turns + 1):
            print(f"\n--- Agent Turn {i} ---")
            current_task_goal = current_agent_state.data.get("task_goal", initial_task)
            user_input = f"User: For the task '{current_task_goal}', in turn {i}, provide an update on the progress and next immediate steps. Specifically, how would you set up the Flask project structure?"
            print(f"User Input: '{user_input}'")

            # Agent processes input - this method orchestrates all subsystems
            agent_response = agent.process_input(agent_id, user_input)
            print(f"Agent's Simulated Response: '{agent_response}'")

            # Update agent state to reflect progress after the turn
            current_state_data = current_agent_state.data
            current_state_data["turn"] = i
            current_state_data["progress"] = f"Completed turn {i}, processed user input regarding project structure."
            state_manager.update_agent_state(agent_id, current_state_data)
            current_agent_state = state_manager.get_agent_state(agent_id) # Refresh state object
            print(f"Agent state updated. Progress: '{current_agent_state.data['progress']}'")

            # Trigger a human checkpoint at critical junctures (e.g., after every 2 turns)
            if i % 2 == 0:
                print("\n--- Triggering Human Checkpoint ---")
                try:
                    summary = checkpoint_manager.generate_summary(agent_id, current_agent_state.data)
                    print(f"Agent's understanding for human review:\n{summary}")
                    # Simulate human feedback
                    confirmation = input("Human confirmation (y/n)? ").lower().strip()
                    if confirmation == 'y':
                        checkpoint_manager.save_checkpoint(agent_id, summary, confirmed=True)
                        print("Checkpoint confirmed and saved.")
                    else:
                        checkpoint_manager.save_checkpoint(agent_id, summary, confirmed=False)
                        print("Checkpoint NOT confirmed. Agent might need to re-evaluate.")
                        # In a real system, unconfirmed checkpoints could trigger re-planning or clarification
                except Exception as e:
                    print(f"Error during checkpoint generation/saving: {e}", file=sys.stderr)

        # Demonstrate Knowledge Crystallization (triggered manually for this prototype)
        print("\n--- Attempting Knowledge Crystallization ---")
        simulated_context_for_crystallization = "During the Flask project setup, the agent successfully identified and implemented a robust directory structure and a standard `app.py` boilerplate. This pattern was applied efficiently across similar hypothetical scenarios."
        try:
            skill_name = "flask_project_init"
            skill_description = "A reusable skill for initializing a standard Flask project structure and basic boilerplate code."
            # The crystallizer uses the LLM to formalize the skill from the given context
            crystallized_skill_path = knowledge_crystallizer.crystallize_knowledge(
                agent_id=agent_id,
                context_for_crystallization=simulated_context_for_crystallization,
                skill_name=skill_name,
                skill_description=skill_description
            )
            print(f"Knowledge crystallized and saved as skill: {skill_name} at {crystallized_skill_path}")
        except Exception as e:
            print(f"Knowledge crystallization failed: {e}", file=sys.stderr)

        # Demonstrate Context Optimization
        print("\n--- Optimizing Context for Next Steps ---")
        current_state_data = current_agent_state.data # Use the latest state
        try:
            # The optimizer would analyze current state, memory, and task to fine-tune context
            optimized_context_summary = context_optimizer.optimize_context(agent_id, current_state_data)
            print(f"Optimized context summary (first 200 chars):\n'{optimized_context_summary[:200]}...'")
            print("Context optimization completed.")
        except Exception as e:
            print(f"Context optimization failed: {e}", file=sys.stderr)

    except Exception as e:
        print(f"An unexpected error occurred during the agent session: {e}", file=sys.stderr)
    finally:
        print("\n--- Agent Session Finished ---")
        if db_session:
            db_session.close()
            print("Database session closed.")


if __name__ == "__main__":
    main()