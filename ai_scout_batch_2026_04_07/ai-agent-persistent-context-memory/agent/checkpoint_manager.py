import json
import os
from datetime import datetime

# Assume these modules exist as per architecture notes
from utils.llm_api import LLMAPI
from storage.database import SessionLocal
from storage.models import Checkpoint, AgentState  # Assuming AgentState can be snapshotted
# from agent.state_manager import StateManager # Not directly used here, but implies data source
# from context.context_aggregator import ContextAggregator # Not directly used here, but implies data source


class CheckpointManager:
    """
    Manages the creation, human review, and persistence of agent checkpoints.
    These checkpoints serve as verifiable anchors for context and progress,
    mitigating catastrophic forgetting by incorporating human-in-the-loop validation.
    """

    def __init__(self, llm_api: LLMAPI, prompts_dir: str = "prompts"):
        """
        Initializes the CheckpointManager.

        Args:
            llm_api (LLMAPI): An instance of the LLMAPI for generating summaries.
            prompts_dir (str): Directory where prompt templates are stored.
        """
        self.llm_api = llm_api
        self.checkpoint_prompt_path = os.path.join(prompts_dir, "checkpoint_prompt.txt")
        self._load_checkpoint_prompt()

    def _load_checkpoint_prompt(self):
        """Loads the checkpoint generation prompt from the specified file."""
        try:
            with open(self.checkpoint_prompt_path, 'r', encoding='utf-8') as f:
                self.checkpoint_prompt_template = f.read()
            if not self.checkpoint_prompt_template:
                raise ValueError("Checkpoint prompt template is empty.")
        except FileNotFoundError:
            self.checkpoint_prompt_template = (
                "You are an AI assistant tasked with summarizing your current understanding, "
                "state, and proposed next steps for a human operator. "
                "Provide a concise summary based on the following information:\n\n"
                "Agent State: {agent_state_summary}\n"
                "Current Context: {current_context_summary}\n"
                "Proposed Actions: {proposed_actions_summary}\n\n"
                "Summary:"
            )
            print(f"Warning: Checkpoint prompt file not found at '{self.checkpoint_prompt_path}'. "
                  f"Using default template.")
        except Exception as e:
            self.checkpoint_prompt_template = (
                "You are an AI assistant tasked with summarizing your current understanding, "
                "state, and proposed next steps for a human operator. "
                "Provide a concise summary based on the following information:\n\n"
                "Agent State: {agent_state_summary}\n"
                "Current Context: {current_context_summary}\n"
                "Proposed Actions: {proposed_actions_summary}\n\n"
                "Summary:"
            )
            print(f"Error loading checkpoint prompt: {e}. Using default template.")

    def generate_summary(self, agent_state: dict, current_context: dict, proposed_actions: list) -> str:
        """
        Generates a concise summary of the agent's current understanding, state,
        and proposed next steps using the LLM.

        Args:
            agent_state (dict): A snapshot of the agent's current state.
            current_context (dict): The aggregated relevant context for the current operation.
            proposed_actions (list): A list of actions the agent proposes to take next.

        Returns:
            str: The generated summary text.
        """
        try:
            # Prepare a simplified string representation of inputs for the LLM
            agent_state_summary = json.dumps(agent_state, indent=2)
            current_context_summary = json.dumps(current_context, indent=2)
            proposed_actions_summary = json.dumps(proposed_actions, indent=2)

            # Format the prompt with dynamic content
            formatted_prompt = self.checkpoint_prompt_template.format(
                agent_state_summary=agent_state_summary,
                current_context_summary=current_context_summary,
                proposed_actions_summary=proposed_actions_summary
            )

            # Call the LLM to generate the summary
            response = self.llm_api.generate_response(
                messages=[{"role": "user", "content": formatted_prompt}],
                max_tokens=500  # A reasonable limit for a summary
            )
            return response.strip()
        except Exception as e:
            print(f"Error generating checkpoint summary: {e}")
            return (f"Failed to generate summary due to an internal error. "
                    f"Agent State: {json.dumps(agent_state)}. "
                    f"Context: {json.dumps(current_context)}. "
                    f"Proposed: {json.dumps(proposed_actions)}")

    def request_human_confirmation(self, summary: str) -> bool:
        """
        Presents the summary to a human and requests confirmation or correction.
        For prototyping, this uses standard input/output.

        Args:
            summary (str): The generated summary text.

        Returns:
            bool: True if the human confirms the summary, False otherwise (e.g., requests correction).
        """
        print("\n--- AGENT CHECKPOINT SUMMARY FOR REVIEW ---")
        print(summary)
        print("------------------------------------------")
        while True:
            response = input("Confirm this summary? (y/n/correct) [y]: ").strip().lower()
            if response in ['y', 'yes', '']:
                return True
            elif response in ['n', 'no']:
                print("Summary not confirmed. Agent will not proceed with this checkpoint.")
                return False
            elif response == 'correct':
                print("Human requested correction. For this prototype, 'correction' means not confirming.")
                return False
            else:
                print("Invalid input. Please enter 'y', 'n', or 'correct'.")

    def save_checkpoint(self, summary: str, agent_state_snapshot: dict,
                        context_snapshot: dict, proposed_actions_snapshot: list,
                        is_confirmed: bool, human_feedback: str = None):
        """
        Saves the checkpoint summary and relevant agent state to the persistent database.

        Args:
            summary (str): The text summary of the checkpoint.
            agent_state_snapshot (dict): A snapshot of the agent's full state at the time of the checkpoint.
            context_snapshot (dict): A snapshot of the aggregated context at the time of the checkpoint.
            proposed_actions_snapshot (list): A snapshot of proposed actions.
            is_confirmed (bool): True if the human confirmed the checkpoint.
            human_feedback (str, optional): Any specific feedback from the human. Defaults to None.
        """
        session = SessionLocal()
        try:
            # Create a new AgentState entry to snapshot the state/context
            # This assumes AgentState model has a flexible data storage like JSONB
            # If not, it would need more specific fields.
            # For simplicity, we'll store agent_state_snapshot and context_snapshot directly in Checkpoint
            # as JSON strings. A more robust solution might link to an AgentState history table.

            new_checkpoint = Checkpoint(
                timestamp=datetime.utcnow(),
                summary_text=summary,
                agent_state_snapshot=json.dumps(agent_state_snapshot),
                context_snapshot=json.dumps(context_snapshot),
                proposed_actions_snapshot=json.dumps(proposed_actions_snapshot),
                is_confirmed_by_human=is_confirmed,
                human_feedback=human_feedback if human_feedback else ""
            )
            session.add(new_checkpoint)
            session.commit()
            print(f"Checkpoint saved successfully (ID: {new_checkpoint.id}). Confirmed: {is_confirmed}")
        except Exception as e:
            session.rollback()
            print(f"Error saving checkpoint to database: {e}")
        finally:
            session.close()

    def create_checkpoint(self, agent_state: dict, current_context: dict, proposed_actions: list) -> bool:
        """
        Orchestrates the entire checkpoint creation process:
        generate summary -> request human confirmation -> save.

        Args:
            agent_state (dict): The current state of the agent.
            current_context (dict): The current aggregated context.
            proposed_actions (list): The actions proposed by the agent.

        Returns:
            bool: True if a checkpoint was successfully created and confirmed by human, False otherwise.
        """
        print("\n--- Initiating checkpoint process ---")
        summary = self.generate_summary(agent_state, current_context, proposed_actions)
        if not summary:
            print("Checkpoint summary generation failed. Aborting checkpoint creation.")
            return False

        is_confirmed = self.request_human_confirmation(summary)
        feedback = None # For this prototype, we don't capture detailed feedback

        self.save_checkpoint(
            summary=summary,
            agent_state_snapshot=agent_state,
            context_snapshot=current_context,
            proposed_actions_snapshot=proposed_actions,
            is_confirmed=is_confirmed,
            human_feedback=feedback
        )

        if is_confirmed:
            print("Checkpoint confirmed and saved. Agent can proceed.")
            return True
        else:
            print("Checkpoint not confirmed. Agent should re-evaluate or await further instructions.")
            return False

# Example usage (for testing purposes, assumes a basic setup of LLMAPI and DB)
if __name__ == "__main__":
    # Mock LLMAPI and DB for standalone testing
    class MockLLMAPI:
        def generate_response(self, messages, max_tokens):
            print(f"Mock LLM received prompt: {messages[0]['content'][:100]}...")
            return "This is a mock summary generated by the AI based on its current understanding and proposed next steps."

    # Create dummy prompt file
    if not os.path.exists("prompts"):
        os.makedirs("prompts")
    with open("prompts/checkpoint_prompt.txt", "w") as f:
        f.write(
            "Summarize the agent's current situation for a human. Be concise.\n\n"
            "Agent State:\n{agent_state_summary}\n\n"
            "Current Context:\n{current_context_summary}\n\n"
            "Proposed Actions:\n{proposed_actions_summary}\n\n"
            "Summary:"
        )

    # Setup dummy database (in-memory SQLite for testing)
    from sqlalchemy import create_engine, Column, Integer, String, Text, Boolean, DateTime
    from sqlalchemy.orm import declarative_base, sessionmaker
    
    Base = declarative_base()

    class Checkpoint(Base):
        __tablename__ = 'checkpoints'
        id = Column(Integer, primary_key=True)
        timestamp = Column(DateTime, default=datetime.utcnow)
        summary_text = Column(Text, nullable=False)
        agent_state_snapshot = Column(Text) # Storing as JSON string
        context_snapshot = Column(Text) # Storing as JSON string
        proposed_actions_snapshot = Column(Text) # Storing as JSON string
        is_confirmed_by_human = Column(Boolean, default=False)
        human_feedback = Column(Text)

    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    mock_llm_api = MockLLMAPI()
    checkpoint_manager = CheckpointManager(llm_api=mock_llm_api, prompts_dir="prompts")

    # Dummy data for agent state and context
    mock_agent_state = {
        "task_id": "proj-X-123",
        "status": "in_progress",
        "current_step": 3,
        "goal": "Implement user authentication feature"
    }
    mock_current_context = {
        "relevant_files": ["src/auth.py", "tests/test_auth.py"],
        "recent_diff": "Added login route to auth.py",
        "conversation_history_summary": "User asked to prioritize security."
    }
    mock_proposed_actions = [
        {"action": "write_code", "details": "Implement password hashing in auth.py"},
        {"action": "run_tests", "details": "Execute unit tests for auth module"},
        {"action": "seek_review", "details": "Ask human for code review on auth.py"}
    ]

    print("\nAttempting to create a checkpoint (you will be prompted for confirmation)...")
    checkpoint_success = checkpoint_manager.create_checkpoint(
        mock_agent_state,
        mock_current_context,
        mock_proposed_actions
    )

    print(f"\nCheckpoint process completed. Success: {checkpoint_success}")

    # Verify checkpoint in DB
    session = SessionLocal()
    checkpoints_in_db = session.query(Checkpoint).all()
    print(f"\nTotal checkpoints in DB: {len(checkpoints_in_db)}")
    for cp in checkpoints_in_db:
        print(f"  ID: {cp.id}, Time: {cp.timestamp}, Confirmed: {cp.is_confirmed_by_human}, Summary: {cp.summary_text[:50]}...")
    session.close()

    # Clean up dummy prompt file
    os.remove("prompts/checkpoint_prompt.txt")
    os.rmdir("prompts")