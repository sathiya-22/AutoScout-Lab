import os
import json
import datetime
import uuid
import google.generativeai as genai
from config import config

class TraceLogger:
    def __init__(self, log_file="agent_traces.jsonl"):
        self.log_file = log_file
        self.current_trace = None
        self.traces = []

    def start_trace(self, trace_type="agent_interaction", initial_data=None):
        self.current_trace = {
            "trace_id": str(uuid.uuid4()),
            "timestamp_start": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "type": trace_type,
            "events": []
        }
        if initial_data:
            self.current_trace.update(initial_data)
        self.log_event("trace_started", initial_data={"message": "Trace initiated."})
        return self.current_trace["trace_id"]

    def log_event(self, event_type, data=None):
        if not self.current_trace:
            print("Warning: No active trace. Event not logged.")
            return

        event = {
            "event_id": str(uuid.uuid4()),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "type": event_type,
            "data": data if data is not None else {}
        }
        self.current_trace["events"].append(event)

    def end_trace(self, final_data=None):
        if not self.current_trace:
            print("Warning: No active trace to end.")
            return

        self.log_event("trace_ended", final_data={"message": "Trace completed."})
        self.current_trace["timestamp_end"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        if final_data:
            self.current_trace.update(final_data)

        self.traces.append(self.current_trace)
        self.current_trace = None # Reset for next trace
        return self.traces[-1]["trace_id"]

    def save_traces(self):
        """Appends all accumulated traces to the log file in JSON Lines format."""
        with open(self.log_file, "a") as f:
            for trace in self.traces:
                f.write(json.dumps(trace) + "\n")
        self.traces = [] # Clear saved traces

def main():
    # Configure Google GenAI
    genai.configure(api_key=config.api_key)

    # Initialize logger
    logger = TraceLogger()

    # Initialize the generative model
    model = genai.GenerativeModel(
        model_name=config.model_name,
        generation_config={
            "temperature": config.temperature,
            "max_output_tokens": config.max_output_tokens,
        },
    )

    # --- Simulate Agent Interaction 1 ---
    print("\n--- Agent Interaction 1: Asking about Python ---")
    prompt1 = "Explain the concept of decorators in Python."
    trace_id1 = logger.start_trace(initial_data={"agent_task": "Explain Python concept", "input_prompt": prompt1})
    logger.log_event("prompt_input", data={"prompt": prompt1})

    try:
        logger.log_event("model_call_start", data={"model": config.model_name, "prompt_length": len(prompt1)})
        response1 = model.generate_content(prompt1)
        logger.log_event("model_response_received", data={
            "text_output": response1.text,
            "safety_ratings": [str(s) for s in response1.prompt_feedback.safety_ratings],
            "finish_reason": response1.candidates[0].finish_reason.name if response1.candidates else "N/A"
        })
        print(f"Model Response (shortened): {response1.text[:150]}...")
    except Exception as e:
        logger.log_event("model_error", data={"error_message": str(e)})
        print(f"Error during model call: {e}")
    finally:
        logger.end_trace(final_data={"status": "completed"})

    # --- Simulate Agent Interaction 2 ---
    print("\n--- Agent Interaction 2: Asking for a creative story ---")
    prompt2 = "Write a very short, imaginative story about a cat who can speak but only in haikus."
    trace_id2 = logger.start_trace(initial_data={"agent_task": "Generate creative story", "input_prompt": prompt2})
    logger.log_event("prompt_input", data={"prompt": prompt2})

    try:
        logger.log_event("model_call_start", data={"model": config.model_name, "prompt_length": len(prompt2)})
        response2 = model.generate_content(prompt2)
        logger.log_event("model_response_received", data={
            "text_output": response2.text,
            "safety_ratings": [str(s) for s in response2.prompt_feedback.safety_ratings],
            "finish_reason": response2.candidates[0].finish_reason.name if response2.candidates else "N/A"
        })
        print(f"Model Response (shortened): {response2.text[:150]}...")
    except Exception as e:
        logger.log_event("model_error", data={"error_message": str(e)})
        print(f"Error during model call: {e}")
    finally:
        logger.end_trace(final_data={"status": "completed"})

    # Save all accumulated traces to file
    logger.save_traces()
    print(f"\nTraces saved to {logger.log_file}")

if __name__ == "__main__":
    main()
