from flask import Flask, request, jsonify
import uuid
import os
import json

# --- Mock Implementations for Core Components (if real ones are not present) ---
class MockGroundedTokenGenerator:
    """A mock generator for prototype purposes, simulating the functionality of src/core/generator.py."""
    def generate_descriptions(self, tokens_data):
        candidates = []
        for token_info in tokens_data:
            token_name = token_info.get('token')
            initial_context = token_info.get('context', '')
            if not token_name: continue
            for i in range(1, 3): # Generate 2 candidates per token for demonstration
                candidates.append({
                    "candidate_id": str(uuid.uuid4()),
                    "token": token_name,
                    "description": f"Generated description {i} for '{token_name}'. Context: '{initial_context}'.",
                    "status": "pending_review",
                    "quality_scores": {"fluency": round(0.5 + i*0.2, 2), "relevance": round(0.6 + i*0.1, 2)}
                })
        return candidates

class MockQualityAssessment:
    """A mock quality assessor for prototype purposes, simulating src/core/quality_metrics.py."""
    def assess(self, description, token_info=None):
        # Simulate some assessment scores
        return {"fluency": 0.75, "relevance": 0.8, "diversity": 0.6}

class MockActiveLearningStrategy:
    """A mock active learning strategy for prototype purposes, simulating src/active_learning/strategy.py."""
    def select_for_review(self, candidates, num_to_select=5):
        # For simplicity, select the first N candidates that are pending review,
        # sorted by lowest relevance score (simulating uncertainty or high-need for review)
        pending = [c for c in candidates if c.get('status') == 'pending_review']
        return sorted(pending, key=lambda x: x.get('quality_scores', {}).get('relevance', 1.0))[:num_to_select]

# --- Initialize Core Components (attempt real imports, fall back to mocks) ---
try:
    from src.core.generator import GroundedTokenGenerator
    generator = GroundedTokenGenerator()
except ImportError:
    print("Warning: src.core.generator.GroundedTokenGenerator not found. Using mock generator.")
    generator = MockGroundedTokenGenerator()

try:
    from src.core.quality_metrics import QualityAssessment
    quality_assessor = QualityAssessment() # pylint: disable=unused-variable
except ImportError:
    print("Warning: src.core.quality_metrics.QualityAssessment not found. Using mock quality assessor.")
    quality_assessor = MockQualityAssessment() # pylint: disable=unused-variable

try:
    from src.active_learning.strategy import ActiveLearningStrategy
    active_learner = ActiveLearningStrategy()
except ImportError:
    print("Warning: src.active_learning.strategy.ActiveLearningStrategy not found. Using mock active learning strategy.")
    active_learner = MockActiveLearningStrategy()

app = Flask(__name__)

# --- In-memory data store for prototype ---
# Stores all candidate descriptions (pending, approved, rejected, edited) keyed by candidate_id.
# For a production system, this would be a database.
CANDIDATE_STORE = {}

# Path for outputting validated data
OUTPUT_DIR = 'src/data/output'
os.makedirs(OUTPUT_DIR, exist_ok=True)
VALIDATED_OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'generated_supervision.json')

# Helper functions for prototype persistence (simple JSON file)
def _load_candidates_from_file():
    """Loads previously validated (approved/edited) descriptions from the output file into the CANDIDATE_STORE."""
    global CANDIDATE_STORE
    try:
        if os.path.exists(VALIDATED_OUTPUT_FILE):
            with open(VALIDATED_OUTPUT_FILE, 'r', encoding='utf-8') as f:
                validated_data = json.load(f)
                for item in validated_data:
                    CANDIDATE_STORE[item.get('candidate_id')] = item
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"No existing or valid '{VALIDATED_OUTPUT_FILE}' found. Starting with empty store. Error: {e}")
        CANDIDATE_STORE = {}
    except Exception as e:
        print(f"Error loading initial validated data from file: {e}")
        CANDIDATE_STORE = {}

def _save_candidates_to_file():
    """Saves only approved and edited candidate descriptions to the designated output file."""
    try:
        validated_data_for_output = [
            c for c in CANDIDATE_STORE.values()
            if c.get('status') in ['approved', 'edited']
        ]
        with open(VALIDATED_OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(validated_data_for_output, f, indent=4)
    except Exception as e:
        print(f"Error saving validated candidates to file: {e}")

# Load initial state on API startup
_load_candidates_from_file()

@app.route('/tokens/generate', methods=['POST'])
def generate_token_supervision():
    """
    Endpoint to submit new tokens for generating candidate linguistic supervision.
    Expected request body:
    {
        "tokens": [
            {"token": "new_word_1", "context": "biology term related to cellular processes"},
            {"token": "new_word_2", "context": "AI concept, type of neural network"},
            ...
        ]
    }
    """
    data = request.get_json()
    if not data or 'tokens' not in data:
        return jsonify({"error": "Missing 'tokens' list in request body."}), 400

    new_tokens_data = data['tokens']
    if not isinstance(new_tokens_data, list):
        return jsonify({"error": "'tokens' must be a list of token objects."}), 400

    generated_candidates = []
    try:
        candidates = generator.generate_descriptions(new_tokens_data)

        for candidate in candidates:
            # Ensure each candidate has a unique ID, even if generator doesn't provide one
            if 'candidate_id' not in candidate or not candidate['candidate_id']:
                candidate['candidate_id'] = str(uuid.uuid4())
            CANDIDATE_STORE[candidate['candidate_id']] = candidate
            generated_candidates.append(candidate)

        return jsonify({
            "message": f"Successfully generated {len(generated_candidates)} candidate descriptions.",
            "candidates_generated": generated_candidates
        }), 200
    except Exception as e:
        print(f"Error during token generation: {e}")
        return jsonify({"error": f"Failed to generate descriptions: {str(e)}"}), 500

@app.route('/supervision/candidates', methods=['GET'])
def get_supervision_candidates():
    """
    Endpoint to retrieve candidate descriptions for human review.
    Supports filtering and active learning selection.
    Query parameters:
    - status: Filter by status (e.g., 'pending_review', 'approved', 'rejected', 'edited').
    - token: Filter by original token name.
    - limit: Limit the number of results.
    - select_for_review: If 'true', use active learning strategy to select candidates.
    """
    status_filter = request.args.get('status')
    token_filter = request.args.get('token')
    limit = request.args.get('limit', type=int)
    select_for_review = request.args.get('select_for_review', 'false').lower() == 'true'

    filtered_candidates = list(CANDIDATE_STORE.values())

    if status_filter:
        filtered_candidates = [c for c in filtered_candidates if c.get('status') == status_filter]
    if token_filter:
        filtered_candidates = [c for c in filtered_candidates if c.get('token') == token_filter]

    if select_for_review:
        # Use active learning strategy to pick the most relevant ones for review
        candidates_to_present = active_learner.select_for_review(filtered_candidates, num_to_select=limit if limit else 10)
    else:
        candidates_to_present = filtered_candidates

    if limit and not select_for_review: # Apply limit only if not already done by active learner
        candidates_to_present = candidates_to_present[:limit]

    return jsonify(candidates_to_present), 200

@app.route('/supervision/validate', methods=['POST'])
def validate_supervision_feedback():
    """
    Endpoint to submit human validation feedback for candidate descriptions.
    Expected request body:
    {
        "validations": [
            {
                "candidate_id": "uuid_1",
                "status": "approved",  // "approved", "rejected", "edited"
                "edited_text": null,   // Required if status is "edited"
                "reviewer_id": "human_1"
            },
            {
                "candidate_id": "uuid_2",
                "status": "edited",
                "edited_text": "A refined and accurate description for new_word_2.",
                "reviewer_id": "human_1"
            },
            ...
        ]
    }
    """
    data = request.get_json()
    if not data or 'validations' not in data:
        return jsonify({"error": "Missing 'validations' list in request body."}), 400

    validations = data['validations']
    if not isinstance(validations, list):
        return jsonify({"error": "'validations' must be a list of validation objects."}), 400

    updated_count = 0
    errors = []
    updated_items_summary = []

    for validation in validations:
        candidate_id = validation.get('candidate_id')
        status = validation.get('status')
        edited_text = validation.get('edited_text')
        reviewer_id = validation.get('reviewer_id', 'anonymous')

        if not candidate_id or not status:
            errors.append(f"Invalid validation entry: Missing candidate_id or status. Entry: {validation}")
            continue

        if candidate_id not in CANDIDATE_STORE:
            errors.append(f"Candidate ID '{candidate_id}' not found in store.")
            continue

        candidate = CANDIDATE_STORE[candidate_id]

        if status not in ['approved', 'rejected', 'edited']:
            errors.append(f"Invalid status '{status}' for candidate ID '{candidate_id}'. Allowed: approved, rejected, edited.")
            continue

        if status == 'edited' and not isinstance(edited_text, str): # Ensure edited_text is a non-empty string for 'edited' status
            errors.append(f"Edited status requires a valid 'edited_text' string for candidate ID '{candidate_id}'.")
            continue

        candidate['status'] = status
        candidate['reviewer_id'] = reviewer_id
        if edited_text:
            candidate['edited_text'] = edited_text
        elif 'edited_text' in candidate: # Clear previous edited_text if status changes from 'edited'
            del candidate['edited_text']

        updated_count += 1
        CANDIDATE_STORE[candidate_id] = candidate # Update in the global store
        updated_items_summary.append({"candidate_id": candidate_id, "status": status})

    # Persist the changes to the output file (only validated items)
    _save_candidates_to_file()

    if errors:
        return jsonify({
            "message": f"Processed {updated_count} validations with {len(errors)} errors.",
            "errors": errors,
            "updated_items_summary": updated_items_summary
        }), 207 # Multi-Status indicates partial success/failures
    else:
        return jsonify({
            "message": f"Successfully processed {updated_count} validations.",
            "updated_items_summary": updated_items_summary
        }), 200

if __name__ == '__main__':
    # Run the Flask app in debug mode for development.
    # For production, use a more robust WSGI server like Gunicorn.
    app.run(debug=True, host='0.0.0.0', port=5000)