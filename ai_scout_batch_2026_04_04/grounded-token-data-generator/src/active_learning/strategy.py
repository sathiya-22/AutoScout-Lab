import random
from typing import List, Dict, Optional, Any

class ActiveLearningStrategy:
    """
    Implements various active learning strategies to prioritize candidate descriptions
    for human review, maximizing annotation efficiency.

    Candidates are expected to be dictionaries with at least:
    - 'id': Unique identifier for the candidate description.
    - 'token': The novel token associated with the description.
    - 'description_text': The generated linguistic supervision.
    - 'metrics': A dictionary of quality metrics, e.g., {'uncertainty_score': 0.X, ...}.
                 'uncertainty_score' is crucial for 'uncertainty' strategy.
    - 'status': Current status, e.g., 'pending', 'reviewed', 'approved', 'rejected'.
                Only 'pending' candidates are considered for selection.
    """
    def __init__(self, initial_candidates: Optional[List[Dict[str, Any]]] = None):
        """
        Initializes the ActiveLearningStrategy with a list of candidate descriptions.

        Args:
            initial_candidates: A list of dictionaries, each representing a candidate description.
        """
        self._candidates = initial_candidates if initial_candidates is not None else []
        self._reviewed_ids = set() # To keep track of IDs that have been sent for review

    def update_candidates(self, new_candidates: List[Dict[str, Any]]):
        """
        Updates the internal list of candidates, adding new ones or refreshing existing ones.
        Existing candidates are identified by their 'id'.

        Args:
            new_candidates: A list of dictionaries for new or updated candidate descriptions.
        """
        candidate_map = {c['id']: c for c in self._candidates}
        for candidate in new_candidates:
            candidate_map[candidate['id']] = candidate
        self._candidates = list(candidate_map.values())

        # Clean up _reviewed_ids if a candidate status changes from 'pending'
        # Or simply reset and let get_next_batch handle only 'pending'
        # For simplicity, _reviewed_ids tracks what was *sent* for review, not necessarily completed.
        # A proper HIL loop would explicitly mark items as 'reviewed' (not just 'pending')
        # Here we rely on `get_pending_candidates` to filter.

    def _get_pending_candidates(self) -> List[Dict[str, Any]]:
        """
        Filters the candidate list to return only those with 'pending' status
        that have not yet been sent for review in the current session (if _reviewed_ids is used).
        """
        pending = [
            c for c in self._candidates
            if c.get('status', 'pending') == 'pending' and c['id'] not in self._reviewed_ids
        ]
        return pending

    def _sample_by_uncertainty(self, candidates: List[Dict[str, Any]], batch_size: int) -> List[Dict[str, Any]]:
        """
        Selects candidates with the highest uncertainty score (or lowest quality score).
        Assumes 'uncertainty_score' is present in candidate['metrics'].
        Higher uncertainty_score means more uncertain/challenging.
        """
        if not candidates:
            return []

        # Ensure all candidates have an uncertainty_score
        valid_candidates = []
        for c in candidates:
            if 'metrics' in c and 'uncertainty_score' in c['metrics']:
                valid_candidates.append(c)
            else:
                # Log a warning or handle candidates without the required metric
                # For now, we'll just skip them.
                pass # print(f"Warning: Candidate {c.get('id', 'N/A')} missing 'uncertainty_score' in metrics.")

        if not valid_candidates:
            return []

        # Sort by uncertainty score in descending order (highest uncertainty first)
        sorted_candidates = sorted(
            valid_candidates,
            key=lambda c: c['metrics'].get('uncertainty_score', -1.0), # Use -1.0 for missing score to put them at the end
            reverse=True
        )
        return sorted_candidates[:batch_size]

    def _sample_by_diversity(self, candidates: List[Dict[str, Any]], batch_size: int) -> List[Dict[str, Any]]:
        """
        Selects candidates to maximize diversity. This is a simplified placeholder.
        A more sophisticated implementation would use embedding similarity, n-gram overlap, etc.
        For now, it might try to pick distinct tokens if available, or just fallback to random if
        a specific diversity metric is not available.
        A very basic diversity might involve spreading across different source tokens.
        """
        if not candidates:
            return []

        # A truly diverse sampling would involve clustering or measuring distances.
        # For a basic prototype, we can prioritize candidates that are distinct in some simple way
        # or simply fall back to random if no explicit diversity metric is provided.
        # Let's assume a 'diversity_score' exists where higher is more diverse.
        valid_candidates = []
        for c in candidates:
            if 'metrics' in c and 'diversity_score' in c['metrics']:
                valid_candidates.append(c)
            else:
                # Fallback: if no explicit diversity_score, treat all as equally 'diverse'
                # or use uncertainty as a secondary.
                pass # print(f"Warning: Candidate {c.get('id', 'N/A')} missing 'diversity_score' in metrics.")

        if valid_candidates:
            # Sort by diversity score in descending order (highest diversity first)
            sorted_candidates = sorted(
                valid_candidates,
                key=lambda c: c['metrics'].get('diversity_score', -1.0),
                reverse=True
            )
            return sorted_candidates[:batch_size]
        else:
            # Fallback to random if no diversity_score is present
            return self._sample_by_random(candidates, batch_size)


    def _sample_by_random(self, candidates: List[Dict[str, Any]], batch_size: int) -> List[Dict[str, Any]]:
        """
        Selects candidates randomly.
        """
        if not candidates:
            return []
        return random.sample(candidates, min(batch_size, len(candidates)))

    def get_next_batch(self, batch_size: int = 10, strategy: str = 'uncertainty') -> List[Dict[str, Any]]:
        """
        Returns the next batch of candidate descriptions for human review based on the specified strategy.

        Args:
            batch_size: The maximum number of descriptions to return in the batch.
            strategy: The active learning strategy to use ('uncertainty', 'diversity', 'random').

        Returns:
            A list of candidate description dictionaries prioritized for review.
        """
        pending_candidates = self._get_pending_candidates()

        if not pending_candidates:
            return []

        selected_batch = []
        if strategy == 'uncertainty':
            selected_batch = self._sample_by_uncertainty(pending_candidates, batch_size)
        elif strategy == 'diversity':
            selected_batch = self._sample_by_diversity(pending_candidates, batch_size)
        elif strategy == 'random':
            selected_batch = self._sample_by_random(pending_candidates, batch_size)
        else:
            raise ValueError(f"Unknown active learning strategy: {strategy}. Choose from 'uncertainty', 'diversity', 'random'.")

        # Mark selected candidates as "sent for review" by adding their IDs to _reviewed_ids
        # This prevents selecting the same items multiple times in subsequent calls before their status is updated.
        for item in selected_batch:
            self._reviewed_ids.add(item['id'])

        return selected_batch

    def reset_reviewed_status(self, candidate_ids: Optional[List[str]] = None):
        """
        Resets the 'sent for review' status for specified candidates or all candidates.
        This is useful if a batch was sent but not actually reviewed, or for re-evaluation.

        Args:
            candidate_ids: A list of candidate IDs to reset. If None, all reviewed status is cleared.
        """
        if candidate_ids is None:
            self._reviewed_ids.clear()
        else:
            for _id in candidate_ids:
                self._reviewed_ids.discard(_id)

    def get_candidate_by_id(self, candidate_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a specific candidate by its ID.
        """
        for c in self._candidates:
            if c['id'] == candidate_id:
                return c
        return None

# Example Usage (for testing purposes, remove in final production code if not needed)
if __name__ == "__main__":
    # Mock data for demonstration
    mock_candidates = [
        {
            'id': 'desc_001', 'token': 'tokenA', 'description_text': 'Description for token A, highly uncertain.',
            'metrics': {'uncertainty_score': 0.95, 'diversity_score': 0.1}, 'status': 'pending'
        },
        {
            'id': 'desc_002', 'token': 'tokenB', 'description_text': 'Description for token B, moderately certain.',
            'metrics': {'uncertainty_score': 0.60, 'diversity_score': 0.7}, 'status': 'pending'
        },
        {
            'id': 'desc_003', 'token': 'tokenC', 'description_text': 'Description for token C, very certain.',
            'metrics': {'uncertainty_score': 0.10, 'diversity_score': 0.9}, 'status': 'pending'
        },
        {
            'id': 'desc_004', 'token': 'tokenD', 'description_text': 'Another description for token A, somewhat uncertain.',
            'metrics': {'uncertainty_score': 0.70, 'diversity_score': 0.2}, 'status': 'pending'
        },
        {
            'id': 'desc_005', 'token': 'tokenE', 'description_text': 'Description for token E, moderately uncertain.',
            'metrics': {'uncertainty_score': 0.80, 'diversity_score': 0.5}, 'status': 'pending'
        },
        {
            'id': 'desc_006', 'token': 'tokenF', 'description_text': 'Description for token F, already approved.',
            'metrics': {'uncertainty_score': 0.05, 'diversity_score': 0.95}, 'status': 'approved'
        },
        {
            'id': 'desc_007', 'token': 'tokenG', 'description_text': 'Description for token G, missing metrics.',
            'metrics': {}, 'status': 'pending'
        },
        {
            'id': 'desc_008', 'token': 'tokenH', 'description_text': 'Description for token H, also highly uncertain.',
            'metrics': {'uncertainty_score': 0.90, 'diversity_score': 0.3}, 'status': 'pending'
        },
    ]

    print("--- Initializing Active Learning Strategy ---")
    active_learner = ActiveLearningStrategy(initial_candidates=mock_candidates)

    print("\n--- Getting next batch (Uncertainty Strategy, size 3) ---")
    batch1_uncertainty = active_learner.get_next_batch(batch_size=3, strategy='uncertainty')
    for i, item in enumerate(batch1_uncertainty):
        print(f"  {i+1}. ID: {item['id']}, Token: {item['token']}, Uncertainty: {item['metrics'].get('uncertainty_score')}")

    print("\n--- Getting next batch (Uncertainty Strategy, size 2) - should pick from remaining pending ---")
    batch2_uncertainty = active_learner.get_next_batch(batch_size=2, strategy='uncertainty')
    for i, item in enumerate(batch2_uncertainty):
        print(f"  {i+1}. ID: {item['id']}, Token: {item['token']}, Uncertainty: {item['metrics'].get('uncertainty_score')}")

    print("\n--- Simulating human review: 'desc_001' and 'desc_004' are approved ---")
    # Update candidate status based on human feedback
    updated_candidates = [
        {'id': 'desc_001', 'token': 'tokenA', 'description_text': 'Description for token A, highly uncertain.', 'metrics': {'uncertainty_score': 0.95, 'diversity_score': 0.1}, 'status': 'approved'},
        {'id': 'desc_004', 'token': 'tokenD', 'description_text': 'Another description for token A, somewhat uncertain.', 'metrics': {'uncertainty_score': 0.70, 'diversity_score': 0.2}, 'status': 'approved'},
    ]
    active_learner.update_candidates(updated_candidates)
    active_learner.reset_reviewed_status(candidate_ids=['desc_001', 'desc_004']) # Reset reviewed status for approved items

    print("\n--- Getting next batch (Diversity Strategy, size 3) ---")
    batch_diversity = active_learner.get_next_batch(batch_size=3, strategy='diversity')
    for i, item in enumerate(batch_diversity):
        print(f"  {i+1}. ID: {item['id']}, Token: {item['token']}, Diversity: {item['metrics'].get('diversity_score')}")

    print("\n--- Getting next batch (Random Strategy, size 2) ---")
    active_learner.reset_reviewed_status() # Clear all internal reviewed statuses to get a fresh random sample
    batch_random = active_learner.get_next_batch(batch_size=2, strategy='random')
    for i, item in enumerate(batch_random):
        print(f"  {i+1}. ID: {item['id']}, Token: {item['token']}")

    print("\n--- Attempting to get a batch with an invalid strategy ---")
    try:
        active_learner.get_next_batch(batch_size=1, strategy='invalid_strategy')
    except ValueError as e:
        print(f"  Error: {e}")

    print("\n--- Testing with no pending candidates (after all are processed/selected) ---")
    # Mark all remaining pending as reviewed/approved for this test
    all_pending_ids = [c['id'] for c in active_learner._get_pending_candidates()]
    active_learner.update_candidates([
        {**c, 'status': 'approved'} for c in active_learner._get_pending_candidates()
    ])
    active_learner.reset_reviewed_status() # Clear internal tracking to simulate system restart for statuses to matter

    empty_batch = active_learner.get_next_batch(batch_size=1, strategy='uncertainty')
    print(f"  Batch from no pending candidates: {empty_batch}")

    print("\n--- Testing candidate without uncertainty score for uncertainty strategy ---")
    candidates_no_uncertainty = [
        {'id': 'desc_100', 'token': 'tokenX', 'description_text': 'Desc X.', 'metrics': {'diversity_score': 0.5}, 'status': 'pending'},
        {'id': 'desc_101', 'token': 'tokenY', 'description_text': 'Desc Y.', 'metrics': {'uncertainty_score': 0.7}, 'status': 'pending'}
    ]
    active_learner_test = ActiveLearningStrategy(initial_candidates=candidates_no_uncertainty)
    batch_mixed_metrics = active_learner_test.get_next_batch(batch_size=2, strategy='uncertainty')
    # Should only return 'desc_101' as 'desc_100' lacks uncertainty_score and is filtered out by _sample_by_uncertainty
    print(f"  Batch with mixed metrics (uncertainty strategy): {[{'id': c['id'], 'uncertainty': c['metrics'].get('uncertainty_score')} for c in batch_mixed_metrics]}")

    print("\n--- Testing candidate without diversity score for diversity strategy ---")
    candidates_no_diversity = [
        {'id': 'desc_200', 'token': 'tokenZ', 'description_text': 'Desc Z.', 'metrics': {'uncertainty_score': 0.8}, 'status': 'pending'},
        {'id': 'desc_201', 'token': 'tokenW', 'description_text': 'Desc W.', 'metrics': {'diversity_score': 0.6}, 'status': 'pending'}
    ]
    active_learner_test_diversity = ActiveLearningStrategy(initial_candidates=candidates_no_diversity)
    batch_mixed_metrics_diversity = active_learner_test_diversity.get_next_batch(batch_size=2, strategy='diversity')
    # If a diversity score is present, it will use that. If not, it falls back to random.
    # Here, desc_200 lacks diversity_score, so _sample_by_diversity falls back to random, meaning both could be picked.
    print(f"  Batch with mixed metrics (diversity strategy): {[{'id': c['id'], 'diversity': c['metrics'].get('diversity_score')} for c in batch_mixed_metrics_diversity]}")<ctrl63>