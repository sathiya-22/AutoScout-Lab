```python
import uuid
from typing import Dict, List, Any, Optional, Union

class Subject:
    """
    Encapsulates per-subject data: ID, spatial coordinates, action queue,
    latent token index, and optional identity features.

    Attributes:
        ID (Union[str, int]): A unique identifier for the subject.
        bounding_box (List[float]): Current and potentially historical 2D/3D positions
                                     (e.g., [x, y, w, h] or keypoints).
        action_queue (List[str]): A sequence of pending or active actions/intentions
                                  associated with the subject.
        latent_token_idx (int): Index referencing the subject's dedicated latent vector
                                in the LatentVectorPool.
        identity_features (Optional[Any]): Optional, pre-computed or extracted features
                                           (e.g., CLIP embeddings) for robust identity tracking.
    """
    def __init__(
        self,
        subject_id: Union[str, int],
        bounding_box: List[float], # Example: [x, y, w, h] or [x, y, z] or list of keypoints
        action_queue: List[str], # Example: ["walking", "waving left arm"]
        latent_token_idx: int,
        identity_features: Optional[Any] = None # e.g., a Tensor for embeddings
    ):
        # Basic type and value validation
        if not isinstance(subject_id, (str, int)):
            raise TypeError("subject_id must be a string or integer.")
        if not isinstance(bounding_box, list) or not all(isinstance(x, (int, float)) for x in bounding_box):
            raise TypeError("bounding_box must be a list of numeric values (int or float).")
        if not isinstance(action_queue, list) or not all(isinstance(x, str) for x in action_queue):
            raise TypeError("action_queue must be a list of strings.")
        if not isinstance(latent_token_idx, int):
            raise TypeError("latent_token_idx must be an integer.")
        if latent_token_idx < 0:
            raise ValueError("latent_token_idx must be a non-negative integer.")

        self.ID: Union[str, int] = subject_id
        # Store a copy to prevent external modification of the passed list
        self.bounding_box: List[float] = list(bounding_box)
        # Store a copy
        self.action_queue: List[str] = list(action_queue)
        self.latent_token_idx: int = latent_token_idx
        self.identity_features: Optional[Any] = identity_features

    def update(
        self,
        bounding_box: Optional[List[float]] = None,
        action_queue: Optional[List[str]] = None,
        latent_token_idx: Optional[int] = None,
        identity_features: Optional[Any] = None
    ):
        """
        Updates the subject's state with new information.
        Only non-None parameters will update the corresponding attributes.

        Args:
            bounding_box (Optional[List[float]]): New spatial coordinates.
            action_queue (Optional[List[str]]): New list of actions.
            latent_token_idx (Optional[int]): New latent token index.
            identity_features (Optional[Any]): New identity features.

        Raises:
            TypeError: If an update parameter has an invalid type.
            ValueError: If an update parameter has an invalid value.
        """
        if bounding_box is not None:
            if not isinstance(bounding_box, list) or not all(isinstance(x, (int, float)) for x in bounding_box):
                raise TypeError("bounding_box must be a list of numeric values (int or float).")
            self.bounding_box = list(bounding_box) # Store a copy
        if action_queue is not None:
            if not isinstance(action_queue, list) or not all(isinstance(x, str) for x in action_queue):
                raise TypeError("action_queue must be a list of strings.")
            self.action_queue = list(action_queue) # Store a copy
        if latent_token_idx is not None:
            if not isinstance(latent_token_idx, int):
                raise TypeError("latent_token_idx must be an integer.")
            if latent_token_idx < 0:
                raise ValueError("latent_token_idx must be a non-negative integer.")
            self.latent_token_idx = latent_token_idx
        if identity_features is not None:
            self.identity_features = identity_features

    def __repr__(self) -> str:
        """Provides a string representation of the Subject object."""
        return (f"Subject(ID={self.ID}, bbox={self.bounding_box}, "
                f"actions={self.action_queue}, latent_idx={self.latent_token_idx}, "
                f"has_identity_features={self.identity_features is not None})")

    def __eq__(self, other: object) -> bool:
        """Compares two Subject objects based on their IDs."""
        if not isinstance(other, Subject):
            return NotImplemented
        return self.ID == other.ID


class DynamicSubjectRegistry:
    """
    Maintains a real-time record of all active subjects in the scene.
    It acts as a central repository, mapping unique `subject_ID`s to `Subject` objects.
    """
    def __init__(self):
        self._subjects: Dict[Union[str, int], Subject] = {}

    def add_subject(
        self,
        subject_id: Union[str, int],
        bounding_box: List[float],
        action_queue: List[str],
        latent_token_idx: int,
        identity_features: Optional[Any] = None
    ) -> Subject:
        """
        Adds a new subject to the registry.

        Args:
            subject_id (Union[str, int]): A unique identifier for the subject.
            bounding_box (List[float]): Current spatial coordinates (e.g., [x, y, w, h]).
            action_queue (List[str]): A list of pending or active actions.
            latent_token_idx (int): Index referencing the subject's dedicated latent vector.
            identity_features (Optional[Any]): Optional, pre-computed or extracted features.

        Returns:
            Subject: The newly created Subject object.

        Raises:
            ValueError: If a subject with the given ID already exists or if input validation fails.
        """
        if subject_id in self._subjects:
            raise ValueError(f"Subject with ID '{subject_id}' already exists in the registry.")

        try:
            subject = Subject(subject_id, bounding_box, action_queue, latent_token_idx, identity_features)
            self._subjects[subject_id] = subject
            return subject
        except (TypeError, ValueError) as e:
            raise ValueError(f"Failed to add subject '{subject_id}' due to invalid input: {e}")

    def remove_subject(self, subject_id: Union[str, int]) -> None:
        """
        Removes a subject from the registry.

        Args:
            subject_id (Union[str, int]): The ID of the subject to remove.

        Raises:
            KeyError: If the subject with the given ID does not exist.
        """
        if subject_id not in self._subjects:
            raise KeyError(f"Subject with ID '{subject_id}' not found in the registry.")
        del self._subjects[subject_id]

    def update_subject_state(
        self,
        subject_id: Union[str, int],
        bounding_box: Optional[List[float]] = None,
        action_queue: Optional[List[str]] = None,
        latent_token_idx: Optional[int] = None,
        identity_features: Optional[Any] = None
    ) -> Subject:
        """
        Updates the state of an existing subject.

        Args:
            subject_id (Union[str, int]): The ID of the subject to update.
            bounding_box (Optional[List[float]]): New spatial coordinates.
            action_queue (Optional[List[str]]): New list of actions.
            latent_token_idx (Optional[int]): New latent token index.
            identity_features (Optional[Any]): New identity features.

        Returns:
            Subject: The updated Subject object.

        Raises:
            KeyError: If the subject with the given ID does not exist.
            ValueError: If invalid input types or values are provided during update.
        """
        if subject_id not in self._subjects:
            raise KeyError(f"Subject with ID '{subject_id}' not found in the registry.")

        subject = self._subjects[subject_id]
        try:
            subject.update(bounding_box, action_queue, latent_token_idx, identity_features)
            return subject
        except (TypeError, ValueError) as e:
            raise ValueError(f"Failed to update subject '{subject_id}' due to invalid input: {e}")

    def get_subject_info(self, subject_id: Union[str, int]) -> Subject:
        """
        Retrieves the Subject object for a given ID.

        Args:
            subject_id (Union[str, int]): The ID of the subject to retrieve.

        Returns:
            Subject: The Subject object.

        Raises:
            KeyError: If the subject with the given ID does not exist.
        """
        if subject_id not in self._subjects:
            raise KeyError(f"Subject with ID '{subject_id}' not found in the registry.")
        return self._subjects[subject_id]

    def get_all_subjects(self) -> Dict[Union[str, int], Subject]:
        """
        Returns a shallow copy of the dictionary of all active subjects.
        Modifying the returned dictionary will not affect the registry's internal state.

        Returns:
            Dict[Union[str, int], Subject]: A dictionary mapping subject IDs to Subject objects.
        """
        return self._subjects.copy()

    def get_num_subjects(self) -> int:
        """
        Returns the current number of subjects in the registry.

        Returns:
            int: The count of subjects.
        """
        return len(self._subjects)

    def __len__(self) -> int:
        """Allows `len(registry)` to return the number of subjects."""
        return self.get_num_subjects()

    def __contains__(self, subject_id: Union[str, int]) -> bool:
        """Allows `'subject_id' in registry` check."""
        return subject_id in self._subjects

    def __iter__(self):
        """Allows iteration over Subject objects directly."""
        return iter(self._subjects.values())
```