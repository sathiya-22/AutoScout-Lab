```python
import uuid
import torch
from typing import Dict, List, Any, Optional, Tuple, Set

# Subject Object
class Subject:
    """
    Encapsulates per-subject data, including identity, spatial information,
    action queues, and a reference to its latent token.
    """
    def __init__(
        self,
        subject_id: str,
        bounding_box: List[float], # [x_min, y_min, x_max, y_max] or [x, y, w, h]
        latent_token_idx: int,
        action_queue: Optional[List[str]] = None,
        identity_features: Optional[torch.Tensor] = None, # e.g., CLIP embedding
    ):
        if not isinstance(subject_id, str) or not subject_id:
            raise ValueError("Subject ID must be a non-empty string.")
        if not isinstance(bounding_box, list) or len(bounding_box) != 4:
            raise ValueError("Bounding box must be a list of 4 floats [x, y, w, h].")
        if not all(isinstance(coord, (int, float)) for coord in bounding_box):
            raise ValueError("Bounding box coordinates must be numeric.")
        if not isinstance(latent_token_idx, int) or latent_token_idx < 0:
            raise ValueError("Latent token index must be a non-negative integer.")

        self.id = subject_id
        self.bounding_box = bounding_box
        self.action_queue = action_queue if action_queue is not None else []
        self.latent_token_idx = latent_token_idx
        self.identity_features = identity_features

    def update_state(
        self,
        bounding_box: Optional[List[float]] = None,
        action_queue: Optional[List[str]] = None,
        identity_features: Optional[torch.Tensor] = None,
    ):
        """
        Updates the subject's non-latent state information.
        """
        if bounding_box is not None:
            if not isinstance(bounding_box, list) or len(bounding_box) != 4:
                raise ValueError("Bounding box must be a list of 4 floats [x, y, w, h].")
            if not all(isinstance(coord, (int, float)) for coord in bounding_box):
                raise ValueError("Bounding box coordinates must be numeric.")
            self.bounding_box = bounding_box
        if action_queue is not None:
            if not isinstance(action_queue, list):
                raise ValueError("Action queue must be a list of strings.")
            self.action_queue = action_queue
        if identity_features is not None:
            if not isinstance(identity_features, torch.Tensor):
                raise ValueError("Identity features must be a torch.Tensor.")
            self.identity_features = identity_features

    def __repr__(self):
        return (f"Subject(ID='{self.id}', bbox={self.bounding_box}, "
                f"latent_idx={self.latent_token_idx}, actions={self.action_queue})")


# Dynamic Subject Registry
class SubjectRegistry:
    """
    Maintains a real-time record of all active subjects in the scene.
    Maps unique subject_ID to Subject objects.
    """
    def __init__(self):
        self._registry: Dict[str, Subject] = {}

    def add_subject(
        self,
        subject_id: str,
        bounding_box: List[float],
        latent_token_idx: int,
        action_queue: Optional[List[str]] = None,
        identity_features: Optional[torch.Tensor] = None,
    ) -> Subject:
        """
        Adds a new subject to the registry.
        Raises ValueError if subject_id already exists.
        """
        if subject_id in self._registry:
            raise ValueError(f"Subject with ID '{subject_id}' already exists.")
        
        subject = Subject(
            subject_id=subject_id,
            bounding_box=bounding_box,
            latent_token_idx=latent_token_idx,
            action_queue=action_queue,
            identity_features=identity_features,
        )
        self._registry[subject_id] = subject
        return subject

    def remove_subject(self, subject_id: str) -> int:
        """
        Removes a subject from the registry.
        Returns the latent_token_idx of the removed subject for pool deallocation.
        Raises KeyError if subject_id not found.
        """
        if subject_id not in self._registry:
            raise KeyError(f"Subject with ID '{subject_id}' not found.")
        
        latent_idx = self._registry[subject_id].latent_token_idx
        del self._registry[subject_id]
        return latent_idx

    def update_subject_state(
        self,
        subject_id: str,
        bounding_box: Optional[List[float]] = None,
        action_queue: Optional[List[str]] = None,
        identity_features: Optional[torch.Tensor] = None,
    ):
        """
        Updates the state of an existing subject.
        Raises KeyError if subject_id not found.
        """
        if subject_id not in self._registry:
            raise KeyError(f"Subject with ID '{subject_id}' not found.")
        
        self._registry[subject_id].update_state(
            bounding_box=bounding_box,
            action_queue=action_queue,
            identity_features=identity_features,
        )

    def get_subject_info(self, subject_id: str) -> Optional[Subject]:
        """
        Retrieves the Subject object for a given ID, or None if not found.
        """
        return self._registry.get(subject_id)

    def get_all_subject_ids(self) -> List[str]:
        """
        Returns a list of all active subject IDs.
        """
        return list(self._registry.keys())

    def get_all_subjects(self) -> List[Subject]:
        """
        Returns a list of all active Subject objects.
        """
        return list(self._registry.values())

    def __len__(self):
        return len(self._registry)

    def __contains__(self, subject_id: str) -> bool:
        return subject_id in self._registry


# Latent Vector Pool
class LatentVectorPool:
    """
    Manages a shared pool of dedicated latent vectors (subject state tokens).
    Uses a free-list strategy for efficient allocation and deallocation.
    """
    def __init__(self, latent_dim: int, max_subjects: int = 10, device: str = 'cpu'):
        if not isinstance(latent_dim, int) or latent_dim <= 0:
            raise ValueError("Latent dimension must be a positive integer.")
        if not isinstance(max_subjects, int) or max_subjects <= 0:
            raise ValueError("Max subjects must be a positive integer.")
        if device not in ['cpu', 'cuda']:
            raise ValueError("Device must be 'cpu' or 'cuda'.")

        self._latent_dim = latent_dim
        self._max_size = max_subjects
        self._device = device
        
        # Initialize the pool tensor with zeros.
        self._pool_tensor = torch.zeros(self._max_size, self._latent_dim, device=self._device, dtype=torch.float32)
        
        # Free list for available indices
        self._free_indices: Set[int] = set(range(self._max_size))
        # Keep track of currently allocated indices
        self._allocated_indices: Set[int] = set()

    def allocate_latent(self) -> int:
        """
        Allocates a free latent vector slot and returns its index.
        Raises RuntimeError if the pool is full.
        """
        if not self._free_indices:
            raise RuntimeError("Latent vector pool is full, cannot allocate new latent.")
        
        index = self._free_indices.pop()
        self._allocated_indices.add(index)
        # Optionally, initialize the newly allocated latent here (e.g., with noise or learned embedding)
        # self._pool_tensor[index] = torch.randn(self._latent_dim, device=self._device, dtype=torch.float32) * 0.02
        return index

    def release_latent(self, index: int):
        """
        Releases an allocated latent vector slot, making it available for reuse.
        Zeros out the released latent vector for cleanliness.
        Raises IndexError if index is out of bounds.
        Raises ValueError if index is not currently allocated.
        """
        if not isinstance(index, int) or index < 0 or index >= self._max_size:
            raise IndexError(f"Latent index {index} is out of bounds [0, {self._max_size-1}].")
        if index not in self._allocated_indices:
            raise ValueError(f"Latent index {index} is not currently allocated.")
        
        self._allocated_indices.remove(index)
        self._free_indices.add(index)
        # Zero out the released latent vector
        self._pool_tensor[index].zero_()

    def get_latent_vector(self, index: int) -> torch.Tensor:
        """
        Retrieves the latent vector at the given index.
        Raises IndexError if index is out of bounds.
        Raises ValueError if index is not currently allocated.
        """
        if not isinstance(index, int) or index < 0 or index >= self._max_size:
            raise IndexError(f"Latent index {index} is out of bounds [0, {self._max_size-1}].")
        if index not in self._allocated_indices:
            raise ValueError(f"Latent index {index} is not currently allocated.")
        return self._pool_tensor[index]

    def update_latent_vector(self, index: int, new_latent: torch.Tensor):
        """
        Updates the latent vector at the given index with a new tensor.
        Raises IndexError if index is out of bounds.
        Raises ValueError if index is not currently allocated or new_latent has wrong shape/type.
        """
        if not isinstance(index, int) or index < 0 or index >= self._max_size:
            raise IndexError(f"Latent index {index} is out of bounds [0, {self._max_size-1}].")
        if index not in self._allocated_indices:
            raise ValueError(f"Latent index {index} is not currently allocated.")
        if not isinstance(new_latent, torch.Tensor) or new_latent.shape[-1] != self._latent_dim:
            raise ValueError(f"New latent must be a torch.Tensor of shape (..., {self._latent_dim}). Got {new_latent.shape}.")
        
        # Ensure tensor is on the correct device and has compatible dtype
        self._pool_tensor[index] = new_latent.to(self._device, dtype=torch.float32)

    def get_all_allocated_latents(self) -> torch.Tensor:
        """
        Returns a tensor containing all currently allocated latent vectors,
        ordered by their index.
        Returns an empty tensor if no latents are allocated.
        """
        if not self._allocated_indices:
            return torch.empty(0, self._latent_dim, device=self._device, dtype=torch.float32)
        
        sorted_indices = sorted(list(self._allocated_indices))
        return self._pool_tensor[sorted_indices]

    def get_allocated_indices(self) -> Set[int]:
        """
        Returns a copy of the set of currently allocated latent indices.
        """
        return self._allocated_indices.copy()
    
    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    @property
    def max_subjects(self) -> int:
        return self._max_size
    
    @property
    def num_allocated(self) -> int:
        return len(self._allocated_indices)

    def __len__(self):
        return len(self._allocated_indices)


# Multi-Subject Latent Manager (MSLM)
class MultiSubjectLatentManager:
    """
    Orchestrates the lifecycle and state of multiple subjects within a video sequence.
    It acts as the central hub for managing subject-specific information and their 
    corresponding latent representations using a SubjectRegistry and a LatentVectorPool.
    """
    def __init__(self, latent_dim: int, max_subjects: int = 10, device: str = 'cpu'):
        self._subject_registry = SubjectRegistry()
        self._latent_pool = LatentVectorPool(latent_dim, max_subjects, device)
        self._latent_dim = latent_dim
        self._max_subjects = max_subjects
        self._device = device

    def add_subject(
        self,
        subject_id: Optional[str] = None,
        bounding_box: List[float] = [0.0, 0.0, 1.0, 1.0], # Default to full frame [x, y, w, h]
        action_queue: Optional[List[str]] = None,
        identity_features: Optional[torch.Tensor] = None,
    ) -> str:
        """
        Adds a new subject to the manager.
        If subject_id is None, a new UUID will be generated.
        Returns the ID of the added subject.
        Raises ValueError if subject_id already exists or pool is full.
        """
        if subject_id is None:
            subject_id = str(uuid.uuid4())
        
        if self._subject_registry.__contains__(subject_id):
            raise ValueError(f"Subject with ID '{subject_id}' already exists.")

        latent_token_idx = -1 # Initialize to an invalid index
        try:
            latent_token_idx = self._latent_pool.allocate_latent()
            self._subject_registry.add_subject(
                subject_id=subject_id,
                bounding_box=bounding_box,
                latent_token_idx=latent_token_idx,
                action_queue=action_queue,
                identity_features=identity_features,
            )
            return subject_id
        except RuntimeError as e:
            # Latent pool full
            raise RuntimeError(f"Failed to add subject '{subject_id}': {e}") from e
        except Exception as e:
            # Catch other potential errors during subject creation
            if latent_token_idx != -1: # If allocation happened but registry failed
                self._latent_pool.release_latent(latent_token_idx) # Rollback latent allocation
            raise ValueError(f"Error adding subject '{subject_id}': {e}") from e

    def remove_subject(self, subject_id: str):
        """
        Removes a subject from the manager and releases its latent vector.
        Raises KeyError if subject_id not found.
        """
        try:
            latent_token_idx = self._subject_registry.remove_subject(subject_id)
            self._latent_pool.release_latent(latent_token_idx)
        except KeyError as e:
            raise KeyError(f"Failed to remove subject '{subject_id}': {e}") from e
        except Exception as e:
            # This could happen if registry and pool become out of sync, though unlikely
            raise RuntimeError(f"Error removing subject '{subject_id}': {e}") from e


    def update_subject_state(
        self,
        subject_id: str,
        bounding_box: Optional[List[float]] = None,
        action_queue: Optional[List[str]] = None,
        identity_features: Optional[torch.Tensor] = None,
    ):
        """
        Updates the non-latent state (e.g., coordinates, actions) of an existing subject.
        Raises KeyError if subject_id not found.
        Raises ValueError for invalid update parameters.
        """
        try:
            self._subject_registry.update_subject_state(
                subject_id=subject_id,
                bounding_box=bounding_box,
                action_queue=action_queue,
                identity_features=identity_features,
            )
        except (KeyError, ValueError) as e:
            raise e # Re-raise directly as they are informative errors
        except Exception as e:
            raise RuntimeError(f"Error updating subject state for '{subject_id}': {e}") from e


    def get_subject_info(self, subject_id: str) -> Optional[Subject]:
        """
        Retrieves the Subject object containing its state.
        Returns None if subject_id not found.
        """
        return self._subject_registry.get_subject_info(subject_id)

    def get_subject_latent(self, subject_id: str) -> torch.Tensor:
        """
        Retrieves the latent vector associated with a specific subject.
        Raises KeyError if subject_id not found.
        Raises ValueError if latent index is invalid (should not happen if sync is maintained).
        """
        subject = self._subject_registry.get_subject_info(subject_id)
        if subject is None:
            raise KeyError(f"Subject with ID '{subject_id}' not found.")
        
        return self._latent_pool.get_latent_vector(subject.latent_token_idx)

    def update_subject_latent(self, subject_id: str, new_latent_vector: torch.Tensor):
        """
        Updates the latent vector for a specific subject in the pool.
        Raises KeyError if subject_id not found.
        Raises ValueError for invalid new_latent_vector.
        """
        subject = self._subject_registry.get_subject_info(subject_id)
        if subject is None:
            raise KeyError(f"Subject with ID '{subject_id}' not found.")
        
        try:
            self._latent_pool.update_latent_vector(subject.latent_token_idx, new_latent_vector)
        except ValueError as e:
            raise ValueError(f"Failed to update latent for subject '{subject_id}': {e}") from e
        except Exception as e:
            raise RuntimeError(f"Error updating subject latent for '{subject_id}': {e}") from e


    def get_all_active_subjects_data(self) -> List[Dict[str, Any]]:
        """
        Retrieves a consolidated list of data for all active subjects,
        including their current state and latent vector.
        Each item in the list is a dictionary containing:
        'id', 'bounding_box', 'action_queue', 'latent_token_idx',
        'latent_vector', and 'identity_features'.
        """
        all_subjects_data = []
        for subject in self._subject_registry.get_all_subjects():
            try:
                latent_vector = self._latent_pool.get_latent_vector(subject.latent_token_idx)
                all_subjects_data.append({
                    "id": subject.id,
                    "bounding_box": subject.bounding_box,
                    "action_queue": subject.action_queue,
                    "latent_token_idx": subject.latent_token_idx,
                    "latent_vector": latent_vector,
                    "identity_features": subject.identity_features,
                })
            except (KeyError, ValueError) as e:
                # This indicates an inconsistency between registry and pool,
                # e.g., a subject exists in registry but its latent index is invalid/unallocated.
                print(f"Warning: Subject '{subject.id}' found in registry but failed to retrieve latent: {e}. Skipping.")
        return all_subjects_data

    @property
    def num_active_subjects(self) -> int:
        """
        Returns the current number of active subjects.
        """
        return len(self._subject_registry)
    
    @property
    def latent_dim(self) -> int:
        """
        Returns the dimension of each latent vector in the pool.
        """
        return self._latent_dim

    @property
    def max_subjects(self) -> int:
        """
        Returns the maximum number of subjects the manager can handle.
        """
        return self._max_subjects
```