import uuid
import torch
import random
from typing import Dict, List, Any, Optional, Tuple

LATENT_DIM = 256
NUM_LATENTS_IN_POOL = 10
UNET_FEATURE_DIM = 64
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64


class Subject:
    def __init__(self, subject_id: str, bounding_box: List[float], action_queue: Optional[List[str]] = None):
        self.id = subject_id
        self.bounding_box = bounding_box
        self.spatial_coordinates = bounding_box
        self.action_queue = action_queue if action_queue is not None else []
        self.latent_token_idx: Optional[int] = None
        self.identity_features: Optional[torch.Tensor] = None

    def update_state(self, bounding_box: Optional[List[float]] = None, action_queue: Optional[List[str]] = None):
        if bounding_box is not None:
            self.bounding_box = bounding_box
            self.spatial_coordinates = bounding_box
        if action_queue is not None:
            self.action_queue = action_queue

    def __repr__(self):
        return f"Subject(ID={self.id[:8]}, bbox={self.bounding_box}, latent_idx={self.latent_token_idx})"


class SubjectRegistry:
    def __init__(self):
        self.subjects: Dict[str, Subject] = {}

    def add_subject(self, subject: Subject):
        if subject.id in self.subjects:
            raise ValueError(f"Subject with ID {subject.id} already exists.")
        self.subjects[subject.id] = subject

    def remove_subject(self, subject_id: str):
        if subject_id not in self.subjects:
            raise ValueError(f"Subject with ID {subject_id} not found.")
        del self.subjects[subject_id]

    def update_subject_state(self, subject_id: str, bounding_box: Optional[List[float]] = None, action_queue: Optional[List[str]] = None):
        subject = self.get_subject(subject_id)
        if subject:
            subject.update_state(bounding_box, action_queue)
        else:
            raise ValueError(f"Subject with ID {subject_id} not found for update.")

    def get_subject(self, subject_id: str) -> Optional[Subject]:
        return self.subjects.get(subject_id)

    def get_all_subjects(self) -> List[Subject]:
        return list(self.subjects.values())


class LatentVectorPool:
    def __init__(self, latent_dim: int, max_pool_size: int = NUM_LATENTS_IN_POOL):
        self.latent_dim = latent_dim
        self.max_pool_size = max_pool_size
        self.pool: torch.Tensor = torch.randn(max_pool_size, latent_dim)
        self.allocated_indices: Dict[int, str] = {}
        self.free_indices: List[int] = list(range(max_pool_size))

    def allocate_latent(self, subject_id: str) -> int:
        if not self.free_indices:
            raise RuntimeError("Latent vector pool is full. Cannot allocate more latents.")
        
        idx = self.free_indices.pop(0)
        self.allocated_indices[idx] = subject_id
        return idx

    def deallocate_latent(self, idx: int):
        if idx not in self.allocated_indices:
            raise ValueError(f"Latent index {idx} not allocated or already deallocated.")
        
        subject_id = self.allocated_indices.pop(idx)
        self.free_indices.append(idx)
        self.free_indices.sort()
        self.pool[idx] = torch.randn(self.latent_dim)

    def get_latent_by_index(self, idx: int) -> torch.Tensor:
        if idx not in self.allocated_indices:
            raise ValueError(f"Latent index {idx} is not currently allocated to a subject.")
        return self.pool[idx]

    def get_latents_by_indices(self, indices: List[int]) -> torch.Tensor:
        if not indices:
            return torch.empty(0, self.latent_dim)
        
        valid_indices = [i for i in indices if i in self.allocated_indices]
        
        if not valid_indices:
            return torch.empty(0, self.latent_dim)
            
        return self.pool[valid_indices]
    
    def update_latent(self, idx: int, new_latent: torch.Tensor):
        if idx not in self.allocated_indices:
            raise ValueError(f"Latent index {idx} not allocated for update.")
        if new_latent.shape != (self.latent_dim,):
            raise ValueError(f"New latent shape {new_latent.shape} does not match pool dim {self.latent_dim}.")
        self.pool[idx] = new_latent


class MultiSubjectLatentManager:
    def __init__(self, latent_dim: int = LATENT_DIM, max_subjects: int = NUM_LATENTS_IN_POOL):
        self.registry = SubjectRegistry()
        self.latent_pool = LatentVectorPool(latent_dim, max_pool_size=max_subjects)

    def add_subject(self, subject_id: str, bounding_box: List[float], action_queue: Optional[List[str]] = None):
        if self.registry.get_subject(subject_id):
            return

        try:
            latent_idx = self.latent_pool.allocate_latent(subject_id)
            subject = Subject(subject_id, bounding_box, action_queue)
            subject.latent_token_idx = latent_idx
            self.registry.add_subject(subject)
        except RuntimeError as e:
            print(f"Error adding subject {subject_id[:8]}: {e}")

    def remove_subject(self, subject_id: str):
        subject = self.registry.get_subject(subject_id)
        if subject:
            if subject.latent_token_idx is not None:
                self.latent_pool.deallocate_latent(subject.latent_token_idx)
            self.registry.remove_subject(subject_id)

    def update_subject_state(self, subject_id: str, bounding_box: Optional[List[float]] = None, action_queue: Optional[List[str]] = None):
        self.registry.update_subject_state(subject_id, bounding_box, action_queue)

    def get_subject_info(self, subject_id: str) -> Optional[Subject]:
        return self.registry.get_subject(subject_id)

    def get_all_subject_ids(self) -> List[str]:
        return list(self.registry.subjects.keys())

    def get_subject_latents_and_coords(self) -> Tuple[List[str], torch.Tensor, List[List[float]]]:
        active_subjects = self.registry.get_all_subjects()
        if not active_subjects:
            return [], torch.empty(0, self.latent_pool.latent_dim), []
        
        filtered_subjects = [s for s in active_subjects if s.latent_token_idx is not None]
        
        if not filtered_subjects:
            return [], torch.empty(0, self.latent_pool.latent_dim), []

        aligned_subject_ids = [s.id for s in filtered_subjects]
        aligned_latent_indices = [s.latent_token_idx for s in filtered_subjects if s.latent_token_idx is not None]
        aligned_spatial_coords = [s.spatial_coordinates for s in filtered_subjects]

        latent_tokens = self.latent_pool.get_latents_by_indices(aligned_latent_indices)
        
        return aligned_subject_ids, latent_tokens, aligned_spatial_coords

    def update_latent_token(self, subject_id: str, new_latent: torch.Tensor):
        subject = self.registry.get_subject(subject_id)
        if subject and subject.latent_token_idx is not None:
            self.latent_pool.update_latent(subject.latent_token_idx, new_latent)


class SpatialBiasingAdapter:
    def __init__(self, unet_feature_dim: int = UNET_FEATURE_DIM, latent_dim: int = LATENT_DIM):
        self.unet_feature_dim = unet_feature_dim
        self.latent_dim = latent_dim
        self.query_proj = torch.nn.Linear(unet_feature_dim, latent_dim)
        self.key_proj = torch.nn.Linear(latent_dim, latent_dim)
        self.value_proj = torch.nn.Linear(latent_dim, unet_feature_dim)
        self.gamma_proj = torch.nn.Linear(latent_dim, unet_feature_dim)
        self.beta_proj = torch.nn.Linear(latent_dim, unet_feature_dim)

    def apply_biasing(self, unet_features: torch.Tensor, subject_ids: List[str], 
                      latent_tokens: torch.Tensor, spatial_coordinates: List[List[float]]) -> torch.Tensor:
        if not subject_ids:
            return unet_features
        
        batch_size, channels, H, W = unet_features.shape

        attention_mask = torch.zeros_like(unet_features)
        for i, (coords, sub_id) in enumerate(zip(spatial_coordinates, subject_ids)):
            x, y, w, h = [int(c) for c in coords]
            x_min = max(0, x)
            y_min = max(0, y)
            x_max = min(W, x + w)
            y_max = min(H, y + h)

            if x_max > x_min and y_max > y_min:
                attention_mask[0, :, y_min:y_max, x_min:x_max] = 1.0
        
        aggregated_latent = torch.mean(latent_tokens, dim=0)
        gamma = self.gamma_proj(aggregated_latent).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta_proj(aggregated_latent).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        bias_tensor = gamma * attention_mask + beta 
        
        biased_features = unet_features + bias_tensor * 0.5

        return biased_features


def mock_unet_forward(noise_latent: torch.Tensor, timestep: int, conditioned_features: torch.Tensor) -> torch.Tensor:
    output_latent = noise_latent + conditioned_features.mean(dim=(1,2,3)) * 0.1
    return output_latent


class MockDiffusionPipeline:
    def __init__(self,
                 latent_dim: int = LATENT_DIM,
                 unet_feature_dim: int = UNET_FEATURE_DIM,
                 max_subjects: int = NUM_LATENTS_IN_POOL,
                 image_height: int = IMAGE_HEIGHT,
                 image_width: int = IMAGE_WIDTH):
        
        self.mslm = MultiSubjectLatentManager(latent_dim=latent_dim, max_subjects=max_subjects)
        self.sba = SpatialBiasingAdapter(unet_feature_dim=unet_feature_dim, latent_dim=latent_dim)
        
        self.latent_dim = latent_dim
        self.unet_feature_dim = unet_feature_dim
        self.image_height = image_height
        self.image_width = image_width

    def run(self, 
            initial_noise_latent: torch.Tensor,
            num_timesteps: int = 50,
            mock_subject_detection_data: List[Dict[str, Any]] = None):
        
        current_latent = initial_noise_latent
        
        if mock_subject_detection_data is None:
            mock_subject_detection_data = []

        processed_subject_ids = set()
        
        for t in range(num_timesteps):
            current_timestep_updates = [data for data in mock_subject_detection_data if data.get('timestep') == t]
            
            for update_data in current_timestep_updates:
                if 'add' in update_data:
                    sub_id = update_data['add']['id']
                    bbox = update_data['add']['bbox']
                    actions = update_data['add'].get('actions', [])
                    self.mslm.add_subject(sub_id, bbox, actions)
                    processed_subject_ids.add(sub_id)
                elif 'update' in update_data:
                    sub_id = update_data['update']['id']
                    bbox = update_data['update'].get('bbox')
                    actions = update_data['update'].get('actions')
                    if sub_id in processed_subject_ids:
                         self.mslm.update_subject_state(sub_id, bbox, actions)
                elif 'remove' in update_data:
                    sub_id = update_data['remove']['id']
                    if sub_id in processed_subject_ids:
                        self.mslm.remove_subject(sub_id)
                        processed_subject_ids.discard(sub_id)

            subject_ids, latent_tokens_for_sba, spatial_coords_for_sba = self.mslm.get_subject_latents_and_coords()
            
            mock_unet_features = torch.randn(1, self.unet_feature_dim, self.image_height, self.image_width)

            biased_unet_features = self.sba.apply_biasing(
                mock_unet_features, subject_ids, latent_tokens_for_sba, spatial_coords_for_sba
            )
            
            predicted_noise_or_sample = mock_unet_forward(current_latent, t, biased_unet_features)

            current_latent = current_latent - predicted_noise_or_sample * 0.1 

            if subject_ids:
                for i, sub_id in enumerate(subject_ids):
                    refined_token = latent_tokens_for_sba[i] + torch.randn(self.latent_dim) * 0.01
                    self.mslm.update_latent_token(sub_id, refined_token)

        return current_latent


if __name__ == "__main__":
    initial_noise = torch.randn(1, LATENT_DIM)

    sub1_id = str(uuid.uuid4())
    sub2_id = str(uuid.uuid4())
    sub3_id = str(uuid.uuid4())

    mock_detection_data = [
        {'timestep': 0, 'add': {'id': sub1_id, 'bbox': [10, 10, 20, 30], 'actions': ['standing']}},
        {'timestep': 0, 'add': {'id': sub2_id, 'bbox': [40, 5, 15, 25], 'actions': ['sitting']}},
        
        {'timestep': 5, 'update': {'id': sub1_id, 'bbox': [12, 11, 22, 32], 'actions': ['walking_left']}},
        {'timestep': 5, 'add': {'id': sub3_id, 'bbox': [50, 40, 10, 20], 'actions': ['observing']}},
        
        {'timestep': 10, 'update': {'id': sub2_id, 'bbox': [45, 8, 16, 26], 'actions': ['waving']}},
        
        {'timestep': 15, 'remove': {'id': sub1_id}},
        
        {'timestep': 20, 'update': {'id': sub3_id, 'bbox': [55, 42, 11, 21], 'actions': ['walking_right']}},
        {'timestep': 25, 'add': {'id': sub1_id, 'bbox': [5, 5, 20, 30], 'actions': ['re-entering']}},
        
        {'timestep': 30, 'update': {'id': sub1_id, 'bbox': [8, 7, 21, 31], 'actions': ['walking_forward']}},
        {'timestep': 30, 'update': {'id': sub2_id, 'bbox': [48, 10, 17, 27], 'actions': ['standing_still']}},
        {'timestep': 30, 'update': {'id': sub3_id, 'bbox': [58, 45, 12, 22], 'actions': ['sitting_down']}},
    ]

    pipeline = MockDiffusionPipeline(image_height=IMAGE_HEIGHT, image_width=IMAGE_WIDTH)
    final_latent = pipeline.run(initial_noise, num_timesteps=40, mock_subject_detection_data=mock_detection_data)