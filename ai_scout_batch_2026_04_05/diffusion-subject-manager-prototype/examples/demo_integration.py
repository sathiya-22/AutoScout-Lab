import torch
import uuid
import random
from collections import deque

class Subject:
    def __init__(self, subject_id, bounding_box, latent_token_idx, actions=None, identity_features=None):
        if not isinstance(subject_id, (str, uuid.UUID)):
            raise TypeError("subject_id must be a string or UUID.")
        if not isinstance(bounding_box, (list, tuple)) or len(bounding_box) != 4:
            raise ValueError("bounding_box must be a list/tuple of 4 elements [x, y, w, h].")
        if not isinstance(latent_token_idx, int) or latent_token_idx < 0:
            raise ValueError("latent_token_idx must be a non-negative integer.")

        self.ID = str(subject_id)
        self.bounding_box = list(bounding_box)
        self.action_queue = deque(actions if actions is not None else [])
        self.latent_token_idx = latent_token_idx
        self.identity_features = identity_features

    def update_state(self, bounding_box=None, actions=None):
        if bounding_box is not None:
            if not isinstance(bounding_box, (list, tuple)) or len(bounding_box) != 4:
                raise ValueError("bounding_box must be a list/tuple of 4 elements [x, y, w, h].")
            self.bounding_box = list(bounding_box)
        if actions is not None:
            if not isinstance(actions, (list, tuple)):
                raise ValueError("actions must be a list or tuple.")
            self.action_queue.extend(actions)

    def get_current_action(self):
        return self.action_queue[0] if self.action_queue else None

    def pop_action(self):
        if self.action_queue:
            return self.action_queue.popleft()
        return None

    def __repr__(self):
        return (f"Subject(ID='{self.ID}', bbox={self.bounding_box}, "
                f"actions={list(self.action_queue)}, latent_idx={self.latent_token_idx})")


class MultiSubjectLatentManager:
    def __init__(self, latent_dim, max_subjects=10):
        if not isinstance(latent_dim, int) or latent_dim <= 0:
            raise ValueError("latent_dim must be a positive integer.")
        if not isinstance(max_subjects, int) or max_subjects <= 0:
            raise ValueError("max_subjects must be a positive integer.")

        self.latent_dim = latent_dim
        self.max_subjects = max_subjects
        self.subject_registry = {}
        self.latent_pool = torch.randn(max_subjects, latent_dim)
        self.latent_pool_mask = torch.zeros(max_subjects, dtype=torch.bool)

    def _allocate_latent_slot(self):
        free_indices = torch.where(self.latent_pool_mask == False)[0]
        if free_indices.numel() == 0:
            raise RuntimeError("Latent vector pool is full. Cannot add more subjects.")
        idx = free_indices[0].item()
        self.latent_pool_mask[idx] = True
        return idx

    def _free_latent_slot(self, idx):
        if idx < 0 or idx >= self.max_subjects:
            raise IndexError(f"Latent slot index {idx} out of bounds.")
        if not self.latent_pool_mask[idx]:
            print(f"Warning: Attempted to free an already free latent slot {idx}.")
        self.latent_pool_mask[idx] = False
        self.latent_pool[idx].zero_()

    def add_subject(self, subject_id, initial_bbox, actions=None, identity_features=None):
        if subject_id in self.subject_registry:
            raise ValueError(f"Subject with ID '{subject_id}' already exists.")
        
        try:
            latent_token_idx = self._allocate_latent_slot()
            subject = Subject(subject_id, initial_bbox, latent_token_idx, actions, identity_features)
            self.subject_registry[subject_id] = subject
            print(f"Added subject '{subject_id}' at latent slot {latent_token_idx}.")
            return latent_token_idx
        except RuntimeError as e:
            print(f"Error adding subject '{subject_id}': {e}")
            raise

    def remove_subject(self, subject_id):
        if subject_id not in self.subject_registry:
            raise ValueError(f"Subject with ID '{subject_id}' not found.")
        
        subject = self.subject_registry.pop(subject_id)
        self._free_latent_slot(subject.latent_token_idx)
        print(f"Removed subject '{subject_id}' and freed latent slot {subject.latent_token_idx}.")

    def update_subject_state(self, subject_id, bounding_box=None, actions=None):
        if subject_id not in self.subject_registry:
            raise ValueError(f"Subject with ID '{subject_id}' not found.")
        self.subject_registry[subject_id].update_state(bounding_box, actions)

    def get_subject_info(self, subject_id):
        return self.subject_registry.get(subject_id)

    def get_all_subject_states(self):
        states = []
        for subject_id, subject in self.subject_registry.items():
            latent_token = self.latent_pool[subject.latent_token_idx]
            states.append({
                "ID": subject.ID,
                "bounding_box": subject.bounding_box,
                "latent_token": latent_token,
                "latent_token_idx": subject.latent_token_idx,
                "current_action": subject.get_current_action()
            })
        return states

    def get_latent_pool(self):
        return self.latent_pool


class SpatialBiasingAdapter:
    def __init__(self, latent_dim, feature_map_dim, num_attention_heads=8):
        if not isinstance(latent_dim, int) or latent_dim <= 0:
            raise ValueError("latent_dim must be a positive integer.")
        if not isinstance(feature_map_dim, int) or feature_map_dim <= 0:
            raise ValueError("feature_map_dim (resolution) must be a positive integer.")
        if not isinstance(num_attention_heads, int) or num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be a positive integer.")
        
        self.latent_dim = latent_dim
        self.feature_map_dim = feature_map_dim
        self.num_attention_heads = num_attention_heads
        
        self.query_proj = torch.nn.Linear(latent_dim, latent_dim)
        self.key_proj = torch.nn.Linear(latent_dim, latent_dim)
        self.value_proj = torch.nn.Linear(latent_dim, latent_dim)
        
        self.feature_map_proj = torch.nn.Conv2d(512, latent_dim, kernel_size=1)
        self.output_projection_for_addition = torch.nn.Conv2d(self.latent_dim, 512, kernel_size=1)

    def _generate_spatial_weights(self, subject_bboxes, feature_map_shape, device):
        batch_size, channels, H, W = feature_map_shape
        spatial_weights = torch.zeros(batch_size, len(subject_bboxes), H, W, device=device)

        for i, bbox in enumerate(subject_bboxes):
            x_min, y_min, w, h = bbox
            x_min_scaled = int(x_min * W)
            y_min_scaled = int(y_min * H)
            x_max_scaled = int((x_min + w) * W)
            y_max_scaled = int((y_min + h) * H)

            x_min_scaled = max(0, min(x_min_scaled, W - 1))
            y_min_scaled = max(0, min(y_min_scaled, H - 1))
            x_max_scaled = max(0, min(x_max_scaled, W - 1))
            y_max_scaled = max(0, min(y_max_scaled, H - 1))
            
            if x_max_scaled > x_min_scaled and y_max_scaled > y_min_scaled:
                spatial_weights[0, i, y_min_scaled:y_max_scaled, x_min_scaled:x_max_scaled] = 1.0

        return spatial_weights
    
    def _inject_latents_via_cross_attention(self, feature_maps, subject_latent_states, spatial_weights):
        batch_size, C, H, W = feature_maps.shape
        device = feature_maps.device

        if not subject_latent_states:
            return feature_maps

        num_subjects = len(subject_latent_states)
        
        subject_latents_tensor = torch.stack([s['latent_token'] for s in subject_latent_states]).to(device)

        processed_feature_maps = self.feature_map_proj(feature_maps).permute(0, 2, 3, 1).reshape(batch_size, H*W, self.latent_dim)
        
        Q = self.query_proj(processed_feature_maps)
        K = self.key_proj(subject_latents_tensor)
        V = self.value_proj(subject_latents_tensor)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.latent_dim ** 0.5)

        spatial_weights_reshaped = spatial_weights.permute(0, 2, 3, 1).reshape(batch_size, H*W, num_subjects)
        
        attention_scores = attention_scores + (torch.log(spatial_weights_reshaped + 1e-6))
        
        attention_weights = torch.softmax(attention_scores, dim=-1)

        attended_values = torch.matmul(attention_weights, V)

        biased_features = attended_values.reshape(batch_size, H, W, self.latent_dim).permute(0, 3, 1, 2)

        self.output_projection_for_addition.to(device)
        return feature_maps + self.output_projection_for_addition(biased_features)

    def apply_biasing(self, unet_feature_maps, subject_states, latent_pool_tensor):
        if not isinstance(unet_feature_maps, torch.Tensor):
            raise TypeError("unet_feature_maps must be a torch.Tensor.")
        if not isinstance(subject_states, list):
            raise TypeError("subject_states must be a list.")
        if not isinstance(latent_pool_tensor, torch.Tensor):
            raise TypeError("latent_pool_tensor must be a torch.Tensor.")

        if not subject_states:
            print("No active subjects found. Skipping spatial biasing.")
            return unet_feature_maps

        subject_bboxes = [s['bounding_box'] for s in subject_states]
        
        device = unet_feature_maps.device

        spatial_weights = self._generate_spatial_weights(subject_bboxes, unet_feature_maps.shape, device)

        biased_feature_maps = self._inject_latents_via_cross_attention(
            unet_feature_maps, subject_states, spatial_weights
        )

        print(f"Applied spatial biasing for {len(subject_states)} subjects.")
        return biased_feature_maps


class MockDiffusionUNet:
    def __init__(self, in_channels, out_channels, feature_map_res=(64, 64)):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_map_res = feature_map_res
        self.conv1 = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def __call__(self, x):
        print(f"    Mock U-Net received feature maps of shape: {x.shape}")
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        print(f"    Mock U-Net output feature maps of shape: {x.shape}")
        return x

def main():
    print("--- Starting Demo Integration of Multi-Subject Latent Manager and Spatial Biasing Adapter ---")

    LATENT_DIM = 1024
    MAX_SUBJECTS = 3
    UNET_FEATURE_MAP_CHANNELS = 512
    UNET_FEATURE_MAP_RESOLUTION = (64, 64)
    NUM_DIFFUSION_STEPS = 5

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on device: {device}")

    print("\n1. Initializing MultiSubjectLatentManager and SpatialBiasingAdapter...")
    try:
        mslm = MultiSubjectLatentManager(latent_dim=LATENT_DIM, max_subjects=MAX_SUBJECTS)
        sba = SpatialBiasingAdapter(
            latent_dim=LATENT_DIM,
            feature_map_dim=UNET_FEATURE_MAP_RESOLUTION[0],
            num_attention_heads=8
        )
        mock_unet = MockDiffusionUNet(
            in_channels=UNET_FEATURE_MAP_CHANNELS,
            out_channels=UNET_FEATURE_MAP_CHANNELS,
            feature_map_res=UNET_FEATURE_MAP_RESOLUTION
        )
    except (ValueError, RuntimeError) as e:
        print(f"Initialization error: {e}")
        return

    print("\n2. Simulating initial subject detection...")
    initial_subjects_data = [
        {"id": "person_A", "bbox": [0.1, 0.1, 0.2, 0.3], "actions": ["walking"]},
        {"id": "person_B", "bbox": [0.7, 0.6, 0.15, 0.25], "actions": ["waving"]},
    ]

    for data in initial_subjects_data:
        try:
            mslm.add_subject(data["id"], data["bbox"], data["actions"])
        except (ValueError, RuntimeError) as e:
            print(f"Could not add subject {data['id']}: {e}")

    try:
        mslm.add_subject("person_C", [0.4, 0.4, 0.1, 0.2], ["sitting"])
    except (ValueError, RuntimeError) as e:
        print(f"Could not add person_C: {e}")

    print("\nAttempting to add a subject beyond MAX_SUBJECTS...")
    try:
        mslm.add_subject("person_D", [0.0, 0.0, 0.1, 0.1], ["standing"])
    except (ValueError, RuntimeError) as e:
        print(f"Successfully caught expected error: {e}")

    print(f"\n3. Simulating diffusion process over {NUM_DIFFUSION_STEPS} steps...")
    current_unet_features = torch.randn(
        1, UNET_FEATURE_MAP_CHANNELS, *UNET_FEATURE_MAP_RESOLUTION,
        device=device
    )

    for step in range(NUM_DIFFUSION_STEPS):
        print(f"\n--- Diffusion Step {step + 1}/{NUM_DIFFUSION_STEPS} ---")

        if step == 1:
            print("  Updating person_A's bounding box and adding action 'running'.")
            try:
                mslm.update_subject_state("person_A", bounding_box=[0.15, 0.15, 0.25, 0.35], actions=["running"])
            except ValueError as e:
                print(f"  Error updating person_A: {e}")
        if step == 2:
            print("  Removing person_B from the scene.")
            try:
                mslm.remove_subject("person_B")
            except ValueError as e:
                print(f"  Error removing person_B: {e}")
            print("  Attempting to add person_D again after freeing a slot...")
            try:
                mslm.add_subject("person_D", [0.05, 0.05, 0.1, 0.1], ["standing"])
            except (ValueError, RuntimeError) as e:
                print(f"  Could not add person_D: {e}")

        active_subject_states = mslm.get_all_subject_states()
        full_latent_pool = mslm.get_latent_pool().to(device)

        if not active_subject_states:
            print("  No active subjects to bias. Proceeding with raw U-Net features.")
            biased_features = current_unet_features
        else:
            print(f"  Active subjects: {[s['ID'] for s in active_subject_states]}")
            try:
                biased_features = sba.apply_biasing(current_unet_features, active_subject_states, full_latent_pool)
            except Exception as e:
                print(f"  Error applying spatial biasing: {e}")
                biased_features = current_unet_features

        unet_output_features = mock_unet(biased_features)
        
        current_unet_features = unet_output_features.detach()

    print("\n--- Demo Integration Finished ---")

if __name__ == "__main__":
    main()