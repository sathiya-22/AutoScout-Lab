```python
import torch
import torch.nn as nn
import uuid
from collections import deque
import random
import math

# --- MOCK CLASSES FOR DEPENDENCIES ---
# These are minimal implementations to allow MockUNet to run and demonstrate interaction.

class Subject:
    def __init__(self, subject_id, bounding_box, action_queue, latent_token_idx, identity_features=None):
        self.ID = subject_id
        self.bounding_box = bounding_box  # [x, y, w, h] or similar (float, normalized 0-1)
        self.action_queue = deque(action_queue) if isinstance(action_queue, list) else action_queue
        self.latent_token_idx = latent_token_idx
        self.identity_features = identity_features # Mock: torch.Tensor

    def update_state(self, bounding_box=None, action=None):
        if bounding_box is not None:
            if not (isinstance(bounding_box, (list, tuple)) and len(bounding_box) == 4):
                raise ValueError(f"Bounding box must be [x, y, w, h], got {bounding_box}")
            self.bounding_box = bounding_box
        if action is not None:
            self.action_queue.append(action)

    def __repr__(self):
        return f"Subject(ID={self.ID}, bbox={self.bounding_box}, latent_idx={self.latent_token_idx})"


class SubjectRegistry:
    def __init__(self):
        self._subjects = {} # {subject_ID: Subject}

    def add_subject(self, subject: Subject):
        if not isinstance(subject, Subject):
            raise TypeError("Expected 'subject' to be an instance of Subject.")
        if subject.ID in self._subjects:
            raise ValueError(f"Subject with ID {subject.ID} already exists.")
        self._subjects[subject.ID] = subject

    def remove_subject(self, subject_id):
        if subject_id not in self._subjects:
            raise ValueError(f"Subject with ID {subject_id} not found.")
        del self._subjects[subject_id]

    def update_subject_state(self, subject_id, bounding_box=None, action=None):
        if subject_id not in self._subjects:
            raise ValueError(f"Subject with ID {subject_id} not found.")
        self._subjects[subject_id].update_state(bounding_box, action)

    def get_subject_info(self, subject_id):
        return self._subjects.get(subject_id)

    def get_all_subjects(self):
        return list(self._subjects.values())

    def __len__(self):
        return len(self._subjects)


class LatentVectorPool:
    def __init__(self, max_pool_size, latent_dim, device='cpu'):
        if not isinstance(max_pool_size, int) or max_pool_size <= 0:
            raise ValueError("max_pool_size must be a positive integer.")
        if not isinstance(latent_dim, int) or latent_dim <= 0:
            raise ValueError("latent_dim must be a positive integer.")

        self.max_pool_size = max_pool_size
        self.latent_dim = latent_dim
        self.device = device
        self.pool = torch.zeros(max_pool_size, latent_dim, device=device)
        self._free_indices = list(range(max_pool_size))

    def allocate_latent(self) -> int:
        if not self._free_indices:
            raise RuntimeError("Latent vector pool is full. Cannot allocate new latent.")
        idx = self._free_indices.pop(0) # Use pop(0) for FIFO
        return idx

    def free_latent(self, idx: int):
        if not isinstance(idx, int) or idx < 0 or idx >= self.max_pool_size:
            raise IndexError(f"Invalid latent index: {idx}. Must be within [0, {self.max_pool_size-1}].")
        if idx in self._free_indices:
            return # Already freed or not allocated, do nothing
        self._free_indices.append(idx)
        self._free_indices.sort() # Keep sorted to make allocation more deterministic
        self.pool[idx].zero_() # Zero out the vector for cleanliness

    def get_latent_vector(self, idx: int):
        if not isinstance(idx, int) or idx < 0 or idx >= self.max_pool_size:
            raise IndexError(f"Invalid latent index: {idx}. Must be within [0, {self.max_pool_size-1}].")
        return self.pool[idx]

    def set_latent_vector(self, idx: int, vector: torch.Tensor):
        if not isinstance(idx, int) or idx < 0 or idx >= self.max_pool_size:
            raise IndexError(f"Invalid latent index: {idx}. Must be within [0, {self.max_pool_size-1}].")
        if not isinstance(vector, torch.Tensor):
            raise TypeError("Expected 'vector' to be a torch.Tensor.")
        if vector.shape != (self.latent_dim,):
            raise ValueError(f"Expected vector of shape ({self.latent_dim},), got {vector.shape}.")
        self.pool[idx] = vector.to(self.device)

    def get_all_active_latents(self, active_indices: list):
        """Returns a tensor of all currently active latents based on a list of indices."""
        if not active_indices:
            return torch.empty(0, self.latent_dim, device=self.device)
        valid_indices = sorted(list(set(idx for idx in active_indices if 0 <= idx < self.max_pool_size)))
        if not valid_indices:
            return torch.empty(0, self.latent_dim, device=self.device)
        return self.pool[torch.tensor(valid_indices, device=self.device, dtype=torch.long)]


class MultiSubjectLatentManager:
    def __init__(self, max_subjects: int = 10, latent_dim: int = 128, device='cpu'):
        if not isinstance(max_subjects, int) or max_subjects <= 0:
            raise ValueError("max_subjects must be a positive integer.")
        self.registry = SubjectRegistry()
        self.latent_pool = LatentVectorPool(max_subjects, latent_dim, device)
        self.latent_dim = latent_dim
        self.device = device

    def add_subject(self, bounding_box, action_queue=None, identity_features=None, subject_id=None):
        subject_id = subject_id if subject_id is not None else str(uuid.uuid4())
        try:
            latent_idx = self.latent_pool.allocate_latent()
            subject = Subject(subject_id, bounding_box, action_queue or deque(), latent_idx, identity_features)
            self.registry.add_subject(subject)
            initial_latent = torch.randn(self.latent_dim, device=self.device)
            if identity_features is not None:
                if identity_features.dim() == 1 and identity_features.shape[-1] == self.latent_dim:
                    initial_latent = initial_latent + identity_features
                else:
                    print(f"Warning: Identity features dim ({identity_features.shape[-1]}) mismatch with latent_dim ({self.latent_dim}) or not 1D. Using random init for latent {subject_id}.")
            self.latent_pool.set_latent_vector(latent_idx, initial_latent)
            return subject
        except (RuntimeError, ValueError) as e:
            print(f"Failed to add subject {subject_id}: {e}")
            raise
        except Exception as e:
            if 'latent_idx' in locals():
                try:
                    self.latent_pool.free_latent(latent_idx)
                except Exception as free_e:
                    print(f"Error during cleanup of latent {latent_idx}: {free_e}")
            print(f"An unexpected error occurred while adding subject {subject_id}: {e}")
            raise

    def remove_subject(self, subject_id):
        subject = self.registry.get_subject_info(subject_id)
        if subject:
            self.latent_pool.free_latent(subject.latent_token_idx)
            self.registry.remove_subject(subject_id)
        else:
            raise ValueError(f"Subject with ID {subject_id} not found.")

    def update_subject_state(self, subject_id, bounding_box=None, action=None):
        self.registry.update_subject_state(subject_id, bounding_box, action)

    def get_subject_data_for_biasing(self):
        """Returns a list of tuples: (subject_id, bbox, latent_vector) for active subjects."""
        data = []
        for subject in self.registry.get_all_subjects():
            try:
                latent_vec = self.latent_pool.get_latent_vector(subject.latent_token_idx)
                data.append((subject.ID, subject.bounding_box, latent_vec))
            except IndexError as e:
                print(f"Error retrieving latent for subject {subject.ID} (index {subject.latent_token_idx}): {e}")
        return data


class CoordinateAwareAttentionMechanism(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.spatial_proj = nn.Linear(4, feature_dim) # Project bbox [x,y,w,h] to feature dim
        self.latent_transform = nn.Linear(feature_dim, feature_dim)
        self.combine_proj = nn.Linear(feature_dim * 2, feature_dim)

    def forward(self, unet_features: torch.Tensor, subject_coords: list, subject_latents: torch.Tensor):
        B, C, H, W = unet_features.shape

        if not subject_coords:
            return torch.zeros_like(unet_features)

        spatial_biases_per_subject = []
        for bbox_raw in subject_coords:
            if not (isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 4):
                 raise ValueError(f"Bounding box must be [x, y, w, h], got {bbox_raw}")
            bbox = torch.tensor(bbox_raw, device=unet_features.device, dtype=torch.float)

            grid_y, grid_x = torch.meshgrid(torch.arange(H, device=unet_features.device),
                                            torch.arange(W, device=unet_features.device), indexing='ij')

            center_x = (bbox[0] + bbox[2] / 2) * W # Assuming bbox coords are normalized [0,1]
            center_y = (bbox[1] + bbox[3] / 2) * H

            dist_sq = (grid_y - center_y)**2 + (grid_x - center_x)**2
            sigma = (bbox[2] * W + bbox[3] * H) / 8 + 1 # Min sigma of 1 pixel
            spatial_weights = torch.exp(-dist_sq / (2 * sigma**2)).unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
            
            bbox_features = self.spatial_proj(bbox.unsqueeze(0)) # (1, feature_dim)
            spatial_bias = bbox_features.unsqueeze(-1).unsqueeze(-1) * spatial_weights
            spatial_biases_per_subject.append(spatial_bias)
        
        total_spatial_bias = torch.stack(spatial_biases_per_subject).sum(dim=0) # (1, C, H, W)
        total_spatial_bias = total_spatial_bias.expand(B, -1, -1, -1) # Expand to batch size

        if subject_latents.numel() > 0:
            combined_latent_info = self.latent_transform(subject_latents.mean(dim=0)) # (feature_dim)
        else:
            combined_latent_info = torch.zeros(self.feature_dim, device=unet_features.device)
        
        latent_biasing_map = combined_latent_info.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(B, C, H, W)

        combined_bias_flat = torch.cat([total_spatial_bias, latent_biasing_map], dim=1) # (B, 2*C, H, W)
        combined_bias_final = self.combine_proj(combined_bias_flat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        return combined_bias_final


class SpatialBiasingAdapter(nn.Module):
    def __init__(self, mslm: MultiSubjectLatentManager, feature_dim: int):
        super().__init__()
        if not isinstance(mslm, MultiSubjectLatentManager):
            raise TypeError("Expected 'mslm' to be an instance of MultiSubjectLatentManager.")
        if not isinstance(feature_dim, int) or feature_dim <= 0:
            raise ValueError("feature_dim must be a positive integer.")

        self.mslm = mslm
        self.feature_dim = feature_dim
        self.attention_mechanism = CoordinateAwareAttentionMechanism(feature_dim)
        
        if mslm.latent_dim != feature_dim:
            self.latent_proj = nn.Linear(mslm.latent_dim, feature_dim)
        else:
            self.latent_proj = nn.Identity()

    def generate_biasing_signals(self, unet_features: torch.Tensor):
        """
        Generates spatial and latent biasing signals for a specific U-Net feature map resolution.
        `unet_features`: A sample of the current U-Net feature map (B, C, H, W)
        Returns a tensor of shape (B, C, H, W) representing the biasing signal.
        """
        if not isinstance(unet_features, torch.Tensor) or unet_features.dim() != 4:
            raise ValueError("unet_features must be a 4D tensor (B, C, H, W).")

        subject_data = self.mslm.get_subject_data_for_biasing()
        if not subject_data:
            return torch.zeros_like(unet_features)

        subject_coords = [s[1] for s in subject_data]
        
        subject_latents_raw = torch.stack([s[2] for s in subject_data])
        subject_latents_projected = self.latent_proj(subject_latents_raw)

        biasing_signal = self.attention_mechanism(
            unet_features, subject_coords, subject_latents_projected
        )
        return biasing_signal


# --- MOCK U-NET BLOCKS ---

class MockAttentionBlock(nn.Module):
    """
    A mock attention block within the U-Net that can receive and apply
    biasing signals from the SpatialBiasingAdapter.
    """
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.GroupNorm(8, dim)
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.proj_out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, sba_biasing_signal: torch.Tensor = None):
        B, C, H, W = x.shape
        x_flat = self.norm(x).permute(0, 2, 3, 1).reshape(B, H*W, C) # (B, HW, C)

        q = self.query(x_flat)
        k = self.key(x_flat)
        v = self.value(x_flat)

        if sba_biasing_signal is not None:
            sba_flat = sba_biasing_signal.permute(0, 2, 3, 1).reshape(B, H*W, C)
            
            if sba_flat.shape == q.shape:
                q = q + sba_flat
                v = v * (1 + sba_flat.mean(dim=-1, keepdim=True).sigmoid())
            else:
                # This warning helps in debugging if resolution mismatches
                # print(f"Warning: SBA signal shape {sba_flat.shape} does not match attention input {q.shape}. Skipping SBA application.")
                pass

        attn_weights = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / (C**0.5), dim=-1)
        attn_output = torch.bmm(attn_weights, v)

        attn_output = self.proj_out(attn_output)

        output = x_flat + attn_output # Residual connection

        output = output.reshape(B, H, W, C).permute(0, 3, 1, 2) # Reshape back to (B, C, H, W)
        return output


class MockResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embed_dim=None):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.time_proj = nn.Linear(time_embed_dim, out_channels) if time_embed_dim is not None else None

        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.silu = nn.SiLU()

    def forward(self, x, time_embed=None):
        h = self.silu(self.norm1(x))
        h = self.conv1(h)
        
        if self.time_proj is not None and time_embed is not None:
            time_embed_mapped = self.time_proj(time_embed).unsqueeze(-1).unsqueeze(-1)
            h = h + time_embed_mapped
        
        h = self.conv2(self.silu(self.norm2(h)))
        return h + self.skip_connection(x)


class MockDownStage(nn.Module):
    """Represents a single resolution stage in the downsampling path."""
    def __init__(self, in_channels, out_channels, time_embed_dim, num_res_blocks, has_attention):
        super().__init__()
        blocks = []
        for i in range(num_res_blocks):
            blocks.append(MockResBlock(in_channels if i == 0 else out_channels, out_channels, time_embed_dim))
            if has_attention:
                blocks.append(MockAttentionBlock(out_channels))
        self.blocks = nn.ModuleList(blocks)
        
    def forward(self, x, time_embed=None, sba_biasing_signal=None):
        for block in self.blocks:
            if isinstance(block, MockResBlock):
                x = block(x, time_embed)
            elif isinstance(block, MockAttentionBlock):
                x = block(x, sba_biasing_signal)
            else:
                x = block(x)
        return x


class MockUpStage(nn.Module):
    """Represents a single resolution stage in the upsampling path."""
    def __init__(self, in_channels, out_channels, time_embed_dim, num_res_blocks, has_attention):
        super().__init__()
        blocks = []
        for i in range(num_res_blocks):
            blocks.append(MockResBlock(in_channels if i == 0 else out_channels, out_channels, time_embed_dim))
            if has_attention:
                blocks.append(MockAttentionBlock(out_channels))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, skip_x, time_embed=None, sba_biasing_signal=None):
        x = torch.cat([x, skip_x], dim=1) # Concatenate with skip connection
        for block in self.blocks:
            if isinstance(block, MockResBlock):
                x = block(x, time_embed)
            elif isinstance(block, MockAttentionBlock):
                x = block(x, sba_biasing_signal)
            else:
                x = block(x)
        return x


class MockMiddleBlock(nn.Module):
    def __init__(self, channels, time_embed_dim):
        super().__init__()
        self.res_block1 = MockResBlock(channels, channels, time_embed_dim)
        self.attention = MockAttentionBlock(channels)
        self.res_block2 = MockResBlock(channels, channels, time_embed_dim)

    def forward(self, x, time_embed=None, sba_biasing_signal=None):
        x = self.res_block1(x, time_embed)
        x = self.attention(x, sba_biasing_signal)
        x = self.res_block2(x, time_embed)
        return x


class MockUNet(nn.Module):
    def __init__(self,
                 in_channels: int = 4,
                 out_channels: int = 4,
                 base_channels: int = 64,
                 channel_multipliers: list = [1, 2, 4, 8],
                 num_res_blocks: int = 2,
                 attention_resolutions: list = [16, 8], # Resolutions at which to apply attention (e.g., 16x16, 8x8)
                 time_embed_dim: int = 256,
                 sba: SpatialBiasingAdapter = None): # The SpatialBiasingAdapter instance
        super().__init__()

        if sba is None:
            raise ValueError("SpatialBiasingAdapter instance must be provided to MockUNet.")
        self.sba = sba
        self.base_channels = base_channels
        self.channel_multipliers = channel_multipliers
        self.attention_resolutions = attention_resolutions
        self.time_embed_dim = time_embed_dim

        # Time embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 4, time_embed_dim) # Output time_embed_dim
        )
        
        # Mock text conditioning infusion (projects external text features to time_embed_dim)
        self.text_conditioning_proj = nn.Linear(sba.mslm.latent_dim, time_embed_dim) 

        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Downsampling path
        self.down_stages = nn.ModuleList()
        self.downs = nn.ModuleList() # For downsampling between stages
        current_channels = base_channels
        # Assumed initial latent resolution (e.g., 64x64 if input is 256x256 and first conv reduces by 4)
        # Or if input is already latent (e.g., from VAE) then it's the actual feature map resolution
        # For simplicity, let's assume input to first DownStage is `initial_latent_res`
        self.initial_latent_res = 64 

        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            res_at_stage = self.initial_latent_res // (2**i)
            has_attention_at_res = res_at_stage in attention_resolutions
            
            self.down_stages.append(
                MockDownStage(current_channels, out_ch, time_embed_dim, num_res_blocks, has_attention_at_res)
            )
            current_channels = out_ch
            
            if i < len(channel_multipliers) - 1: # Add downsample between stages
                self.downs.append(
                    nn.Conv2d(current_channels, current_channels, kernel_size=3, stride=2, padding=1)
                )

        # Middle block
        self.middle_block = MockMiddleBlock(current_channels, time_embed_dim)

        # Upsampling path
        self.up_stages = nn.ModuleList()
        self.ups = nn.ModuleList() # For upsampling between stages (e.g., nn.Upsample)
        
        for i in reversed(range(len(channel_multipliers))):
            mult_idx = i
            in_ch = current_channels # Features coming from the previous up-stage or middle block
            skip_ch = base_channels * channel_multipliers[mult_idx] # Features from the corresponding down-stage
            out_ch = base_channels * channel_multipliers[mult_idx] # Output channels for this up-stage

            res_at_stage = self.initial_latent_res // (2**mult_idx)
            has_attention_at_res = res_at_stage in attention_resolutions

            self.up_stages.append(
                MockUpStage(in_ch + skip_ch, out_ch, time_embed_dim, num_res_blocks, has_attention_at_res)
            )
            current_channels = out_ch # For the next up-stage

            if i > 0: # Add upsample between stages
                self.ups.append(
                    nn.Upsample(scale_factor=2, mode='nearest')
                )
        
        # Final convolution
        self.norm_out = nn.GroupNorm(8, base_channels)
        self.conv_out = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)
        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, text_conditioning: torch.Tensor = None):
        if not isinstance(x, torch.Tensor) or x.dim() != 4:
            raise ValueError("Input 'x' must be a 4D tensor (B, C, H, W).")
        if not isinstance(timestep, torch.Tensor) or timestep.dim() != 1:
            raise ValueError("Input 'timestep' must be a 1D tensor (B,).")
        if text_conditioning is not None and (not isinstance(text_conditioning, torch.Tensor) or text_conditioning.dim() != 2):
            raise ValueError("Input 'text_conditioning' must be a 2D tensor (B, embedding_dim).")


        # Time embedding
        time_embed = self.time_embedding(timestep.float())

        # Mock text conditioning infusion (add to time embedding)
        if text_conditioning is not None:
            text_embed_features = self.text_conditioning_proj(text_conditioning)
            time_embed = time_embed + text_embed_features

        # Input convolution
        x = self.conv_in(x)
        skip_connections = []

        # Downsampling path
        for i, stage in enumerate(self.down_stages):
            sba_biasing_signal = self.sba.generate_biasing_signals(x)
            x = stage(x, time_embed, sba_biasing_signal)
            skip_connections.append(x) # Save output of stage for skip connection

            if i < len(self.downs): # Apply downsample between stages
                x = self.downs[i](x)

        # Middle block
        sba_biasing_signal_mid = self.sba.generate_biasing_signals(x)
        x = self.middle_block(x, time_embed, sba_biasing_signal_mid)

        # Upsampling path
        skip_connections = skip_connections[::-1] # Reverse for upsampling path
        
        for i, stage in enumerate(self.up_stages):
            if i > 0: # Apply upsample before the stage (except for the very first up-stage)
                x = self.ups[i-1](x)
            
            skip_x = skip_connections[i] # Get corresponding skip connection

            sba_biasing_signal = self.sba.generate_biasing_signals(x) # Generate new signal for this resolution
            x = stage(x, skip_x, time_embed, sba_biasing_signal)
            
        # Final convolution
        x = self.conv_out(self.silu(self.norm_out(x)))
        return x
```