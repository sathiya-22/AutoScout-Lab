```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SpatialBiasingAttention(nn.Module):
    """
    Implements a coordinate-aware attention mechanism for injecting spatial and subject-specific
    biases into U-Net feature maps. This module combines standard multi-head cross-attention
    between U-Net features (queries) and subject latent tokens (keys/values) with
    a spatial bias derived from subject coordinates.

    This acts as a "plug-and-play" component designed to be inserted into a diffusion model's
    U-Net to provide explicit guidance based on tracked subjects.
    """
    def __init__(self, query_dim: int, latent_dim: int, num_heads: int = 8,
                 spatial_bias_strength: float = 1.0, bbox_sigma_scale: float = 0.1,
                 coord_normalization_factor: float = 1.0):
        """
        Initializes the SpatialBiasingAttention module.

        Args:
            query_dim (int): Dimension of the input U-Net feature maps (queries).
            latent_dim (int): Dimension of the subject latent tokens (keys/values).
            num_heads (int): Number of attention heads.
            spatial_bias_strength (float): Scalar factor to control the influence of spatial bias.
                                           A higher value means stronger spatial emphasis.
                                           Set to 0.0 to effectively disable spatial biasing.
            bbox_sigma_scale (float): Multiplier for standard deviation of Gaussian bias,
                                      relative to bounding box dimensions. A smaller value
                                      makes the bias more concentrated near the center.
            coord_normalization_factor (float): Factor to normalize subject coordinates.
                                                If coordinates are already in [0,1], use 1.0.
                                                If they are pixel values (e.g., 0-511 for 512x512 image),
                                                set to the image dimension (e.g., 512.0) for proper scaling
                                                relative to the [0,1] feature map grid.
        """
        super().__init__()
        self.query_dim = query_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        if self.head_dim * num_heads != query_dim:
            raise ValueError(f"query_dim ({query_dim}) must be divisible by num_heads ({num_heads})")

        self.spatial_bias_strength = spatial_bias_strength
        self.bbox_sigma_scale = bbox_sigma_scale
        self.coord_normalization_factor = coord_normalization_factor

        self.query_proj = nn.Linear(query_dim, query_dim)
        # Key/Value projection can handle latent_dim != query_dim, but output for attention is query_dim
        self.key_proj = nn.Linear(latent_dim, query_dim)
        self.value_proj = nn.Linear(latent_dim, query_dim)
        self.out_proj = nn.Linear(query_dim, query_dim)

    def _generate_gaussian_bias_mask(self, coords: torch.Tensor, spatial_dims: tuple, device: torch.device):
        """
        Generates a Gaussian bias mask for each subject based on their bounding box.
        The mask is higher within and around the subject's bounding box center.

        Args:
            coords (torch.Tensor): Tensor of normalized bounding boxes [B, S, 4]
                                   where 4 is [x_center, y_center, width, height].
                                   Coordinates are expected to be normalized [0, 1] relative
                                   to the original image size, or will be normalized using
                                   `self.coord_normalization_factor`.
            spatial_dims (tuple): (H, W) of the feature map to generate the mask for.
            device (torch.device): Device to create tensors on.

        Returns:
            torch.Tensor: Spatial bias masks of shape [B, S, H * W].
                          Each mask is normalized to have a peak of 1 at its center.
        """
        B, S, coord_dim = coords.shape
        if coord_dim != 4:
            raise ValueError(f"Expected coords to have last dimension of 4 ([x_center, y_center, w, h]), got {coord_dim}")

        H, W = spatial_dims

        # Create a grid of pixel coordinates normalized to [0, 1]
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(0, 1, H, device=device),
            torch.linspace(0, 1, W, device=device),
            indexing='ij'
        )
        pixel_coords = torch.stack([x_coords, y_coords], dim=-1).view(1, 1, H * W, 2) # [1, 1, N, 2]

        # Normalize subject centroids and dimensions if needed
        # Assuming coords are [x_center, y_center, width, height] relative to original image [0,1]
        # and feature map grid is also [0,1].
        # If input coords are pixel values, divide by coord_normalization_factor to bring to [0,1]
        norm_coords = coords / self.coord_normalization_factor
        
        subject_centroids = norm_coords[:, :, :2].unsqueeze(2)  # [B, S, 1, 2]
        subject_dims = norm_coords[:, :, 2:].unsqueeze(2)       # [B, S, 1, 2] (width, height)

        # Calculate distances from each pixel to each subject's centroid
        # Broadcast operation: [B, S, N, 2] - [B, S, 1, 2] -> [B, S, N, 2]
        diff = pixel_coords - subject_centroids

        # Determine sigma for Gaussian based on subject dimensions
        # sigma is proportional to the smallest dimension (width or height) to ensure
        # the Gaussian is contained within the subject's approximate area.
        # Clamp to a small positive value to prevent division by zero for tiny or zero dimensions.
        sigma_x = (subject_dims[:, :, :, 0] * self.bbox_sigma_scale).clamp(min=1e-6) # [B, S, 1]
        sigma_y = (subject_dims[:, :, :, 1] * self.bbox_sigma_scale).clamp(min=1e-6) # [B, S, 1]

        # Calculate squared exponential terms for 2D Gaussian
        # exp_x: [B, S, N], exp_y: [B, S, N]
        exp_x = (diff[..., 0] / sigma_x) ** 2
        exp_y = (diff[..., 1] / sigma_y) ** 2

        # Gaussian formula: exp(-( (x-mu_x)^2 / (2*sigma_x^2) + (y-mu_y)^2 / (2*sigma_y^2) ))
        # We omit the 1/(2*pi*sigma_x*sigma_y) normalization factor as we only care about relative weights
        gaussian_map = torch.exp(-0.5 * (exp_x + exp_y)) # [B, S, N]

        return gaussian_map

    def forward(self, query_features: torch.Tensor, subject_latents: torch.Tensor,
                subject_coords: torch.Tensor, subject_mask: torch.Tensor,
                feature_map_spatial_dims: tuple):
        """
        Applies spatial and subject-latent biasing to U-Net feature maps.

        Args:
            query_features (torch.Tensor): U-Net features (queries) of shape [B, N, D_query],
                                           where N is H*W (flattened spatial dimensions),
                                           D_query is query_dim.
            subject_latents (torch.Tensor): Subject latent tokens of shape [B, S, D_latent],
                                            where S is max_subjects (padded to max for batch),
                                            D_latent is latent_dim.
            subject_coords (torch.Tensor): Normalized bounding boxes for each subject.
                                           Shape [B, S, 4], typically [x_center, y_center, width, height].
                                           Expected to be normalized [0, 1] relative to the original
                                           image, or `coord_normalization_factor` will handle conversion.
            subject_mask (torch.Tensor): Boolean mask indicating active subjects [B, S].
                                         True for active subject, False for inactive/padded subject.
            feature_map_spatial_dims (tuple): (H, W) of the original feature map spatial dimensions.

        Returns:
            torch.Tensor: Biased output features of shape [B, N, D_query].
        """
        B, N, D_query = query_features.shape
        _, S, D_latent = subject_latents.shape
        H, W = feature_map_spatial_dims

        if N != H * W:
            raise ValueError(
                f"Feature map size N ({N}) does not match H*W ({H*W}) for spatial dims ({H}, {W})."
            )

        # 1. Project queries, keys, values for multi-head attention
        queries = self.query_proj(query_features).view(B, N, self.num_heads, self.head_dim).transpose(1, 2) # [B, num_heads, N, head_dim]
        keys = self.key_proj(subject_latents).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)    # [B, num_heads, S, head_dim]
        values = self.value_proj(subject_latents).view(B, S, self.num_heads, self.head_dim).transpose(1, 2) # [B, num_heads, S, head_dim]

        # 2. Compute raw attention scores (query-key dot product)
        # (B, num_heads, N, head_dim) @ (B, num_heads, head_dim, S) -> (B, num_heads, N, S)
        attn_scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 3. Generate Spatial Biases if enabled
        if self.spatial_bias_strength > 0:
            spatial_biases_for_all_subjects = self._generate_gaussian_bias_mask(
                subject_coords, feature_map_spatial_dims, query_features.device
            ) # [B, S, N]

            # Apply spatial bias strength
            spatial_biases_for_all_subjects = spatial_biases_for_all_subjects * self.spatial_bias_strength

            # Expand spatial biases to match attention score dimensions
            # [B, S, N] -> [B, 1, N, S] (for broadcasting across num_heads)
            # Transpose N and S to align with attn_scores last two dimensions
            spatial_biases = spatial_biases_for_all_subjects.permute(0, 2, 1).unsqueeze(1) # [B, 1, N, S]

            # 4. Combine Attention Scores with Spatial Biases
            # spatial_biases will be broadcasted to [B, num_heads, N, S]
            combined_attn_scores = attn_scores + spatial_biases
        else:
            combined_attn_scores = attn_scores


        # 5. Apply Subject Mask to Attention Scores (after spatial bias addition)
        # Mask inactive subjects by setting their attention scores to a very low value (-inf effectively)
        # subject_mask: [B, S] -> [B, 1, 1, S] for broadcasting across N and num_heads
        subject_mask_expanded = subject_mask.unsqueeze(1).unsqueeze(1).bool()
        combined_attn_scores = combined_attn_scores.masked_fill(~subject_mask_expanded, float('-inf'))

        # 6. Apply Softmax to get attention probabilities
        attn_weights = F.softmax(combined_attn_scores, dim=-1)

        # Handle cases where all subjects are masked out to prevent NaNs
        # If a row in attn_weights sums to 0 due to all -inf, softmax might produce NaN.
        # Set to 0 if all subjects are masked for a given query feature.
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # 7. Compute weighted sum of values
        # (B, num_heads, N, S) @ (B, num_heads, S, head_dim) -> (B, num_heads, N, head_dim)
        output = torch.matmul(attn_weights, values)

        # 8. Concatenate heads and apply final projection
        output = output.transpose(1, 2).reshape(B, N, D_query) # [B, N, D_query]
        output = self.out_proj(output)

        return output

```