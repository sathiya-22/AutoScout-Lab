```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttention(nn.Module):
    """
    Implements a deep learning-based cross-modal attention mechanism to learn and identify
    temporal correspondences between processed audio features and visual features.

    This module performs scaled dot-product attention in both directions (audio querying
    visual, and visual querying audio) and returns both the context-aware features
    and the raw attention maps, which directly highlight temporal correspondences.
    """
    def __init__(self, audio_feature_dim: int, visual_feature_dim: int, embed_dim: int, num_heads: int = 1):
        """
        Initializes the CrossModalAttention module.

        Args:
            audio_feature_dim (int): Dimensionality of the input audio features.
            visual_feature_dim (int): Dimensionality of the input visual features.
            embed_dim (int): The dimension to project features into for attention computation.
                             This should be divisible by num_heads.
            num_heads (int): Number of attention heads.
        """
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads}).")
        if num_heads < 1:
            raise ValueError("num_heads must be a positive integer.")

        self.audio_feature_dim = audio_feature_dim
        self.visual_feature_dim = visual_feature_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections for audio features (Query, Key, Value)
        self.audio_query_proj = nn.Linear(audio_feature_dim, embed_dim, bias=False)
        self.audio_key_proj = nn.Linear(audio_feature_dim, embed_dim, bias=False)
        self.audio_value_proj = nn.Linear(audio_feature_dim, embed_dim, bias=False)

        # Linear projections for visual features (Query, Key, Value)
        self.visual_query_proj = nn.Linear(visual_feature_dim, embed_dim, bias=False)
        self.visual_key_proj = nn.Linear(visual_feature_dim, embed_dim, bias=False)
        self.visual_value_proj = nn.Linear(visual_feature_dim, embed_dim, bias=False)

        # Output projection layers to restore original feature dimensions for context-aware features
        self.output_proj_audio_to_visual = nn.Linear(embed_dim, audio_feature_dim)
        self.output_proj_visual_to_audio = nn.Linear(embed_dim, visual_feature_dim)

        self.scale = self.head_dim ** -0.5

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Splits the last dimension of the tensor (embed_dim) into (num_heads, head_dim)
        and rearranges to (batch_size, num_heads, sequence_length, head_dim).
        """
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Combines attention heads back to a single tensor for output projection.
        (batch_size, num_heads, sequence_length, head_dim) -> (batch_size, sequence_length, embed_dim)
        """
        batch_size, _, seq_len, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

    def forward(self, audio_features: torch.Tensor, visual_features: torch.Tensor):
        """
        Performs bidirectional cross-modal attention between audio and visual features.

        Args:
            audio_features (torch.Tensor): Input audio features.
                                           Expected shape: (batch_size, audio_sequence_length, audio_feature_dim)
            visual_features (torch.Tensor): Input visual features.
                                            Expected shape: (batch_size, visual_sequence_length, visual_feature_dim)

        Returns:
            tuple: A tuple containing:
                - audio_aligned_visual (torch.Tensor): Audio features enriched with visual context.
                                                       Shape: (batch_size, audio_sequence_length, audio_feature_dim)
                - visual_aligned_audio (torch.Tensor): Visual features enriched with audio context.
                                                       Shape: (batch_size, visual_sequence_length, visual_feature_dim)
                - audio_to_visual_attn_map (torch.Tensor): Attention map from audio (query) to visual (key/value).
                                                           Shape: (batch_size, num_heads, audio_sequence_length, visual_sequence_length)
                - visual_to_audio_attn_map (torch.Tensor): Attention map from visual (query) to audio (key/value).
                                                           Shape: (batch_size, num_heads, visual_sequence_length, audio_sequence_length)
        Raises:
            ValueError: If input features are None, have incorrect dimensions, or feature dimensions mismatch
                        with the dimensions specified during initialization.
        """
        if audio_features is None or visual_features is None:
            raise ValueError("Both 'audio_features' and 'visual_features' must be provided.")
        if audio_features.dim() != 3 or visual_features.dim() != 3:
            raise ValueError("Input features must be 3-dimensional (batch_size, sequence_length, feature_dim).")

        # Validate feature dimensions
        if audio_features.size(-1) != self.audio_feature_dim:
            raise ValueError(f"Audio feature dimension mismatch. Expected {self.audio_feature_dim}, "
                             f"got {audio_features.size(-1)}.")
        if visual_features.size(-1) != self.visual_feature_dim:
            raise ValueError(f"Visual feature dimension mismatch. Expected {self.visual_feature_dim}, "
                             f"got {visual_features.size(-1)}.")

        batch_size = audio_features.size(0)

        # Project features for attention computation
        # Audio Queries, Keys, Values (B, S_a, E)
        aq = self.audio_query_proj(audio_features)
        ak = self.audio_key_proj(audio_features)
        av = self.audio_value_proj(audio_features)

        # Visual Queries, Keys, Values (B, S_v, E)
        vq = self.visual_query_proj(visual_features)
        vk = self.visual_key_proj(visual_features)
        vv = self.visual_value_proj(visual_features)

        # Split into multiple heads (B, H, S, D_H)
        aq_h = self._split_heads(aq)
        ak_h = self._split_heads(ak)
        av_h = self._split_heads(av)
        vq_h = self._split_heads(vq)
        vk_h = self._split_heads(vk)
        vv_h = self._split_heads(vv)

        # --- Audio attends to Visual (Audio queries Visual) ---
        # Query: audio (aq_h), Key: visual (vk_h), Value: visual (vv_h)
        # Compute raw attention scores (B, H, S_a, D_H) @ (B, H, D_H, S_v) -> (B, H, S_a, S_v)
        scores_av = torch.matmul(aq_h, vk_h.transpose(-2, -1)) * self.scale
        audio_to_visual_attn_map = F.softmax(scores_av, dim=-1) # (B, H, S_a, S_v) - attention weights

        # Apply attention to visual values (B, H, S_a, S_v) @ (B, H, S_v, D_H) -> (B, H, S_a, D_H)
        audio_attended_visual_h = torch.matmul(audio_to_visual_attn_map, vv_h)
        # Combine heads and project back to original audio feature dimension
        audio_attended_visual = self._combine_heads(audio_attended_visual_h) # (B, S_a, E)
        audio_aligned_visual = self.output_proj_audio_to_visual(audio_attended_visual) # (B, S_a, audio_feature_dim)

        # --- Visual attends to Audio (Visual queries Audio) ---
        # Query: visual (vq_h), Key: audio (ak_h), Value: audio (av_h)
        # Compute raw attention scores (B, H, S_v, D_H) @ (B, H, D_H, S_a) -> (B, H, S_v, S_a)
        scores_va = torch.matmul(vq_h, ak_h.transpose(-2, -1)) * self.scale
        visual_to_audio_attn_map = F.softmax(scores_va, dim=-1) # (B, H, S_v, S_a) - attention weights

        # Apply attention to audio values (B, H, S_v, S_a) @ (B, H, S_a, D_H) -> (B, H, S_v, D_H)
        visual_attended_audio_h = torch.matmul(visual_to_audio_attn_map, av_h)
        # Combine heads and project back to original visual feature dimension
        visual_attended_audio = self._combine_heads(visual_attended_audio_h) # (B, S_v, E)
        visual_aligned_audio = self.output_proj_visual_to_audio(visual_attended_audio) # (B, S_v, visual_feature_dim)

        return (audio_aligned_visual, visual_aligned_audio,
                audio_to_visual_attn_map, visual_to_audio_attn_map)

```