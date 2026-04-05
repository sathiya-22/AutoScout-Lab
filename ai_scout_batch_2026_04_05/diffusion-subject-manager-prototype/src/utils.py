import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Any, Tuple

import torch


@dataclass
class BoundingBox:
    """
    Represents a 2D bounding box for a subject.
    Format: [x, y, width, height] where (x,y) is the top-left corner.
    Coordinates can be normalized [0, 1] or absolute pixel values.
    """
    x: float
    y: float
    width: float
    height: float

    def __post_init__(self):
        if self.width < 0 or self.height < 0:
            raise ValueError("Bounding box width and height cannot be negative.")

    def to_list(self) -> List[float]:
        """Returns the bounding box as a list [x, y, width, height]."""
        return [self.x, self.y, self.width, self.height]

    @classmethod
    def from_list(cls, bbox_list: List[float]):
        """Creates a BoundingBox instance from a list [x, y, width, height]."""
        if len(bbox_list) != 4:
            raise ValueError("Bounding box list must contain 4 elements: [x, y, width, height].")
        return cls(x=bbox_list[0], y=bbox_list[1], width=bbox_list[2], height=bbox_list[3])

    def get_center(self) -> Tuple[float, float]:
        """Returns the center (cx, cy) of the bounding box."""
        return self.x + self.width / 2, self.y + self.height / 2


@dataclass
class Action:
    """
    Represents a discrete action or intention associated with a subject.
    """
    name: str  # e.g., "walking", "waving left arm"
    strength: float = 1.0  # e.g., for continuous actions or emphasis
    metadata: dict = field(default_factory=dict) # e.g., {"target_object_id": "table"}


@dataclass
class Subject:
    """
    Encapsulates all per-subject data managed by the Multi-Subject Latent Manager.
    """
    ID: Any = field(default_factory=lambda: str(uuid.uuid4()))
    # Current and potentially historical spatial information
    bounding_box: Optional[BoundingBox] = None
    # Action queue: Sequence of pending or active actions/intentions
    action_queue: List[Action] = field(default_factory=list)
    # Index referencing the subject's dedicated latent vector in the LatentVectorPool
    latent_token_idx: Optional[int] = None
    # Optional: pre-computed or extracted features for robust identity tracking
    identity_features: Optional[torch.Tensor] = None # e.g., CLIP embeddings

    def __post_init__(self):
        if not self.ID:
            raise ValueError("Subject ID cannot be empty.")
        if self.latent_token_idx is not None and self.latent_token_idx < 0:
            raise ValueError("Latent token index must be non-negative if assigned.")

    def update_spatial_coordinates(self, bbox: BoundingBox):
        """Updates the subject's current bounding box."""
        self.bounding_box = bbox

    def add_action(self, action: Action):
        """Adds an action to the subject's action queue."""
        self.action_queue.append(action)

    def get_current_action(self) -> Optional[Action]:
        """Returns the first action in the queue, if any."""
        return self.action_queue[0] if self.action_queue else None

    def remove_completed_action(self):
        """Removes the first action from the queue, typically after completion."""
        if self.action_queue:
            self.action_queue.pop(0)


def normalize_coords(coords: List[float], image_size: Tuple[int, int]) -> List[float]:
    """
    Normalizes coordinates [x, y, w, h] to [0, 1] range based on image_size.
    Assumes image_size = (width, height).
    """
    if len(coords) != 4:
        raise ValueError("Coordinates list must contain 4 elements: [x, y, w, h].")
    img_w, img_h = image_size
    if img_w <= 0 or img_h <= 0:
        raise ValueError("Image width and height must be positive for normalization.")
    
    return [
        coords[0] / img_w,
        coords[1] / img_h,
        coords[2] / img_w,
        coords[3] / img_h
    ]

def denormalize_coords(normalized_coords: List[float], image_size: Tuple[int, int]) -> List[float]:
    """
    Denormalizes coordinates [x, y, w, h] from [0, 1] range to absolute pixel values.
    Assumes image_size = (width, height).
    """
    if len(normalized_coords) != 4:
        raise ValueError("Normalized coordinates list must contain 4 elements: [x, y, w, h].")
    img_w, img_h = image_size
    if img_w <= 0 or img_h <= 0:
        raise ValueError("Image width and height must be positive for denormalization.")
    
    return [
        normalized_coords[0] * img_w,
        normalized_coords[1] * img_h,
        normalized_coords[2] * img_w,
        normalized_coords[3] * img_h
    ]