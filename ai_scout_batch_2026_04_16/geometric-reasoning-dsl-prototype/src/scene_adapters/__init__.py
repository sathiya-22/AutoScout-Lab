class SceneAdapterError(Exception):
    """Base exception for errors occurring within scene adapters."""
    pass

from .base_adapter import BaseSceneAdapter
from .mesh_adapter import MeshAdapter
from .pointcloud_adapter import PointCloudAdapter
from .voxel_adapter import VoxelAdapter

__all__ = [
    "SceneAdapterError",
    "BaseSceneAdapter",
    "MeshAdapter",
    "PointCloudAdapter",
    "VoxelAdapter",
]