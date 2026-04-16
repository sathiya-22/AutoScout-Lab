```python
import numpy as np
from typing import List

from src.scene_adapters.base_adapter import BaseAdapter
from src.core.primitives import BoundingBox, Point # Assuming these are defined in core/primitives.py

class VoxelAdapter(BaseAdapter):
    """
    Adapter for converting voxel grid data into a list of geometric primitives.

    Translates a 3D NumPy array representing a voxel grid (where non-zero values
    indicate an occupied voxel) into a collection of BoundingBox primitives.
    """

    def __init__(self, voxel_size: float = 1.0):
        """
        Initializes the VoxelAdapter.

        Args:
            voxel_size (float): The side length of each individual voxel.
                                 Must be a positive value.
        """
        if not isinstance(voxel_size, (int, float)) or voxel_size <= 0:
            raise ValueError(f"voxel_size must be a positive number, received {voxel_size}.")
        self.voxel_size = float(voxel_size)

    def load_scene(self, voxel_data: np.ndarray) -> List[BoundingBox]:
        """
        Loads voxel data and converts it into a list of BoundingBox primitives.

        Each occupied voxel in the input `voxel_data` (represented by a non-zero
        value) is converted into a BoundingBox primitive. The origin (0,0,0)
        is assumed to be at the corner of the voxel grid index (0,0,0).

        The indices of the NumPy array are typically ordered as (depth, row, col)
        or (z, y, x). This adapter maps `(z_idx, y_idx, x_idx)` from the array
        to spatial coordinates `(x, y, z)` where `x = x_idx * voxel_size`,
        `y = y_idx * voxel_size`, and `z = z_idx * voxel_size`.

        Args:
            voxel_data (np.ndarray): A 3D NumPy array where non-zero elements
                                     indicate occupied voxels.

        Returns:
            List[BoundingBox]: A list of BoundingBox primitives, each
                               representing an occupied voxel.

        Raises:
            TypeError: If voxel_data is not a NumPy array.
            ValueError: If voxel_data is not a 3-dimensional array.
        """
        if not isinstance(voxel_data, np.ndarray):
            raise TypeError("Input voxel_data must be a NumPy array.")
        if voxel_data.ndim != 3:
            raise ValueError(
                f"Input voxel_data must be 3-dimensional, but has {voxel_data.ndim} dimensions."
            )

        occupied_voxels: List[BoundingBox] = []

        # Find indices of all occupied voxels (where value is non-zero).
        # np.argwhere returns an array of shape (N, 3), where each row is [z_idx, y_idx, x_idx].
        occupied_indices = np.argwhere(voxel_data != 0)

        for z_idx, y_idx, x_idx in occupied_indices:
            # Calculate the minimum corner of the voxel in world coordinates.
            # Assuming the (0,0,0) voxel starts at (0,0,0) in world space.
            min_x = x_idx * self.voxel_size
            min_y = y_idx * self.voxel_size
            min_z = z_idx * self.voxel_size

            # Calculate the maximum corner of the voxel in world coordinates.
            max_x = (x_idx + 1) * self.voxel_size
            max_y = (y_idx + 1) * self.voxel_size
            max_z = (z_idx + 1) * self.voxel_size

            min_point = Point(min_x, min_y, min_z)
            max_point = Point(max_x, max_y, max_z)

            # Create a BoundingBox for the voxel.
            # The BoundingBox constructor is assumed to handle min/max ordering internally
            # to ensure min_point components are always less than or equal to max_point components.
            occupied_voxels.append(BoundingBox(min_point, max_point))

        return occupied_voxels
```