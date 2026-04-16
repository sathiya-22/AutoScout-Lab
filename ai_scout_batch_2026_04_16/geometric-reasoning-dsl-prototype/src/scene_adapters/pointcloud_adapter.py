```python
import numpy as np
import os
from typing import List, Union, Any, TYPE_CHECKING

# To facilitate type checking and prevent potential circular dependencies
# if core primitives or base adapters were to depend on this adapter in the future.
if TYPE_CHECKING:
    from src.scene_adapters.base_adapter import AbstractSceneAdapter
    from src.core.primitives import Point
else:
    # Runtime imports
    from src.scene_adapters.base_adapter import AbstractSceneAdapter
    from src.core.primitives import Point


class PointcloudAdapter(AbstractSceneAdapter):
    """
    Adapter for converting various point cloud data representations into a
    list of core Point primitives.

    This adapter supports loading point cloud data from:
    1.  A NumPy array of shape (N, 3) or (N, M>=3), where N is the number
        of points and the first three columns represent (x, y, z) coordinates.
    2.  A text file (e.g., .xyz, .txt) where each line contains space-separated
        x y z coordinates.

    The adapter ensures that the input data is correctly parsed and transformed
    into the canonical `src.core.primitives.Point` objects for subsequent
    geometric operations within the DGE.
    """

    def load_scene_data(self, source: Union[str, np.ndarray]) -> List['Point']:
        """
        Loads point cloud data from the given source and converts it into
        a list of `src.core.primitives.Point` objects.

        Args:
            source: The input source for the point cloud data.
                    Can be a file path (str) to an .xyz/.txt file,
                    or a NumPy array (np.ndarray) of shape (N, 3) or (N, M>=3).

        Returns:
            A list of `src.core.primitives.Point` objects representing the
            point cloud.

        Raises:
            TypeError: If the source type is not supported (not a string or NumPy array).
            FileNotFoundError: If the provided file path does not exist.
            ValueError: If the data format in the file or array is invalid,
                        or if no valid points can be extracted.
            IOError: For other issues encountered during file reading.
        """
        if isinstance(source, str):
            # Assume it's a file path
            return self._load_from_file(source)
        elif isinstance(source, np.ndarray):
            # Assume it's a NumPy array
            return self._load_from_numpy_array(source)
        else:
            raise TypeError(
                f"Unsupported source type for PointcloudAdapter: {type(source)}. "
                "Expected str (file path) or np.ndarray."
            )

    def _load_from_file(self, file_path: str) -> List['Point']:
        """
        Internal method to load point cloud data from a simple text file.
        Each line is expected to contain space-separated x y z coordinates.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Point cloud file not found: '{file_path}'")
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Path is not a file: '{file_path}'")

        points_list: List['Point'] = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    stripped_line = line.strip()
                    if not stripped_line:  # Skip empty lines
                        continue

                    parts = stripped_line.split()
                    if len(parts) < 3:
                        raise ValueError(
                            f"Invalid line format in '{file_path}' at line {line_num}. "
                            f"Expected at least 3 numeric values for x y z, got: '{stripped_line}'."
                        )
                    try:
                        # Only take the first three components for x, y, z
                        x = float(parts[0])
                        y = float(parts[1])
                        z = float(parts[2])
                        points_list.append(Point(x, y, z))
                    except ValueError as e:
                        raise ValueError(
                            f"Could not parse numeric values in '{file_path}' at line {line_num}: {e}. "
                            f"Line content: '{stripped_line}'."
                        ) from e
        except UnicodeDecodeError as e:
            raise IOError(f"Error decoding file '{file_path}'. Try a different encoding: {e}") from e
        except Exception as e:
            # Catch other potential file reading errors (e.g., permissions)
            raise IOError(f"Error reading point cloud file '{file_path}': {e}") from e

        if not points_list:
            raise ValueError(f"No valid points found in file: '{file_path}'. File might be empty or malformed.")

        return points_list

    def _load_from_numpy_array(self, array: np.ndarray) -> List['Point']:
        """
        Internal method to load point cloud data from a NumPy array.
        The array is expected to have a shape (N, 3) or (N, M>=3).
        """
        if not isinstance(array, np.ndarray):
            raise TypeError(f"Expected a NumPy array, but got {type(array)}.")
        if array.ndim != 2 or array.shape[1] < 3:
            raise ValueError(
                f"Invalid NumPy array shape for point cloud. "
                f"Expected (N, 3) or (N, M>=3), but got {array.shape}. "
                "The array must have at least 3 columns for x, y, z coordinates."
            )

        # Ensure numeric type; convert if necessary, but raise error for unconvertible types
        if not np.issubdtype(array.dtype, np.number):
            raise ValueError(f"Unsupported NumPy array dtype: {array.dtype}. Expected a numeric type.")

        # Convert to float for consistency, if not already
        if not np.issubdtype(array.dtype, np.floating):
            try:
                array = array.astype(np.float32)
            except TypeError as e:
                raise ValueError(f"Could not convert NumPy array dtype {array.dtype} to float: {e}") from e

        # We only care about the first 3 columns (x, y, z)
        # Using a list comprehension for efficient conversion to a list of Point objects.
        # This is generally faster than iterating with for loops for large arrays
        # if the Point constructor is lightweight.
        points_list = [Point(row[0], row[1], row[2]) for row in array[:, :3]]

        if not points_list:
            raise ValueError("No points found in the provided NumPy array. Array might be empty.")

        return points_list
```