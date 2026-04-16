import abc
from typing import Any, Iterable, List

class BaseSceneAdapter(abc.ABC):
    """
    Abstract base class for scene data adapters.

    This interface defines a consistent contract for converting various external
    3D scene representations (e.g., mesh files, point cloud data, voxel grids)
    into a collection of core geometric primitives used by the DGE's
    internal representation.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the base adapter.

        Concrete adapters might use this to store configuration,
        paths, or other necessary parameters.
        """
        pass

    @abc.abstractmethod
    def load_scene_data(self, source: Any) -> Iterable[Any]:
        """
        Abstract method to load scene data from a given source and convert it
        into a collection of core geometric primitives.

        Concrete implementations must override this method.

        Args:
            source: The input source for the scene data. This could be a file path,
                    a data buffer, a database connection, or any specific
                    format understood by the concrete adapter.

        Returns:
            An iterable (e.g., list or generator) of `core` geometric primitives.
            These primitives (e.g., Point, Triangle, BoundingBox) form the
            unified internal representation.

        Raises:
            NotImplementedError: If the concrete adapter fails to implement this method.
            IOError: If there's an issue reading from the source.
            ValueError: If the source data is malformed or unsupported.
        """
        raise NotImplementedError("Concrete adapters must implement load_scene_data()")

    @abc.abstractmethod
    def get_supported_formats(self) -> List[str]:
        """
        Abstract method to return a list of file extensions or format identifiers
        that this adapter supports.

        Returns:
            A list of strings, where each string is a supported format identifier
            (e.g., ['.obj', '.ply'] for a MeshAdapter).
        """
        raise NotImplementedError("Concrete adapters must implement get_supported_formats()")

    def _handle_loading_error(self, message: str, original_exception: Exception):
        """
        Helper method for consistent error handling in concrete adapters.
        Can be overridden or extended by subclasses.
        """
        # Log the error, re-raise with more context, etc.
        # For a base class, a simple re-raise might be sufficient,
        # but concrete adapters might log or map to specific exceptions.
        raise ValueError(f"Scene data loading error: {message}") from original_exception

# Example of how a concrete adapter might use this (not part of this file's output):
# class MeshAdapter(BaseSceneAdapter):
#     def load_scene_data(self, file_path: str) -> Iterable[Any]:
#         try:
#             # Logic to load mesh from file_path
#             # Convert mesh data into core.primitives.Triangle, etc.
#             print(f"Loading mesh from {file_path}")
#             # Dummy implementation
#             from src.core.primitives import Point, Triangle
#             return [
#                 Triangle(Point(0,0,0), Point(1,0,0), Point(0,1,0)),
#                 Triangle(Point(0,0,0), Point(0,1,0), Point(-1,0,0))
#             ]
#         except FileNotFoundError as e:
#             self._handle_loading_error(f"Mesh file not found: {file_path}", e)
#         except Exception as e:
#             self._handle_loading_error(f"Failed to parse mesh file: {file_path}", e)
#
#     def get_supported_formats(self) -> List[str]:
#         return ['.obj', '.stl', '.fbx']
#
# if __name__ == '__main__':
#     # This block is for demonstration/testing and won't be in the final file
#     try:
#         # Attempting to instantiate the abstract class will raise an error
#         # adapter = BaseSceneAdapter() # This would raise TypeError
#         print("BaseSceneAdapter is an abstract class and cannot be instantiated directly.")
#
#         # You would instantiate a concrete adapter like this:
#         # mesh_adapter = MeshAdapter()
#         # primitives = mesh_adapter.load_scene_data("path/to/my_mesh.obj")
#         # print(f"Loaded {len(list(primitives))} primitives.")
#         # print(f"Supported formats for MeshAdapter: {mesh_adapter.get_supported_formats()}")
#     except Exception as e:
#         print(f"An error occurred: {e}")