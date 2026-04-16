```python
import uuid
from typing import List, Optional, Tuple, Dict, Callable

# Assume these are correctly implemented in their respective files as per architecture notes
from src.core.primitives import GeometricPrimitive, Point, BoundingBox
from src.internal_representation.spatial_data_structure import SpatialDataStructure

class SceneGraph:
    """
    A unified, query-optimized representation of the 3D scene.

    It holds all geometric entities (converted by adapters) and their relationships,
    backed by an efficient spatial data structure. This is the central model against
    which all DSL rules are evaluated.
    """

    _entities: Dict[str, GeometricPrimitive]
    _spatial_structure: SpatialDataStructure

    def __init__(self, spatial_structure_factory: Callable[[BoundingBox], SpatialDataStructure], bounds: BoundingBox):
        """
        Initializes the SceneGraph with a chosen spatial data structure and overall scene bounds.

        Args:
            spatial_structure_factory: A callable (e.g., a class constructor like Octree, BVH)
                                       that takes a BoundingBox and returns an instance of SpatialDataStructure.
                                       This allows for flexible choice of spatial indexing.
            bounds: The overall bounding box of the scene, defining the root extent for the
                    spatial data structure. All entities are expected to reside within these bounds
                    or be handled appropriately by the chosen spatial structure.

        Raises:
            ValueError: If the provided bounds are invalid.
            TypeError: If spatial_structure_factory is not a callable.
            RuntimeError: If the spatial data structure fails to initialize.
        """
        if not isinstance(bounds, BoundingBox) or not bounds.is_valid():
            raise ValueError("SceneGraph must be initialized with a valid BoundingBox for its overall bounds.")
        if not callable(spatial_structure_factory):
            raise TypeError("spatial_structure_factory must be a callable (e.g., a class type).")

        self._entities = {}
        try:
            # Instantiate the spatial data structure using the factory and provided bounds
            self._spatial_structure = spatial_structure_factory(bounds)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize spatial data structure with provided factory: {e}") from e

    def add_entity(self, entity: GeometricPrimitive, entity_id: str = None) -> str:
        """
        Adds a geometric primitive to the scene graph.

        Each entity is stored with a unique identifier and indexed in the spatial data structure
        based on its bounding box.

        Args:
            entity: The GeometricPrimitive to add. Must have a `bounding_box()` method.
            entity_id: An optional unique identifier for the entity. If None, a UUID will be generated.

        Returns:
            The unique identifier of the added entity.

        Raises:
            TypeError: If the entity is not a GeometricPrimitive or entity_id is not a string.
            ValueError: If the generated or provided entity_id already exists.
            RuntimeError: If the entity cannot be inserted into the spatial structure (e.g., invalid bounds).
        """
        if not isinstance(entity, GeometricPrimitive):
            raise TypeError(f"Only instances of GeometricPrimitive can be added to the SceneGraph, got {type(entity).__name__}.")

        if entity_id is None:
            entity_id = str(uuid.uuid4())
        
        if not isinstance(entity_id, str):
            raise TypeError("entity_id must be a string or None.")

        if entity_id in self._entities:
            raise ValueError(f"An entity with ID '{entity_id}' already exists in the SceneGraph.")

        try:
            entity_bounds = entity.bounding_box()
            if not isinstance(entity_bounds, BoundingBox) or not entity_bounds.is_valid():
                raise ValueError(f"Entity '{entity_id}' returned an invalid bounding box.")
            
            self._spatial_structure.insert(entity_id, entity_bounds)
            self._entities[entity_id] = entity
            return entity_id
        except Exception as e:
            # Re-raise with more context for debugging insertion failures
            raise RuntimeError(f"Failed to insert entity '{entity_id}' into spatial structure: {e}") from e

    def remove_entity(self, entity_id: str):
        """
        Removes a geometric primitive from the scene graph by its ID.

        Args:
            entity_id: The unique identifier of the entity to remove.

        Raises:
            TypeError: If entity_id is not a string.
            ValueError: If no entity with the given ID is found.
            RuntimeError: If the entity cannot be removed from the spatial structure.
        """
        if not isinstance(entity_id, str):
            raise TypeError("entity_id must be a string.")

        if entity_id not in self._entities:
            raise ValueError(f"No entity with ID '{entity_id}' found in the SceneGraph.")

        entity = self._entities[entity_id]
        try:
            entity_bounds = entity.bounding_box()
            self._spatial_structure.remove(entity_id, entity_bounds)
            del self._entities[entity_id]
        except Exception as e:
            # Re-raise with more context for debugging removal failures
            raise RuntimeError(f"Failed to remove entity '{entity_id}' from spatial structure: {e}") from e

    def get_entity(self, entity_id: str) -> Optional[GeometricPrimitive]:
        """
        Retrieves a geometric primitive by its unique identifier.

        Args:
            entity_id: The unique identifier of the entity to retrieve.

        Returns:
            The GeometricPrimitive object if found, otherwise None.

        Raises:
            TypeError: If entity_id is not a string.
        """
        if not isinstance(entity_id, str):
            raise TypeError("entity_id must be a string.")
        return self._entities.get(entity_id)

    def query_by_bounds(self, query_bounds: BoundingBox) -> List[GeometricPrimitive]:
        """
        Queries the scene graph for entities whose bounding boxes intersect the given query bounds.

        Args:
            query_bounds: The BoundingBox to query against.

        Returns:
            A list of GeometricPrimitive objects whose bounding boxes intersect the query bounds.
            The order of results is not guaranteed.

        Raises:
            TypeError: If query_bounds is not a BoundingBox.
            ValueError: If query_bounds is not valid.
            RuntimeError: If the underlying spatial structure query fails.
        """
        if not isinstance(query_bounds, BoundingBox) or not query_bounds.is_valid():
            raise ValueError("Query bounds must be a valid BoundingBox.")

        try:
            # Delegate to the spatial structure to get a list of intersecting entity IDs
            entity_ids = self._spatial_structure.query_bounds(query_bounds)
            # Map the IDs back to the actual GeometricPrimitive objects
            return [self._entities[eid] for eid in entity_ids if eid in self._entities]
        except Exception as e:
            raise RuntimeError(f"Spatial query by bounds failed: {e}") from e

    def query_nearest(self, point: Point, k: int = 1) -> List[Tuple[GeometricPrimitive, float]]:
        """
        Queries the scene graph for the k nearest entities to a given point.

        Args:
            point: The Point from which to find the nearest entities.
            k: The number of nearest entities to retrieve. Must be a positive integer.

        Returns:
            A list of tuples, where each tuple contains a GeometricPrimitive object and its
            distance to the query point, sorted by distance in ascending order.
            If fewer than k entities are found, all found entities are returned.

        Raises:
            TypeError: If point is not a Point or k is not an integer.
            ValueError: If k is less than 1.
            RuntimeError: If the underlying spatial structure query fails.
        """
        if not isinstance(point, Point):
            raise TypeError("Query point must be an instance of Point.")
        if not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer.")

        try:
            # Delegate to the spatial structure to get a list of nearest entity IDs and their distances
            # Expected format: List[Tuple[str, float]] (entity_id, distance)
            nearest_items_info = self._spatial_structure.query_nearest(point, k)
            
            # Map the IDs back to GeometricPrimitive objects and return with distances
            results = []
            for eid, dist in nearest_items_info:
                if eid in self._entities:
                    results.append((self._entities[eid], dist))
            return results
        except Exception as e:
            raise RuntimeError(f"Spatial query for nearest entities failed: {e}") from e

    def get_all_entities(self) -> List[GeometricPrimitive]:
        """
        Retrieves all geometric primitives currently stored in the scene graph.

        Returns:
            A list containing all GeometricPrimitive objects in the scene graph.
            The order of entities in the list is not guaranteed.
        """
        return list(self._entities.values())
```