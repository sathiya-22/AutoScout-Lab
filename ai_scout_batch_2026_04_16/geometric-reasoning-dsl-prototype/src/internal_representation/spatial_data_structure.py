import math
from typing import List, Optional, Sequence, Union, Tuple
from abc import ABC, abstractmethod

# --- Minimal Geometric Primitives for internal AABB & BVH operations ---
# In a full implementation, these would typically be imported from `src/core/primitives.py`.
# They are included here for demonstration and to make the file self-contained
# for prototyping purposes.

class Point3D:
    """A minimal 3D point for internal AABB calculations."""
    __slots__ = ('x', 'y', 'z')

    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self) -> str:
        return f"Point3D(x={self.x}, y={self.y}, z={self.z})"

    def __add__(self, other: 'Point3D') -> 'Point3D':
        return Point3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: 'Point3D') -> 'Point3D':
        return Point3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> 'Point3D':
        return Point3D(self.x * scalar, self.y * scalar, self.z * scalar)

    def __truediv__(self, scalar: float) -> 'Point3D':
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide Point3D by zero scalar.")
        return Point3D(self.x / scalar, self.y / scalar, self.z / scalar)

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


class AABB:
    """A minimal Axis-Aligned Bounding Box for internal BVH operations."""
    __slots__ = ('min_pt', 'max_pt')

    def __init__(self, min_pt: Point3D, max_pt: Point3D):
        if not (isinstance(min_pt, Point3D) and isinstance(max_pt, Point3D)):
            raise TypeError("min_pt and max_pt must be Point3D instances.")

        # Ensure min_pt components are <= max_pt components
        self.min_pt = Point3D(
            min(min_pt.x, max_pt.x),
            min(min_pt.y, max_pt.y),
            min(min_pt.z, max_pt.z)
        )
        self.max_pt = Point3D(
            max(min_pt.x, max_pt.x),
            max(min_pt.y, max_pt.y),
            max(min_pt.z, max_pt.z)
        )

    def __repr__(self) -> str:
        return f"AABB(min_pt={self.min_pt}, max_pt={self.max_pt})"

    def union(self, other: 'AABB') -> 'AABB':
        """Returns a new AABB that encloses both this AABB and another."""
        min_x = min(self.min_pt.x, other.min_pt.x)
        min_y = min(self.min_pt.y, other.min_pt.y)
        min_z = min(self.min_pt.z, other.min_pt.z)

        max_x = max(self.max_pt.x, other.max_pt.x)
        max_y = max(self.max_pt.y, other.max_pt.y)
        max_z = max(self.max_pt.z, other.max_pt.z)
        return AABB(Point3D(min_x, min_y, min_z), Point3D(max_x, max_y, max_z))

    def intersects(self, other: 'AABB') -> bool:
        """Checks if this AABB intersects with another AABB."""
        return (self.min_pt.x <= other.max_pt.x and self.max_pt.x >= other.min_pt.x and
                self.min_pt.y <= other.max_pt.y and self.max_pt.y >= other.min_pt.y and
                self.min_pt.z <= other.max_pt.z and self.max_pt.z >= other.min_pt.z)

    def get_centroid(self) -> Point3D:
        """Returns the center point of the AABB."""
        return Point3D(
            (self.min_pt.x + self.max_pt.x) / 2,
            (self.min_pt.y + self.max_pt.y) / 2,
            (self.min_pt.z + self.max_pt.z) / 2
        )

    def get_extent(self) -> Point3D:
        """Returns the dimensions (width, height, depth) of the AABB."""
        return Point3D(
            self.max_pt.x - self.min_pt.x,
            self.max_pt.y - self.min_pt.y,
            self.max_pt.z - self.min_pt.z
        )

    def get_longest_axis(self) -> int:
        """Returns the index (0 for X, 1 for Y, 2 for Z) of the longest axis."""
        extent = self.get_extent()
        if extent.x >= extent.y and extent.x >= extent.z:
            return 0  # X-axis
        elif extent.y >= extent.x and extent.y >= extent.z:
            return 1  # Y-axis
        else:
            return 2  # Z-axis

# Mocking `GeometricPrimitive` from `src/core/primitives.py`
class GeometricPrimitive(ABC):
    """
    Abstract base class for all geometric primitives.
    In a real scenario, this would be imported from `src/core/primitives.py`.
    """
    _id_counter = 0

    def __init__(self):
        # Assign a unique ID for identification, useful for query results
        self._id = f"Primitive_{GeometricPrimitive._id_counter}"
        GeometricPrimitive._id_counter += 1

    @property
    @abstractmethod
    def bounding_box(self) -> AABB:
        """
        Returns the Axis-Aligned Bounding Box (AABB) of the primitive.
        Must be implemented by concrete primitive classes.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id='{self._id}')"


# --- Spatial Data Structure Abstract Base Class ---

class SpatialDataStructure(ABC):
    """
    Abstract base class for spatial indexing structures.
    These structures organize geometric primitives to accelerate spatial queries.
    """
    @abstractmethod
    def add_primitives(self, primitives: Sequence[GeometricPrimitive]):
        """
        Adds a sequence of geometric primitives to the structure.
        The structure must be rebuilt (`build()` method called) after adding
        primitives for changes to take effect in subsequent queries.
        """
        pass

    @abstractmethod
    def build(self):
        """
        Constructs or rebuilds the spatial data structure from its current set of primitives.
        This operation can be computationally intensive and should be called after
        all primitives have been added or modified.
        """
        pass

    @abstractmethod
    def query_overlap(self, query_primitive: GeometricPrimitive) -> List[GeometricPrimitive]:
        """
        Performs an accelerated query to find all primitives whose bounding boxes overlap
        with the bounding box of the `query_primitive`.

        Note: This is an AABB-level overlap query, which provides a list of candidates.
              Precise geometric intersection tests using `src/core/operations.py`
              would typically be performed on these candidates by the caller.

        Args:
            query_primitive: The primitive to query against. Its `bounding_box` property
                             is used to define the query region.

        Returns:
            A list of `GeometricPrimitive` objects whose AABBs overlap with the
            `query_primitive`'s AABB. Returns an empty list if the structure is empty
            or no overlaps are found.
        """
        pass

    @abstractmethod
    def clear(self):
        """
        Clears all primitives and resets the spatial structure, making it empty.
        """
        pass


# --- Bounding Volume Hierarchy (BVH) Implementation ---

class BVHNode:
    """
    Represents a node in the Bounding Volume Hierarchy.
    A node can either be an internal node (having left/right children) or
    a leaf node (holding a list of primitives).
    """
    __slots__ = ('bbox', 'primitives', 'left', 'right')

    def __init__(self, bbox: AABB):
        self.bbox: AABB = bbox
        self.primitives: Optional[List[GeometricPrimitive]] = None  # Populated only for leaf nodes
        self.left: Optional['BVHNode'] = None  # Populated only for internal nodes
        self.right: Optional['BVHNode'] = None # Populated only for internal nodes

    @property
    def is_leaf(self) -> bool:
        """True if this node is a leaf (contains primitives, no children)."""
        return self.left is None and self.right is None


class BoundingVolumeHierarchy(SpatialDataStructure):
    """
    An Axis-Aligned Bounding Box (AABB) based Bounding Volume Hierarchy (BVH).
    This spatial data structure is optimized for fast spatial queries (e.g., overlap tests)
    over a static collection of `GeometricPrimitive` objects.
    """
    def __init__(self,
                 primitives: Optional[Sequence[GeometricPrimitive]] = None,
                 max_primitives_per_leaf: int = 8):
        if not (isinstance(max_primitives_per_leaf, int) and max_primitives_per_leaf > 0):
            raise ValueError("max_primitives_per_leaf must be a positive integer.")

        self._primitives: List[GeometricPrimitive] = []
        if primitives is not None:
            self.add_primitives(primitives) # Use add_primitives for validation

        self._root: Optional[BVHNode] = None
        self._max_primitives_per_leaf = max_primitives_per_leaf

        if self._primitives:
            self.build()

    def add_primitives(self, primitives: Sequence[GeometricPrimitive]):
        """
        Adds a sequence of geometric primitives to the internal list.
        The BVH's `build()` method must be called subsequently for these
        new primitives to be included in the spatial structure.
        """
        if not isinstance(primitives, Sequence):
            raise TypeError("Primitives must be a sequence (e.g., list, tuple).")
        for p in primitives:
            if not isinstance(p, GeometricPrimitive):
                raise TypeError(f"All items in primitives must be GeometricPrimitive instances, got {type(p)}.")
            try:
                # Access bounding_box to ensure it's implemented and valid
                _ = p.bounding_box
            except AttributeError:
                raise AttributeError(f"Primitive {p} lacks a 'bounding_box' property, "
                                     "which is required for BVH indexing.")

        self._primitives.extend(primitives)

    def clear(self):
        """
        Clears all primitives and resets the BVH structure.
        """
        self._primitives.clear()
        self._root = None

    def build(self):
        """
        Constructs the Bounding Volume Hierarchy from the current set of primitives.
        This operation sorts and partitions primitives and can be time-consuming.
        It should be called after all desired primitives have been added.
        """
        if not self._primitives:
            self._root = None
            return

        # Ensure all primitives have a valid bounding_box before attempting to build
        for p in self._primitives:
            try:
                _ = p.bounding_box
            except AttributeError:
                raise AttributeError(f"Primitive {p} lacks a 'bounding_box' property required for BVH build.")

        self._root = self._build_recursive(self._primitives)

    def _build_recursive(self, primitives_subset: List[GeometricPrimitive]) -> BVHNode:
        """
        Recursively builds a BVH node from a given subset of primitives.
        """
        if not primitives_subset:
            raise ValueError("Cannot build a BVH node from an empty list of primitives.")

        # 1. Calculate the combined AABB for the current subset of primitives
        current_bbox: Optional[AABB] = None
        for p in primitives_subset:
            if current_bbox is None:
                current_bbox = p.bounding_box
            else:
                current_bbox = current_bbox.union(p.bounding_box)

        if current_bbox is None: # This case should ideally not be reached if primitives_subset is not empty
            raise RuntimeError("Failed to compute bounding box for primitives subset during BVH build.")

        # 2. Base case: If the number of primitives is below the leaf threshold, create a leaf node
        if len(primitives_subset) <= self._max_primitives_per_leaf:
            node = BVHNode(current_bbox)
            node.primitives = primitives_subset
            return node

        # 3. Find the longest axis of the current_bbox for splitting
        longest_axis = current_bbox.get_longest_axis()

        # 4. Sort primitives based on the centroid along the longest axis
        def get_centroid_coord(p: GeometricPrimitive) -> float:
            centroid = p.bounding_box.get_centroid()
            if longest_axis == 0: return centroid.x
            if longest_axis == 1: return centroid.y
            if longest_axis == 2: return centroid.z
            return 0.0 # Should not be reached

        primitives_subset.sort(key=get_centroid_coord)

        # 5. Split the list into two halves
        mid_point = len(primitives_subset) // 2
        left_primitives = primitives_subset[:mid_point]
        right_primitives = primitives_subset[mid_point:]

        # Edge case: If splitting results in an empty list (e.g., all centroids identical,
        # or an odd number of primitives where one side gets fewer, and the other side is empty
        # due to `max_primitives_per_leaf` being 1 and `mid_point` is 0).
        # For simplicity in this prototype, if a split cannot be effectively performed
        # into two non-empty subsets, we convert the current node into a leaf.
        if not left_primitives or not right_primitives:
            node = BVHNode(current_bbox)
            node.primitives = primitives_subset # Store all primitives in this leaf
            return node

        # 6. Recursively build children nodes
        left_child = self._build_recursive(left_primitives)
        right_child = self._build_recursive(right_primitives)

        # 7. Create an internal BVH node
        node = BVHNode(current_bbox)
        node.left = left_child
        node.right = right_child
        return node

    def query_overlap(self, query_primitive: GeometricPrimitive) -> List[GeometricPrimitive]:
        """
        Finds all primitives in the BVH whose bounding boxes overlap with
        the bounding box of the `query_primitive`.

        Args:
            query_primitive: The primitive to query against. Its bounding_box is used.

        Returns:
            A list of `GeometricPrimitive` objects whose AABBs overlap with the
            query primitive's AABB. Returns an empty list if no overlaps are found
            or if the BVH has not been built.
        """
        if self._root is None:
            return []

        if not isinstance(query_primitive, GeometricPrimitive):
            raise TypeError("Query primitive must be a GeometricPrimitive instance.")
        try:
            query_aabb = query_primitive.bounding_box
        except AttributeError:
            raise AttributeError(f"Query primitive {query_primitive} lacks a 'bounding_box' property.")

        results: List[GeometricPrimitive] = []
        self._query_overlap_recursive(self._root, query_aabb, results)
        return results

    def _query_overlap_recursive(self, node: Optional[BVHNode], query_aabb: AABB, results: List[GeometricPrimitive]):
        """
        Recursive helper for query_overlap. Traverses the BVH to find overlapping nodes.
        """
        if node is None or not node.bbox.intersects(query_aabb):
            return

        if node.is_leaf:
            # If it's a leaf node, check bounding box overlap for each primitive it contains.
            # Only primitives whose AABBs actually overlap are added to results.
            if node.primitives is not None:
                for p in node.primitives:
                    if p.bounding_box.intersects(query_aabb):
                        results.append(p)
        else:
            # If it's an internal node, recursively check its children.
            self._query_overlap_recursive(node.left, query_aabb, results)
            self._query_overlap_recursive(node.right, query_aabb, results)