```python
import math
from dataclasses import dataclass, field
from typing import Tuple, Sequence

# Define a small epsilon for floating-point comparisons to handle numerical stability
_EPSILON = 1e-9

@dataclass(frozen=True, slots=True)
class Point3D:
    """Represents a point in 3D space."""
    x: float
    y: float
    z: float

    def __post_init__(self):
        if not all(isinstance(coord, (int, float)) for coord in (self.x, self.y, self.z)):
            raise TypeError("Point3D coordinates must be numeric.")

    def __str__(self) -> str:
        return f"P({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

@dataclass(frozen=True, slots=True)
class Vector3D:
    """Represents a vector (direction and magnitude) in 3D space."""
    x: float
    y: float
    z: float

    def __post_init__(self):
        if not all(isinstance(comp, (int, float)) for comp in (self.x, self.y, self.z)):
            raise TypeError("Vector3D components must be numeric.")

    def __str__(self) -> str:
        return f"V({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"

    def length(self) -> float:
        """Calculates the magnitude (length) of the vector."""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def length_sq(self) -> float:
        """Calculates the squared magnitude of the vector (avoids sqrt for comparisons)."""
        return self.x**2 + self.y**2 + self.z**2

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

@dataclass(frozen=True, slots=True)
class Line3D:
    """
    Represents an infinite line in 3D space.
    Defined by an origin point and a direction vector.
    """
    origin: Point3D
    direction: Vector3D

    def __post_init__(self):
        if not isinstance(self.origin, Point3D):
            raise TypeError("Line3D 'origin' must be a Point3D instance.")
        if not isinstance(self.direction, Vector3D):
            raise TypeError("Line3D 'direction' must be a Vector3D instance.")
        if self.direction.length_sq() < _EPSILON:
            raise ValueError("Line3D 'direction' vector cannot be a zero vector.")

    def __str__(self) -> str:
        return f"Line(Origin={self.origin}, Dir={self.direction})"

@dataclass(frozen=True, slots=True)
class Ray3D:
    """
    Represents a ray in 3D space, starting at an origin point and
    extending infinitely in a given direction.
    """
    origin: Point3D
    direction: Vector3D

    def __post_init__(self):
        if not isinstance(self.origin, Point3D):
            raise TypeError("Ray3D 'origin' must be a Point3D instance.")
        if not isinstance(self.direction, Vector3D):
            raise TypeError("Ray3D 'direction' must be a Vector3D instance.")
        if self.direction.length_sq() < _EPSILON:
            raise ValueError("Ray3D 'direction' vector cannot be a zero vector.")

    def __str__(self) -> str:
        return f"Ray(Origin={self.origin}, Dir={self.direction})"

@dataclass(frozen=True, slots=True)
class Segment3D:
    """Represents a finite line segment in 3D space, defined by two endpoints."""
    start_point: Point3D
    end_point: Point3D

    def __post_init__(self):
        if not isinstance(self.start_point, Point3D):
            raise TypeError("Segment3D 'start_point' must be a Point3D instance.")
        if not isinstance(self.end_point, Point3D):
            raise TypeError("Segment3D 'end_point' must be a Point3D instance.")

    def __str__(self) -> str:
        return f"Segment(Start={self.start_point}, End={self.end_point})"

    def length(self) -> float:
        """Calculates the length of the segment."""
        dx = self.end_point.x - self.start_point.x
        dy = self.end_point.y - self.start_point.y
        dz = self.end_point.z - self.start_point.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)

@dataclass(frozen=True, slots=True)
class Plane3D:
    """
    Represents an infinite plane in 3D space.
    Defined by a point on the plane and a normal vector.
    The normal vector should ideally be normalized for consistent behavior.
    """
    point_on_plane: Point3D
    normal: Vector3D

    def __post_init__(self):
        if not isinstance(self.point_on_plane, Point3D):
            raise TypeError("Plane3D 'point_on_plane' must be a Point3D instance.")
        if not isinstance(self.normal, Vector3D):
            raise TypeError("Plane3D 'normal' must be a Vector3D instance.")
        if self.normal.length_sq() < _EPSILON:
            raise ValueError("Plane3D 'normal' vector cannot be a zero vector.")

    def __str__(self) -> str:
        return f"Plane(Point={self.point_on_plane}, Normal={self.normal})"

@dataclass(frozen=True, slots=True)
class AABB:
    """
    Represents an Axis-Aligned Bounding Box (AABB) in 3D space.
    Defined by its minimum and maximum corner points.
    """
    min_point: Point3D
    max_point: Point3D

    def __post_init__(self):
        if not isinstance(self.min_point, Point3D):
            raise TypeError("AABB 'min_point' must be a Point3D instance.")
        if not isinstance(self.max_point, Point3D):
            raise TypeError("AABB 'max_point' must be a Point3D instance.")

        # Ensure min_point is truly the minimum and max_point is the maximum.
        # This makes the AABB canonical, even if constructor args are swapped.
        # This is a bit of a tricky case for frozen=True, as it prevents direct modification.
        # So we have to re-create the object if points are out of order, or
        # rely on the caller to provide valid points.
        # For simplicity and sticking to the immutable principle, let's just validate
        # and raise an error if they are not ordered correctly.
        if not (self.min_point.x <= self.max_point.x and
                self.min_point.y <= self.max_point.y and
                self.min_point.z <= self.max_point.z):
            # As a frozen dataclass, we cannot modify `self.min_point` or `self.max_point`.
            # A more robust solution for canonical representation would be a factory method
            # or creating a new instance. For now, we enforce valid input.
            raise ValueError("AABB min_point must be less than or equal to max_point in all dimensions.")

    @property
    def center(self) -> Point3D:
        return Point3D(
            (self.min_point.x + self.max_point.x) / 2,
            (self.min_point.y + self.max_point.y) / 2,
            (self.min_point.z + self.max_point.z) / 2
        )

    @property
    def size(self) -> Vector3D:
        return Vector3D(
            self.max_point.x - self.min_point.x,
            self.max_point.y - self.min_point.y,
            self.max_point.z - self.min_point.z
        )

    def __str__(self) -> str:
        return f"AABB(Min={self.min_point}, Max={self.max_point})"

@dataclass(frozen=True, slots=True)
class Sphere:
    """Represents a sphere in 3D space, defined by a center point and a radius."""
    center: Point3D
    radius: float

    def __post_init__(self):
        if not isinstance(self.center, Point3D):
            raise TypeError("Sphere 'center' must be a Point3D instance.")
        if not isinstance(self.radius, (int, float)):
            raise TypeError("Sphere 'radius' must be numeric.")
        if self.radius < 0:
            raise ValueError("Sphere 'radius' cannot be negative.")
        # Optionally, handle zero radius as a point, but typically a sphere has positive radius.
        # if self.radius < _EPSILON:
        #     self.radius = 0.0 # Canonicalize tiny radii to zero if desired.

    def __str__(self) -> str:
        return f"Sphere(Center={self.center}, Radius={self.radius:.3f})"

@dataclass(frozen=True, slots=True)
class Triangle3D:
    """Represents a triangle in 3D space, defined by three vertex points."""
    vertex_a: Point3D
    vertex_b: Point3D
    vertex_c: Point3D

    def __post_init__(self):
        if not all(isinstance(v, Point3D) for v in (self.vertex_a, self.vertex_b, self.vertex_c)):
            raise TypeError("Triangle3D vertices must be Point3D instances.")
        
        # Degeneracy check (collinear points):
        # This check can be moved to operations.py for efficiency if needed,
        # or assumed valid by adapters. For a primitive, storing the points is enough.
        # However, for robustness, a basic check helps.
        # This can be done by checking if the area is close to zero.
        # Area vector is 0.5 * cross_product(B-A, C-A)
        
        # Calculate vectors from vertex_a
        vec_ab = Vector3D(self.vertex_b.x - self.vertex_a.x,
                          self.vertex_b.y - self.vertex_a.y,
                          self.vertex_b.z - self.vertex_a.z)
        vec_ac = Vector3D(self.vertex_c.x - self.vertex_a.x,
                          self.vertex_c.y - self.vertex_a.y,
                          self.vertex_c.z - self.vertex_a.z)
        
        # Cross product components
        cx = vec_ab.y * vec_ac.z - vec_ab.z * vec_ac.y
        cy = vec_ab.z * vec_ac.x - vec_ab.x * vec_ac.z
        cz = vec_ab.x * vec_ac.y - vec_ab.y * vec_ac.x
        
        # Squared length of the cross product vector
        cross_prod_sq_length = cx*cx + cy*cy + cz*cz
        
        if cross_prod_sq_length < _EPSILON * _EPSILON: # Compare squared length to avoid sqrt, and use squared epsilon
            raise ValueError("Triangle3D vertices are collinear or coincident, forming a degenerate triangle.")


    def __str__(self) -> str:
        return f"Triangle(A={self.vertex_a}, B={self.vertex_b}, C={self.vertex_c})"

@dataclass(frozen=True, slots=True)
class OrientedBoundingBox:
    """
    Represents an Oriented Bounding Box (OBB) in 3D space.
    Defined by its center, half-extents (size along local axes), and orientation (a 3x3 rotation matrix
    or three orthonormal basis vectors). Using basis vectors here for simplicity.
    """
    center: Point3D
    half_extents: Vector3D  # Half size along each local axis (x,y,z)
    # The axes are expected to be normalized and mutually orthogonal.
    # For simplicity, we'll store them as a tuple of vectors.
    # Operations.py would typically handle matrix conversions or provide utilities.
    axis_x: Vector3D
    axis_y: Vector3D
    axis_z: Vector3D

    def __post_init__(self):
        if not isinstance(self.center, Point3D):
            raise TypeError("OBB 'center' must be a Point3D instance.")
        if not isinstance(self.half_extents, Vector3D):
            raise TypeError("OBB 'half_extents' must be a Vector3D instance.")
        if not all(isinstance(ax, Vector3D) for ax in (self.axis_x, self.axis_y, self.axis_z)):
            raise TypeError("OBB axes must be Vector3D instances.")
        
        # Basic validation: half_extents should be non-negative.
        if not (self.half_extents.x >= 0 and self.half_extents.y >= 0 and self.half_extents.z >= 0):
            raise ValueError("OBB 'half_extents' components cannot be negative.")

        # Robust validation of axes (orthonormality) is crucial for an OBB.
        # This is a bit heavy for __post_init__ but necessary for a robust primitive.
        # Operations.py might handle normalization, but we validate orthogonality and non-zero length.
        if self.axis_x.length_sq() < _EPSILON or \
           self.axis_y.length_sq() < _EPSILON or \
           self.axis_z.length_sq() < _EPSILON:
            raise ValueError("OBB axis vectors cannot be zero vectors.")
        
        # Cross-product-based orthogonality check
        # This is simplified and assumes axes should form a right-handed system if normalized.
        # For full validation, normalize all axes and then check dot products for orthogonality (close to 0)
        # and cross products for correct orientation (e.g., axis_x cross axis_y ~ axis_z).
        # We'll just check if they are very nearly orthogonal for now, assuming they should be.
        
        # Helper for dot product (usually in operations.py, but needed for internal validation)
        def _dot(v1: Vector3D, v2: Vector3D) -> float:
            return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

        # Check for near-orthogonality after normalization (conceptually)
        # More precise check: angles between axes should be 90 degrees.
        # This is a simplified check. A full OBB construction should ensure this.
        if abs(_dot(self.axis_x, self.axis_y)) > _EPSILON or \
           abs(_dot(self.axis_x, self.axis_z)) > _EPSILON or \
           abs(_dot(self.axis_y, self.axis_z)) > _EPSILON:
            # Note: This checks orthogonality *assuming* they are unit vectors, or at least that their dot product is small.
            # A full check would also ensure they are unit vectors.
            raise ValueError("OBB axes must be orthogonal. Consider normalizing them before creating the OBB.")

    def __str__(self) -> str:
        return f"OBB(Center={self.center}, Extents={self.half_extents}, Axes=(X={self.axis_x}, Y={self.axis_y}, Z={self.axis_z}))"

# Alias for common use cases
MeshFace = Triangle3D
```