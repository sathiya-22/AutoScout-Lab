import numpy as np

# --- Placeholder Primitives (would normally come from src/core/primitives.py) ---
# For the sake of making operations runnable, we define minimal representations here.
# In a full system, these would be rich, immutable objects.

class Point:
    __slots__ = ('_coords',)
    def __init__(self, x: float, y: float, z: float):
        self._coords = np.array([float(x), float(y), float(z)])

    @property
    def coords(self) -> np.ndarray:
        return self._coords

    def __repr__(self) -> str:
        return f"Point({self._coords[0]:.2f}, {self._coords[1]:.2f}, {self._coords[2]:.2f})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Point):
            return NotImplemented
        return np.allclose(self.coords, other.coords)

    def __hash__(self) -> int:
        return hash(self.coords.tobytes())

class Vector:
    __slots__ = ('_coords',)
    def __init__(self, x: float, y: float, z: float):
        self._coords = np.array([float(x), float(y), float(z)])

    @property
    def coords(self) -> np.ndarray:
        return self._coords

    def normalize(self) -> 'Vector':
        norm = np.linalg.norm(self._coords)
        if np.isclose(norm, 0.0):
            raise ValueError("Cannot normalize a zero vector.")
        return Vector(*(self._coords / norm))

    def __repr__(self) -> str:
        return f"Vector({self._coords[0]:.2f}, {self._coords[1]:.2f}, {self._coords[2]:.2f})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector):
            return NotImplemented
        return np.allclose(self.coords, other.coords)

    def __hash__(self) -> int:
        return hash(self.coords.tobytes())

class BoundingBox:
    __slots__ = ('_min_coords', '_max_coords')
    def __init__(self, min_point: Point, max_point: Point):
        if not isinstance(min_point, Point) or not isinstance(max_point, Point):
            raise InvalidInputError("BoundingBox initialization requires Point objects.")
        self._min_coords = np.minimum(min_point.coords, max_point.coords)
        self._max_coords = np.maximum(min_point.coords, max_point.coords)

    @property
    def min_point(self) -> Point:
        return Point(*self._min_coords)

    @property
    def max_point(self) -> Point:
        return Point(*self._max_coords)

    @property
    def min_coords(self) -> np.ndarray:
        return self._min_coords

    @property
    def max_coords(self) -> np.ndarray:
        return self._max_coords

    def __repr__(self) -> str:
        return f"BoundingBox(min={self.min_point}, max={self.max_point})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BoundingBox):
            return NotImplemented
        return np.allclose(self.min_coords, other.min_coords) and \
               np.allclose(self.max_coords, other.max_coords)

    def __hash__(self) -> int:
        return hash((self.min_coords.tobytes(), self.max_coords.tobytes()))

class Sphere:
    __slots__ = ('_center', '_radius')
    def __init__(self, center: Point, radius: float):
        if not isinstance(center, Point):
            raise InvalidInputError("Sphere initialization requires a Point object for center.")
        if radius < 0:
            raise ValueError("Radius cannot be negative.")
        self._center = center
        self._radius = float(radius)

    @property
    def center(self) -> Point:
        return self._center

    @property
    def radius(self) -> float:
        return self._radius

    def __repr__(self) -> str:
        return f"Sphere(center={self.center}, radius={self.radius:.2f})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Sphere):
            return NotImplemented
        return self.center == other.center and np.isclose(self.radius, other.radius)

    def __hash__(self) -> int:
        return hash((self.center, self.radius))

class Plane:
    __slots__ = ('_point', '_normal')
    def __init__(self, point_on_plane: Point, normal_vector: Vector):
        if not isinstance(point_on_plane, Point) or not isinstance(normal_vector, Vector):
            raise InvalidInputError("Plane initialization requires Point and Vector objects.")
        if np.isclose(np.linalg.norm(normal_vector.coords), 0):
            raise ValueError("Plane normal vector cannot be a zero vector.")
        self._point = point_on_plane
        self._normal = normal_vector.normalize()

    @property
    def point(self) -> Point:
        return self._point

    @property
    def normal(self) -> Vector:
        return self._normal

    def __repr__(self) -> str:
        return f"Plane(point={self.point}, normal={self.normal})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Plane):
            return NotImplemented
        # Two planes are equal if their normals are equal (or opposite) and a point on one is on the other.
        # For simplicity in prototype, we'll check direct attributes. A more robust check would involve
        # comparing normal and the distance from origin (dot product of normal and point).
        return self.point == other.point and self.normal == other.normal

    def __hash__(self) -> int:
        return hash((self.point, self.normal))

# --- Error Handling ---
class GeometricOperationError(Exception):
    """Base exception for geometric operations."""
    pass

class InvalidInputError(GeometricOperationError):
    """Raised when input primitives are invalid for the operation."""
    pass

class NoIntersectionError(GeometricOperationError):
    """Raised when an operation expects an intersection but none is found."""
    pass

# --- Geometric Operations ---

def distance_points(p1: Point, p2: Point) -> float:
    """Calculates the Euclidean distance between two 3D points."""
    if not isinstance(p1, Point) or not isinstance(p2, Point):
        raise InvalidInputError("Inputs must be Point objects.")
    return np.linalg.norm(p1.coords - p2.coords)

def intersects_aabb_aabb(bbox1: BoundingBox, bbox2: BoundingBox) -> bool:
    """
    Checks if two Axis-Aligned Bounding Boxes (AABBs) intersect.
    Returns True if they intersect, False otherwise.
    """
    if not isinstance(bbox1, BoundingBox) or not isinstance(bbox2, BoundingBox):
        raise InvalidInputError("Inputs must be BoundingBox objects.")

    x_overlap = (bbox1.min_coords[0] <= bbox2.max_coords[0]) and \
                (bbox1.max_coords[0] >= bbox2.min_coords[0])
    y_overlap = (bbox1.min_coords[1] <= bbox2.max_coords[1]) and \
                (bbox1.max_coords[1] >= bbox2.min_coords[1])
    z_overlap = (bbox1.min_coords[2] <= bbox2.max_coords[2]) and \
                (bbox1.max_coords[2] >= bbox2.min_coords[2])

    return x_overlap and y_overlap and z_overlap

def contains_point_aabb(point: Point, bbox: BoundingBox) -> bool:
    """
    Checks if a 3D point is inside an Axis-Aligned Bounding Box (AABB).
    A point on the boundary is considered inside.
    """
    if not isinstance(point, Point) or not isinstance(bbox, BoundingBox):
        raise InvalidInputError("Inputs must be Point and BoundingBox objects.")

    px, py, pz = point.coords
    minx, miny, minz = bbox.min_coords
    maxx, maxy, maxz = bbox.max_coords

    return (px >= minx and px <= maxx and
            py >= miny and py <= maxy and
            pz >= minz and pz <= maxz)

def contains_point_sphere(point: Point, sphere: Sphere) -> bool:
    """
    Checks if a 3D point is inside a Sphere.
    A point on the boundary is considered inside.
    """
    if not isinstance(point, Point) or not isinstance(sphere, Sphere):
        raise InvalidInputError("Inputs must be Point and Sphere objects.")

    dist_sq = np.sum((point.coords - sphere.center.coords)**2)
    return dist_sq <= (sphere.radius**2)

def intersects_sphere_sphere(sphere1: Sphere, sphere2: Sphere) -> bool:
    """
    Checks if two spheres intersect.
    Returns True if they intersect (including touch), False otherwise.
    """
    if not isinstance(sphere1, Sphere) or not isinstance(sphere2, Sphere):
        raise InvalidInputError("Inputs must be Sphere objects.")

    dist_centers = distance_points(sphere1.center, sphere2.center)
    return dist_centers <= (sphere1.radius + sphere2.radius)

def create_translation_matrix(tx: float, ty: float, tz: float) -> np.ndarray:
    """Creates a 4x4 translation matrix."""
    return np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ], dtype=float)

def create_scale_matrix(sx: float, sy: float, sz: float) -> np.ndarray:
    """Creates a 4x4 scale matrix."""
    return np.array([
        [sx, 0, 0, 0],
        [0, sy, 0, 0],
        [0, 0, sz, 0],
        [0, 0, 0, 1]
    ], dtype=float)

def transform_point(point: Point, transformation_matrix: np.ndarray) -> Point:
    """
    Applies a 4x4 transformation matrix to a Point.
    The point is treated as a homogeneous coordinate [x, y, z, 1].
    """
    if not isinstance(point, Point) or not isinstance(transformation_matrix, np.ndarray) or \
       transformation_matrix.shape != (4, 4):
        raise InvalidInputError("Point must be a Point object and transformation_matrix must be a 4x4 numpy array.")

    homogeneous_point = np.append(point.coords, 1.0)
    transformed_homogeneous = np.dot(transformation_matrix, homogeneous_point)

    if np.isclose(transformed_homogeneous[3], 0):
        raise ValueError("Transformation resulted in a point at infinity (w=0).")
    transformed_coords = transformed_homogeneous[:3] / transformed_homogeneous[3]
    return Point(*transformed_coords)

def transform_bounding_box(bbox: BoundingBox, transformation_matrix: np.ndarray) -> BoundingBox:
    """
    Applies a 4x4 transformation matrix to a BoundingBox.
    This calculates the AABB of the transformed 8 corners of the original BoundingBox.
    """
    if not isinstance(bbox, BoundingBox) or not isinstance(transformation_matrix, np.ndarray) or \
       transformation_matrix.shape != (4, 4):
        raise InvalidInputError("BoundingBox must be a BoundingBox object and transformation_matrix must be a 4x4 numpy array.")

    min_c = bbox.min_coords
    max_c = bbox.max_coords
    corners_coords = np.array([
        [min_c[0], min_c[1], min_c[2]],
        [max_c[0], min_c[1], min_c[2]],
        [min_c[0], max_c[1], min_c[2]],
        [min_c[0], min_c[1], max_c[2]],
        [max_c[0], max_c[1], min_c[2]],
        [max_c[0], min_c[1], max_c[2]],
        [min_c[0], max_c[1], max_c[2]],
        [max_c[0], max_c[1], max_c[2]],
    ])

    transformed_corners = []
    for corner_coords in corners_coords:
        homogeneous_corner = np.append(corner_coords, 1.0)
        transformed_homogeneous = np.dot(transformation_matrix, homogeneous_corner)
        if np.isclose(transformed_homogeneous[3], 0):
            raise ValueError("Transformation resulted in a corner at infinity (w=0).")
        transformed_corners.append(transformed_homogeneous[:3] / transformed_homogeneous[3])

    transformed_corners_np = np.array(transformed_corners)

    new_min_coords = np.min(transformed_corners_np, axis=0)
    new_max_coords = np.max(transformed_corners_np, axis=0)

    return BoundingBox(Point(*new_min_coords), Point(*new_max_coords))

def intersects_ray_aabb(ray_origin: Point, ray_direction: Vector, bbox: BoundingBox) -> tuple[bool, float, float]:
    """
    Checks for intersection between a ray and an AABB using the Slab method.
    Returns (True, t_min, t_max) if intersection occurs, where t_min and t_max
    are the entry and exit parameters along the ray.
    Returns (False, -inf, inf) if no intersection.
    """
    if not isinstance(ray_origin, Point) or not isinstance(ray_direction, Vector) or not isinstance(bbox, BoundingBox):
        raise InvalidInputError("Inputs must be Point, Vector, and BoundingBox objects.")

    orig = ray_origin.coords
    direction = ray_direction.coords
    min_c = bbox.min_coords
    max_c = bbox.max_coords

    t_min = -np.inf
    t_max = np.inf

    for i in range(3):
        if np.isclose(direction[i], 0):
            if orig[i] < min_c[i] or orig[i] > max_c[i]:
                return (False, -np.inf, np.inf)
        else:
            t1 = (min_c[i] - orig[i]) / direction[i]
            t2 = (max_c[i] - orig[i]) / direction[i]

            if t1 > t2:
                t1, t2 = t2, t1

            t_min = max(t_min, t1)
            t_max = min(t_max, t2)

            if t_min > t_max:
                return (False, -np.inf, np.inf)

    if t_max < 0:
        return (False, -np.inf, np.inf)

    return (True, t_min, t_max)

def project_point_on_plane(point: Point, plane: Plane) -> Point:
    """
    Projects a 3D point onto a plane.
    Returns the projected Point.
    """
    if not isinstance(point, Point) or not isinstance(plane, Plane):
        raise InvalidInputError("Inputs must be Point and Plane objects.")

    point_coords = point.coords
    plane_point_coords = plane.point.coords
    plane_normal_coords = plane.normal.coords

    vec_to_point = point_coords - plane_point_coords
    distance = np.dot(vec_to_point, plane_normal_coords)

    projected_coords = point_coords - distance * plane_normal_coords
    return Point(*projected_coords)