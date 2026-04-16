```python
import os
from typing import List

# Assuming these imports are available as per the architecture notes
from src.core.primitives import Point, Triangle, Mesh
from src.scene_adapters.base_adapter import BaseAdapter


class MeshAdapter(BaseAdapter):
    """
    Adapter for loading mesh data from common 3D file formats (e.g., OBJ)
    and converting it into the core geometric `Mesh` primitive, composed of `Triangle`s.
    """

    def adapt(self, source_path: str) -> Mesh:
        """
        Loads mesh data from the specified file path and converts it into
        a `Mesh` object containing `Triangle` primitives.

        Currently supports a simplified parsing of OBJ files (vertices and triangular faces).

        Args:
            source_path: The path to the mesh file (e.g., 'path/to/model.obj').

        Returns:
            A `Mesh` object representing the loaded geometry.

        Raises:
            FileNotFoundError: If the source_path does not exist.
            ValueError: If the file format is unsupported or malformed.
        """
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Mesh file not found at: {source_path}")

        # Determine file type based on extension for potential future expansion
        _, ext = os.path.splitext(source_path)
        ext = ext.lower()

        if ext == '.obj':
            return self._load_obj(source_path)
        # Add more parsers for other formats here as needed (e.g., .ply, .stl)
        # elif ext == '.ply':
        #     return self._load_ply(source_path)
        else:
            raise ValueError(
                f"Unsupported mesh file format: '{ext}' for '{source_path}'. "
                f"Only .obj is supported for this prototype."
            )

    def _load_obj(self, obj_path: str) -> Mesh:
        """
        Internal method to parse an OBJ file.
        Focuses on 'v' (vertex) and 'f' (face) definitions to create `Point` and `Triangle` primitives.
        Ignores normals (vn), texture coordinates (vt), groups (g), etc., for simplicity in this prototype.
        Assumes triangular faces. If faces with more than 3 vertices (quads, ngons) are present,
        it performs a simple fan triangulation from the first vertex of the face.
        """
        vertices: List[Point] = []
        triangles: List[Triangle] = []

        with open(obj_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split()
                if not parts:
                    continue

                prefix = parts[0]

                if prefix == 'v':
                    try:
                        x, y, z = map(float, parts[1:4])
                        vertices.append(Point(x, y, z))
                    except (ValueError, IndexError) as e:
                        print(
                            f"Warning: Malformed vertex line {line_num} in {obj_path}: "
                            f"'{line}'. Skipping. Error: {e}"
                        )
                        continue
                elif prefix == 'f':
                    # Faces can be defined as 'f v1 v2 v3' or 'f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3'
                    # We extract only the vertex index (the first number before '/')
                    face_vertex_indices: List[int] = []
                    for part_idx, part in enumerate(parts[1:]):
                        try:
                            # Split by '/' and take the first part (vertex index)
                            # Convert to int and adjust for 0-based indexing (OBJ uses 1-based)
                            v_idx = int(part.split('/')[0]) - 1
                            face_vertex_indices.append(v_idx)
                        except (ValueError, IndexError) as e:
                            print(
                                f"Warning: Malformed face vertex definition "
                                f"'{part}' on line {line_num} in {obj_path}. "
                                f"Skipping face. Error: {e}"
                            )
                            face_vertex_indices = [] # Invalidate the current face
                            break

                    if len(face_vertex_indices) >= 3:
                        # For faces with more than 3 vertices (quads, ngons),
                        # perform a simple fan triangulation from the first vertex.
                        # This is a common simplification for geometric validation.
                        try:
                            p0 = vertices[face_vertex_indices[0]]
                            for i in range(1, len(face_vertex_indices) - 1):
                                p1 = vertices[face_vertex_indices[i]]
                                p2 = vertices[face_vertex_indices[i + 1]]

                                # Basic bounds check for robustness
                                if all(0 <= idx < len(vertices) for idx in [face_vertex_indices[0], face_vertex_indices[i], face_vertex_indices[i+1]]):
                                    triangles.append(Triangle(p0, p1, p2))
                                else:
                                    print(f"Warning: Face on line {line_num} in {obj_path} references out-of-bounds vertex index. Skipping triangle.")
                        except IndexError:
                             print(f"Warning: Face on line {line_num} in {obj_path} references an invalid vertex index. Skipping face.")
                    elif face_vertex_indices: # If there are vertex indices but less than 3
                        print(
                            f"Warning: Face on line {line_num} in {obj_path} has "
                            f"fewer than 3 vertices ({len(face_vertex_indices)}). Skipping face."
                        )
                # Other OBJ prefixes like 'vn', 'vt', 'g', 's' are ignored.

        if not vertices:
            raise ValueError(f"No vertices found in OBJ file: {obj_path}")
        if not triangles:
            # This is a warning, not an error, as a mesh might have vertices but no faces,
            # though usually not useful for geometry.
            print(f"Warning: No triangular faces found in OBJ file: {obj_path}")

        return Mesh(triangles)
```