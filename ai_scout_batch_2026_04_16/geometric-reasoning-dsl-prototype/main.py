import os

# --- src/core/primitives.py ---
class Point:
    def __init__(self, x, y, z):
        if not all(isinstance(arg, (int, float)) for arg in [x, y, z]):
            raise TypeError("Point coordinates must be numeric.")
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"Point(x={self.x}, y={self.y}, z={self.z})"

    def __eq__(self, other):
        if not isinstance(other, Point):
            return NotImplemented
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __hash__(self):
        return hash((self.x, self.y, self.z))

class Vector:
    def __init__(self, x, y, z):
        if not all(isinstance(arg, (int, float)) for arg in [x, y, z]):
            raise TypeError("Vector components must be numeric.")
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"Vector(x={self.x}, y={self.y}, z={self.z})"

class BoundingBox:
    def __init__(self, min_p: Point, max_p: Point):
        if not isinstance(min_p, Point) or not isinstance(max_p, Point):
            raise TypeError("BoundingBox requires two Point instances.")
        if not (min_p.x <= max_p.x and min_p.y <= max_p.y and min_p.z <= max_p.z):
            raise ValueError("Min point must be less than or equal to max point on all axes.")
        self.min_p = min_p
        self.max_p = max_p

    def __repr__(self):
        return f"BoundingBox(min={self.min_p}, max={self.max_p})"

# --- src/core/operations.py ---
class GeometricOperations:
    @staticmethod
    def intersects(geom1, geom2) -> bool:
        """
        Determines if two geometric entities intersect.
        Currently supports BoundingBox-BoundingBox intersection.
        """
        if isinstance(geom1, BoundingBox) and isinstance(geom2, BoundingBox):
            # Check for overlap on all axes
            return not (geom1.max_p.x < geom2.min_p.x or geom1.min_p.x > geom2.max_p.x or
                        geom1.max_p.y < geom2.min_p.y or geom1.min_p.y > geom2.max_p.y or
                        geom1.max_p.z < geom2.min_p.z or geom1.min_p.z > geom2.max_p.z)
        # Add more intersection logic for other primitive types here
        # E.g., Sphere-Sphere, BoundingBox-Sphere, etc.
        print(f"Warning: Intersection check not implemented for types {type(geom1)} and {type(geom2)}. Returning False.")
        return False

    @staticmethod
    def contains(container_geom, contained_geom) -> bool:
        """
        Determines if container_geom fully contains contained_geom.
        Currently supports BoundingBox-BoundingBox containment.
        """
        if isinstance(container_geom, BoundingBox) and isinstance(contained_geom, BoundingBox):
            return (container_geom.min_p.x <= contained_geom.min_p.x and
                    container_geom.min_p.y <= contained_geom.min_p.y and
                    container_geom.min_p.z <= contained_geom.min_p.z and
                    container_geom.max_p.x >= contained_geom.max_p.x and
                    container_geom.max_p.y >= contained_geom.max_p.y and
                    container_geom.max_p.z >= contained_geom.max_p.z)
        # Add more containment logic here
        print(f"Warning: Containment check not implemented for types {type(container_geom)} and {type(contained_geom)}. Returning False.")
        return False

# --- src/scene_adapters/base_adapter.py ---
class BaseSceneAdapter:
    def load_from_path(self, filepath: str) -> dict:
        """
        Abstract method to load scene data from a file and convert it
        into a dictionary of named core geometric primitives.
        """
        raise NotImplementedError("This method must be implemented by concrete adapters.")

# --- src/scene_adapters/mesh_adapter.py ---
class MeshAdapter(BaseSceneAdapter):
    def load_from_path(self, filepath: str) -> dict:
        """
        Simulates loading a mesh file (e.g., OBJ) and converting its content
        into core geometric primitives. For this prototype, it generates
        predefined BoundingBox primitives based on the filename.
        """
        print(f"MeshAdapter: Simulating loading scene data from '{filepath}'...")
        # In a real scenario, this would parse a mesh file (e.g., OBJ)
        # and convert it into a collection of core primitives (e.g., Triangles, Mesh objects).
        # For simplicity, we'll return a dictionary of named BoundingBoxes.

        if os.path.basename(filepath) == "sample_scene.obj":
            # These are example primitives representing objects in a scene.
            return {
                "Room": BoundingBox(Point(0, 0, 0), Point(10, 10, 10)),
                "Table": BoundingBox(Point(1, 1, 1), Point(5, 5, 5)),
                "Chair": BoundingBox(Point(0.5, 0.5, 0.5), Point(2, 2, 2)),
                "Lamp": BoundingBox(Point(6, 6, 6), Point(7, 7, 7)),
                "SmallBox": BoundingBox(Point(2, 2, 2), Point(3, 3, 3)), # Contained in Table and Room
                "BigBox": BoundingBox(Point(8, 8, 8), Point(12, 12, 12)) # Intersects Room, but not contained
            }
        else:
            raise FileNotFoundError(f"Scene data for '{filepath}' not found or recognized by adapter.")

# --- src/internal_representation/spatial_data_structure.py ---
class SpatialDataStructure:
    """
    A placeholder for an optimized spatial indexing structure (e.g., Octree, BVH).
    For the prototype, its operations are minimal.
    """
    def __init__(self):
        self._entities = {} # Store a reference to entities
        print("SpatialDataStructure: Initialized (placeholder).")

    def build(self, entities: dict):
        """Builds or updates the spatial index based on the given entities."""
        self._entities = entities
        # In a real implementation, this would build an Octree, BVH, etc.
        # This operation is performance-critical for large scenes.
        print("SpatialDataStructure: Building optimized spatial index (simulated)...")

    def query(self, geom_query):
        """
        Performs a spatial query (e.g., all objects intersecting geom_query).
        Placeholder for advanced spatial queries.
        """
        # For a full implementation, this would leverage the spatial tree for efficiency.
        # For this prototype, it's a simple linear scan if used.
        print(f"SpatialDataStructure: Executing query for {geom_query} (simulated linear scan)...")
        results = []
        # Example of how it might be used with GeometricOperations
        # for name, entity in self._entities.items():
        #     if GeometricOperations.intersects(geom_query, entity):
        #         results.append((name, entity))
        return results

# --- src/internal_representation/scene_graph.py ---
class SceneGraph:
    """
    A unified, query-optimized representation of the 3D scene,
    holding all geometric entities and backed by a spatial data structure.
    """
    def __init__(self):
        self._entities = {} # Stores named primitives (e.g., "Room": BoundingBox(...))
        self._spatial_index = SpatialDataStructure()
        print("SceneGraph: Initialized.")

    def add_entity(self, name: str, primitive_obj):
        """
        Adds a named geometric primitive to the scene graph.
        """
        if not isinstance(name, str) or not name:
            raise ValueError("Entity name must be a non-empty string.")
        if not hasattr(primitive_obj, '__repr__'): # Basic check for valid primitive
            print(f"Warning: Primitive for '{name}' might not be a valid geometric object type.")

        self._entities[name] = primitive_obj
        self._spatial_index.build(self._entities) # Rebuild or incrementally update index
        print(f"SceneGraph: Added entity '{name}': {type(primitive_obj).__name__}")

    def query_entity(self, name: str):
        """
        Retrieves a geometric primitive by its name.
        """
        entity = self._entities.get(name)
        if entity is None:
            print(f"Warning: Entity '{name}' not found in SceneGraph.")
        return entity

    def get_all_entities(self) -> dict:
        """Returns a copy of all named entities in the graph."""
        return self._entities.copy()

    def get_spatial_index(self) -> SpatialDataStructure:
        """Returns the underlying spatial data structure for advanced queries."""
        return self._spatial_index

# --- src/dsl/declarations.py ---
class RuleDefinition:
    """Internal representation of a DSL rule."""
    def __init__(self, name: str, description: str, logic_func):
        if not name or not description:
            raise ValueError("Rule name and description cannot be empty.")
        if not callable(logic_func):
            raise TypeError("Rule logic must be a callable function.")
        self.name = name
        self.description = description
        self.logic = logic_func # The function that defines the rule logic

def define_rule(name: str, description: str):
    """
    Decorator to define a spatial reasoning rule using the DSL.
    It wraps a function containing the rule's logic into a RuleDefinition object.
    """
    def decorator(func):
        return RuleDefinition(name, description, func)
    return decorator

# --- src/dsl/compiler.py ---
class RuleCompiler:
    @staticmethod
    def compile_rule(rule_def: RuleDefinition):
        """
        Translates a high-level DSL rule definition into executable validation logic.
        For this prototype, it simply returns a callable that can execute the rule's
        logic against a SceneGraph, injecting the GeometricOperations library.
        """
        print(f"DSL Compiler: Compiling rule '{rule_def.name}'...")
        if not isinstance(rule_def, RuleDefinition):
            raise TypeError("Input must be a RuleDefinition instance.")

        # The compiled logic will receive the scene_graph and the operations library
        def compiled_logic(scene_graph: SceneGraph):
            # Execute the user-defined rule logic, passing the scene graph
            # and the GeometricOperations utility.
            return rule_def.logic(scene_graph, GeometricOperations)
        
        return compiled_logic

# --- src/oracle/results.py ---
class ValidationResult:
    """
    Data structure for capturing and reporting validation outcomes.
    """
    def __init__(self, rule_id: str, passed: bool, message: str = "", details: dict = None):
        if not isinstance(rule_id, str) or not rule_id:
            raise ValueError("Rule ID must be a non-empty string.")
        if not isinstance(passed, bool):
            raise TypeError("Passed status must be a boolean.")
        self.rule_id = rule_id
        self.passed = passed
        self.message = message if message is not None else ""
        self.details = details if details is not None else {} # Optional additional data

    def __repr__(self):
        status = "PASSED" if self.passed else "FAILED"
        msg_prefix = f"Rule '{self.rule_id}':"
        full_message = f"{msg_prefix} {self.message}" if self.message else msg_prefix
        return f"[{status}] {full_message}"

# --- src/oracle/validation_engine.py ---
class ValidationEngine:
    """
    The central orchestrator for running spatial reasoning validations.
    It takes a SceneGraph and compiled DSL rules, executes them, and reports results.
    """
    def __init__(self, scene_graph: SceneGraph):
        if not isinstance(scene_graph, SceneGraph):
            raise TypeError("ValidationEngine requires a SceneGraph instance.")
        self._scene_graph = scene_graph
        print("ValidationEngine: Initialized.")

    def run_validation(self, rule_id: str, compiled_rule_logic) -> ValidationResult:
        """
        Executes a single compiled rule against the loaded scene graph.
        """
        print(f"ValidationEngine: Running validation for rule '{rule_id}'...")
        if not callable(compiled_rule_logic):
            raise TypeError("Compiled rule logic must be a callable function.")

        try:
            # Execute the compiled rule logic. It's expected to return:
            # - a boolean (True for passed, False for failed)
            # - or a tuple (bool, str) for (passed_status, message)
            # - or a tuple (bool, str, dict) for (passed_status, message, details)
            result = compiled_rule_logic(self._scene_graph)

            passed = False
            message = "Rule execution completed, but result format was unexpected."
            details = {}

            if isinstance(result, tuple):
                if len(result) >= 1:
                    passed = bool(result[0])
                if len(result) >= 2:
                    message = str(result[1])
                if len(result) >= 3 and isinstance(result[2], dict):
                    details = result[2]
            elif isinstance(result, bool):
                passed = result
                message = f"Rule condition {'met' if passed else 'not met'}."
            else:
                message = f"Rule '{rule_id}' returned an unexpected type: {type(result)}."
                return ValidationResult(rule_id, False, message)

            return ValidationResult(rule_id, passed, message, details)
        except Exception as e:
            print(f"Error during rule '{rule_id}' execution: {e}")
            return ValidationResult(rule_id, False, f"Error during rule execution: {e}")

# --- main.py ---
def main():
    print("--------------------------------------------------")
    print("       Deterministic Geometric Environment (DGE)")
    print("          Pluggable Geometric Oracle Prototype")
    print("--------------------------------------------------\n")

    # --- 1. Load an external 3D scene via a scene adapter ---
    print("Step 1: Loading scene data...")
    adapter = MeshAdapter()
    scene_filepath = "sample_scene.obj" # Simulate a file path
    
    # Create a dummy file so the adapter has something to 'find'
    try:
        with open(scene_filepath, "w") as f:
            f.write("# This is a dummy .obj file for the prototype.\n")
            f.write("v 1.0 1.0 1.0\n") # Minimal content
        
        scene_data_primitives = adapter.load_from_path(scene_filepath)
        print(f"Successfully loaded {len(scene_data_primitives)} entities from '{scene_filepath}'.")
    except FileNotFoundError as e:
        print(f"Critical Error: {e}. Please ensure '{scene_filepath}' can be accessed or mocked correctly.")
        return
    except Exception as e:
        print(f"An unexpected error occurred during scene loading: {e}")
        return
    finally:
        # Clean up the dummy file
        if os.path.exists(scene_filepath):
            os.remove(scene_filepath)
            
    # --- 2. Populate the SceneGraph with core primitives ---
    print("\nStep 2: Building SceneGraph from loaded primitives...")
    scene_graph = SceneGraph()
    for name, primitive in scene_data_primitives.items():
        try:
            scene_graph.add_entity(name, primitive)
        except (ValueError, TypeError) as e:
            print(f"Error adding entity '{name}' to SceneGraph: {e}")
            # Decide whether to continue or exit based on error severity
    print(f"SceneGraph built with {len(scene_graph.get_all_entities())} entities.")


    # --- 3. Define new spatial reasoning tasks using the DSL ---
    print("\nStep 3: Defining spatial reasoning rules using the DSL...")

    @define_rule(
        name="SmallBox_Inside_Room",
        description="Checks if 'SmallBox' is entirely contained within 'Room'."
    )
    def small_box_in_room_rule(sg: SceneGraph, ops_lib):
        room = sg.query_entity("Room")
        small_box = sg.query_entity("SmallBox")

        if not room:
            return False, "Room entity not found in scene."
        if not small_box:
            return False, "SmallBox entity not found in scene."
        
        # Ensure correct primitive types for the operation
        if not isinstance(room, BoundingBox) or not isinstance(small_box, BoundingBox):
            return False, "Room or SmallBox are not BoundingBox primitives, cannot perform containment check."

        is_contained = ops_lib.contains(room, small_box)
        message = f"SmallBox is {'contained' if is_contained else 'NOT contained'} within Room."
        return is_contained, message

    @define_rule(
        name="BigBox_Intersects_Room",
        description="Checks if 'BigBox' intersects with 'Room'."
    )
    def big_box_intersects_room_rule(sg: SceneGraph, ops_lib):
        room = sg.query_entity("Room")
        big_box = sg.query_entity("BigBox")

        if not room:
            return False, "Room entity not found in scene."
        if not big_box:
            return False, "BigBox entity not found in scene."

        if not isinstance(room, BoundingBox) or not isinstance(big_box, BoundingBox):
            return False, "Room or BigBox are not BoundingBox primitives, cannot perform intersection check."

        is_intersecting = ops_lib.intersects(room, big_box)
        message = f"BigBox {'intersects' if is_intersecting else 'does NOT intersect'} with Room."
        return is_intersecting, message

    @define_rule(
        name="Chair_Intersects_Table",
        description="Checks if 'Chair' intersects with 'Table'."
    )
    def chair_intersects_table_rule(sg: SceneGraph, ops_lib):
        chair = sg.query_entity("Chair")
        table = sg.query_entity("Table")

        if not chair:
            return False, "Chair entity not found in scene."
        if not table:
            return False, "Table entity not found in scene."

        if not isinstance(chair, BoundingBox) or not isinstance(table, BoundingBox):
            return False, "Chair or Table are not BoundingBox primitives, cannot perform intersection check."

        is_intersecting = ops_lib.intersects(chair, table)
        message = f"Chair {'intersects' if is_intersecting else 'does NOT intersect'} with Table."
        return is_intersecting, message

    # List of defined rules
    defined_rules = [
        small_box_in_room_rule,
        big_box_intersects_room_rule,
        chair_intersects_table_rule
    ]
    print(f"Defined {len(defined_rules)} rules.")

    # --- 4. Compile the DSL rules ---
    print("\nStep 4: Compiling DSL rules...")
    compiled_rules = {}
    for rule_def in defined_rules:
        try:
            compiled_rules[rule_def.name] = RuleCompiler.compile_rule(rule_def)
        except Exception as e:
            print(f"Error compiling rule '{rule_def.name}': {e}")

    print(f"Successfully compiled {len(compiled_rules)} rules.")

    # --- 5. Execute validations using the ValidationEngine ---
    print("\nStep 5: Executing validations...")
    validation_engine = ValidationEngine(scene_graph)

    all_results = []
    for rule_name, compiled_logic in compiled_rules.items():
        try:
            result = validation_engine.run_validation(rule_name, compiled_logic)
            all_results.append(result)
        except Exception as e:
            all_results.append(ValidationResult(rule_name, False, f"Unexpected error during validation: {e}"))

    # --- 6. Report results ---
    print("\n--------------------------------------------------")
    print("              Validation Results")
    print("--------------------------------------------------")
    for res in all_results:
        print(res)
    print("--------------------------------------------------")

    # Summary
    passed_count = sum(1 for r in all_results if r.passed)
    failed_count = len(all_results) - passed_count
    print(f"\nSummary: {passed_count} rules PASSED, {failed_count} rules FAILED out of {len(all_results)}.")

if __name__ == "__main__":
    main()