# geometric-reasoning-dsl-prototype

## Pluggable Geometric Oracle Library for Deterministic Geometric Environments

### Introduction

The Deterministic Geometric Environment (DGE) in SpatialEvo was designed to formalize specific spatial reasoning task categories under explicit geometric validation rules. While effective for its initial 16 predefined categories, extending the DGE to new or more abstract spatial reasoning tasks presented significant challenges. Each new task required substantial, custom engineering effort to define new geometric validation rules and integrate them into the existing oracle, leading to high architectural coupling and developer friction for expansion.

This prototype, named `geometric-reasoning-dsl-prototype`, directly addresses these limitations by providing a modular and extensible framework for defining and validating geometric reasoning tasks.

### Solution: Geometric Reasoning Domain-Specific Language (DSL)

The `geometric-reasoning-dsl-prototype` implements a 'Pluggable Geometric Oracle Library' or 'Geometric Reasoning Domain-Specific Language (DSL)'. This library offers a modular framework for:

*   **Defining and Composing Geometric Primitives:** Fundamental entities like points, vectors, planes, and bounding volumes.
*   **Robust Geometric Operations:** A comprehensive set of operations such as intersection, distance, and containment queries.

Users can define new spatial reasoning tasks and their validation rules by composing these primitives and operations, potentially through a high-level scripting interface or a declarative rule engine. The library includes adapters to abstract away various 3D scene representations (point clouds, meshes, voxels) into a unified internal representation optimized for geometric queries, minimizing the need for full DGE reimplementation.

This approach significantly reduces architectural coupling for defining new spatial reasoning tasks.

### Core Principles

1.  **Modularity & Decoupling:** Clear separation of concerns across geometric foundations (`core`), data ingestion (`scene_adapters`), optimized internal representation (`internal_representation`), rule definition (`dsl`), and validation execution (`oracle`).
2.  **Abstraction:** Abstracting underlying 3D scene representations and providing a high-level language for rule definition.
3.  **Extensibility:** Allowing easy addition of new geometric primitives, operations, data adapters, and new spatial reasoning tasks without modifying core logic.

### Architecture Overview

The `geometric-reasoning-dsl-prototype` is structured into several key components:

#### `src/core/` (Geometric Foundations)
*   **`primitives.py`**: Defines fundamental geometric entities (e.g., `Point`, `Vector`, `Line`, `Plane`, `BoundingBox`, `Sphere`, `Triangle`, `Mesh` primitives). These are immutable, canonical representations.
*   **`operations.py`**: Implements robust, optimized geometric algorithms (e.g., intersection tests, distance calculations, containment queries, transformations, projections) that operate on the primitives. This is the performance-critical layer.

#### `src/scene_adapters/` (Data Ingestion & Abstraction)
*   **`base_adapter.py`**: Defines an abstract interface for scene data adapters, ensuring a consistent contract for converting external 3D data into a common internal format.
*   **`mesh_adapter.py`**, **`pointcloud_adapter.py`**, **`voxel_adapter.py`**: Concrete implementations that translate various 3D scene representations (e.g., OBJ, PLY, PCD, NumPy arrays for voxels) into a set of `core` geometric primitives (e.g., a mesh becomes a collection of triangles, a point cloud a collection of points).

#### `src/internal_representation/` (Optimized Scene Model)
*   **`spatial_data_structure.py`**: Implements efficient spatial indexing structures (e.g., Octree, BVH - Bounding Volume Hierarchy, KD-tree) over the geometric primitives to accelerate spatial queries.
*   **`scene_graph.py`**: A unified, query-optimized representation that holds all geometric entities (converted by adapters) and their relationships, backed by a spatial data structure. This is the model against which all DSL rules are evaluated.

#### `src/dsl/` (Geometric Reasoning DSL)
*   **`declarations.py`**: Provides the high-level Pythonic interface for defining spatial reasoning rules. This could involve decorators (`@rule`), special classes (`Condition`, `Relationship`), or functions that compose `core` operations in a readable manner (e.g., `is_inside(obj_a, obj_b)`, `intersects(geom1, geom2)`). The goal is to be declarative and composable.
*   **`compiler.py`**: Translates the high-level DSL rule definitions into executable validation logic. This involves generating a sequence of calls to `core/operations.py` that will be executed against the `scene_graph`. For a prototype, this might be a simple function registration or a basic AST traversal.

#### `src/oracle/` (Validation Engine)
*   **`validation_engine.py`**: The central orchestrator. It loads a `scene_graph` (populated by an adapter), loads compiled DSL rules (from `dsl/compiler.py`), and executes them. It manages the state of the validation process and coordinates queries against the `scene_graph` using `core/operations.py`.
*   **`results.py`**: Defines data structures for capturing and reporting validation outcomes (e.g., which rules passed/failed, error messages, locations of violations).

### Workflow

1.  An external 3D scene (e.g., mesh file) is loaded via a `scene_adapter`.
2.  The adapter converts the data into `core` primitives.
3.  These primitives are organized into a `scene_graph` within the `internal_representation`, optimized with spatial data structures.
4.  A user defines new spatial reasoning tasks using the `dsl/declarations.py` (e.g., `examples/basic_containment_rule.py`).
5.  The `dsl/compiler.py` processes these definitions into executable logic.
6.  The `validation_engine.py` (triggered by `main.py` or `examples/run_validation_scenario.py`) takes the `scene_graph` and the compiled rules, executes the validations, and produces `results.py`.

### Getting Started (Conceptual)

Details on installation, dependencies, and running example scenarios will be provided in a later iteration. For now, the architecture lays the foundation for a highly flexible and extensible geometric reasoning system.

### License

(To be determined)