# Multi-Subject Latent Management for Consistent Video Diffusion

## Project Overview

This project implements a scalable and efficient architecture designed to enhance video diffusion models by dynamically managing 'subject state tokens' and integrating a 'spatial biasing mechanism'. The primary goal is to ensure consistent multi-subject action binding and identity preservation, especially as the number of subjects and scene complexity increases, which are common challenges in generating complex multi-character video content.

## Solution Sketch

We are developing a **'Multi-Subject Latent Manager' (MSLM)** library. This library will act as a plug-and-play component, mediating between a diffusion model's input conditioning and its internal U-Net layers. It features:
1.  A dynamic registry for tracking subjects, including their unique IDs, spatial coordinates, and action queues.
2.  A pool of dedicated latent vectors, referred to as 'subject state tokens', representing the aggregate state and identity of each subject.
3.  An attention-based **'Spatial Biasing Adapter' (SBA)** designed to guide the diffusion model's U-Net layers by infusing spatial and subject-specific guidance.

The MSLM library aims to minimize computational overhead through efficient sparse or coordinate-aware attention mechanisms.

## Architecture

The proposed architecture centers around a modular `MultiSubjectLatentManager` library, designed for seamless integration with existing video diffusion models.

### 1. Multi-Subject Latent Manager (MSLM) (`src/subject_manager.py`)

*   **Core Functionality:** Orchestrates the lifecycle and state of multiple subjects within a video sequence. It acts as the central hub for managing subject-specific information and their corresponding latent representations.

*   **Components:**

    *   #### Dynamic Subject Registry (`src/subject_registry.py`)
        *   **Purpose:** Maintains a real-time record of all active subjects in the scene.
        *   **Data Structure:** A dictionary mapping unique `subject_ID` (e.g., UUIDs or sequential integers) to `Subject` objects.
        *   **`Subject` Object:** Encapsulates per-subject data:
            *   `ID`: Unique identifier.
            *   `bounding_box`/`spatial_coordinates`: Current and potentially historical 2D/3D positions (e.g., `[x, y, w, h]` or keypoints). Updated per frame/timestep.
            *   `action_queue`: A sequence of pending or active actions/intentions associated with the subject (e.g., "walking," "waving left arm").
            *   `latent_token_idx`: Index referencing the subject's dedicated latent vector in the `LatentVectorPool`.
            *   `identity_features`: Optional, pre-computed or extracted features (e.g., CLIP embeddings of face/body) for robust identity tracking.
        *   **Operations:** `add_subject()`, `remove_subject()`, `update_subject_state()` (coordinates, actions), `get_subject_info()`.

    *   #### Latent Vector Pool (`src/latent_pool.py`)
        *   **Purpose:** Manages a shared pool of dedicated latent vectors (subject state tokens), each representing the current aggregate state and identity of a specific subject.
        *   **Structure:** A fixed-size (or dynamically resizable) tensor `L_pool` where `L_pool[i]` is the latent vector for `Subject_i`.
        *   **Initialization:** Latents can be initialized randomly, with learned embeddings, or derived from input conditioning.
        *   **Updates:** These latents are designed to be refined by the diffusion model during the generation process, potentially through dedicated cross-attention layers or direct updates guided by input conditioning.
        *   **Efficiency:** Manages allocation and deallocation efficiently, potentially using a free-list or similar strategy to reuse latent slots.

### 2. Spatial Biasing Adapter (SBA) (`src/spatial_biasing/adapter.py`)

*   **Core Functionality:** This module acts as the "plug-and-play" interface between the MSLM and the diffusion model's U-Net. Its primary role is to infuse spatial and subject-specific guidance directly into the U-Net's internal feature maps, ensuring consistent action binding and identity.

*   **Integration Point:** Designed to intercept or augment the attention mechanisms (e.g., self-attention or cross-attention layers) within the U-Net at various resolutions. It can also operate as an external conditioning mechanism applied before or during U-Net processing.

*   **Mechanism:**
    *   Receives `subject_ID`s, their `spatial_coordinates`, and corresponding `latent_tokens` from the `MultiSubjectLatentManager`.

    *   #### Attention-based Biasing (`src/spatial_biasing/attention_mechanisms.py`)
        *   **Coordinate-Aware Attention:** Generates attention masks or weights that prioritize regions of the feature map corresponding to subject bounding boxes or keypoints. This can involve:
            *   **Spatial Grids:** Creating a grid of spatial embeddings that are combined with query/key/value vectors, incorporating subject positions.
            *   **Distance-based Masking:** Applying higher attention weights to pixels/patches closer to a subject's centroid or within its bounding box.
            *   **Sparse Attention:** Focusing computational resources only on relevant spatial regions, reducing overhead.
        *   **Subject-Latent Injection:** The `latent_tokens` from the pool are used as additional conditioning for the U-Net's feature maps. This can be achieved via:
            *   **Cross-Attention:** Using subject latents as keys/values for U-Net feature map queries, or vice-versa.
            *   **Direct Gating/Modulation:** Applying learned transformations of the latent tokens to modulate U-Net features (e.g., via FiLM layers or adaptive normalization).

*   **Scalability & Efficiency:**
    *   **Sparse Attention:** Only computes attention for relevant regions, minimizing quadratic complexity issues as scene complexity grows.
    *   **Parallel Processing:** The biasing logic for multiple subjects can be parallelized.
    *   **Dynamic Allocation:** Latent pool dynamically adjusts to the number of subjects, avoiding fixed-size constraints.

## Integration Workflow (Illustrated in `examples/demo_integration.py`)

1.  **Initialization:** `MultiSubjectLatentManager` and `SpatialBiasingAdapter` are initialized.
2.  **Subject Detection/Input:** A "perception" module (mocked in prototype) provides subject IDs, initial spatial coordinates, and actions.
3.  **Manager Update:** The MSLM receives this information, updates its `SubjectRegistry`, and potentially allocates new `latent_tokens` or retrieves existing ones.
4.  **Diffusion Loop:** At each timestep of the diffusion process:
    *   The current U-Net feature maps are prepared.
    *   The `SpatialBiasingAdapter` queries the MSLM for current subject states (IDs, coordinates, latents).
    *   The SBA generates spatial biasing signals (e.g., attention masks, modulated features) using the subject information.
    *   These signals are injected into the U-Net's internal attention layers or feature processing blocks, guiding the generation to respect subject identities and actions within their specified spatial regions.
    *   The `latent_tokens` themselves might be updated or refined based on U-Net outputs or additional conditioning.

## Getting Started (Placeholder)

This section would typically include:
*   Installation instructions (`pip install -e .` or similar)
*   Dependencies
*   Basic usage examples
*   How to run the provided `demo_integration.py`

## Contributing (Placeholder)

Information on how to contribute to the project, including guidelines for bug reports, feature requests, and pull requests.