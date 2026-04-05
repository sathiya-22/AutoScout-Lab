import torch

class LatentVectorPool:
    """
    Manages a shared pool of dedicated latent vectors (subject state tokens).

    Each latent vector represents the current aggregate state and identity of a specific subject.
    The pool efficiently handles allocation, deallocation, and retrieval of these vectors.
    """

    def __init__(self, max_capacity: int, latent_dim: int, device: str = "cpu", dtype: torch.dtype = torch.float32):
        """
        Initializes the LatentVectorPool.

        Args:
            max_capacity (int): The maximum number of latent vectors the pool can hold.
            latent_dim (int): The dimensionality of each latent vector.
            device (str): The PyTorch device ('cpu' or 'cuda').
            dtype (torch.dtype): The data type for the latent vectors (e.g., torch.float32).
        """
        if not isinstance(max_capacity, int) or max_capacity <= 0:
            raise ValueError("max_capacity must be a positive integer.")
        if not isinstance(latent_dim, int) or latent_dim <= 0:
            raise ValueError("latent_dim must be a positive integer.")
        if not isinstance(device, str) or device not in ["cpu", "cuda"]:
            if not torch.cuda.is_available() and device == "cuda":
                raise ValueError("CUDA is not available, but 'cuda' device was specified.")
            # Allow other device strings if PyTorch supports them later, but warn for now
            # For this prototype, strict "cpu" or "cuda" is sufficient.
            if device != "cpu" and not device.startswith("cuda"):
                print(f"Warning: Using an unconventional device string '{device}'. Ensure PyTorch supports it.")
        if not isinstance(dtype, torch.dtype):
            raise ValueError("dtype must be a torch.dtype (e.g., torch.float32).")

        self.max_capacity = max_capacity
        self.latent_dim = latent_dim
        self.device = torch.device(device)
        self.dtype = dtype

        # Main storage for latent vectors. Initialized to zeros.
        # This tensor will store all latents, active and inactive.
        self.latent_pool = torch.zeros(
            (self.max_capacity, self.latent_dim),
            device=self.device,
            dtype=self.dtype
        )

        # Free-list: A stack of available indices for efficient allocation.
        self.free_indices = list(range(self.max_capacity))
        # Set of indices currently allocated and in use by subjects.
        self.active_indices = set()

    def _initialize_new_latent(self, index: int, init_method: str = "zeros", **kwargs):
        """
        Initializes a new latent vector at the given index using the specified method.

        Args:
            index (int): The index in the latent pool to initialize.
            init_method (str): Method for initialization ('zeros', 'random', 'embedding').
            **kwargs: Additional arguments for specific initialization methods.
                      For 'embedding', 'embedding' (torch.Tensor) must be provided.
        """
        if not (0 <= index < self.max_capacity):
            raise IndexError(f"Index {index} is out of bounds for the latent pool.")

        if init_method == "random":
            self.latent_pool[index] = torch.randn(self.latent_dim, device=self.device, dtype=self.dtype)
        elif init_method == "zeros":
            self.latent_pool[index] = torch.zeros(self.latent_dim, device=self.device, dtype=self.dtype)
        elif init_method == "embedding":
            embedding = kwargs.get('embedding')
            if embedding is None or not isinstance(embedding, torch.Tensor):
                raise ValueError("For 'embedding' init_method, a 'embedding' tensor must be provided in kwargs.")
            if embedding.shape != (self.latent_dim,):
                raise ValueError(
                    f"Provided embedding shape {embedding.shape} does not match expected latent_dim ({self.latent_dim})."
                )
            self.latent_pool[index] = embedding.to(self.device, self.dtype)
        else:
            raise ValueError(f"Unsupported latent initialization method: '{init_method}'. "
                             "Choose from 'zeros', 'random', or 'embedding'.")

    def allocate_latent(self, init_method: str = "zeros", **kwargs) -> int:
        """
        Allocates a new latent vector slot from the pool.
        The allocated latent is initialized based on `init_method`.

        Args:
            init_method (str): Method for initialization ('zeros', 'random', 'embedding').
            **kwargs: Arguments for initialization (e.g., 'embedding' tensor for 'embedding' method).

        Returns:
            int: The index of the allocated latent vector slot.

        Raises:
            RuntimeError: If the latent pool is full and cannot allocate more vectors.
            ValueError: If an unsupported initialization method is provided or required arguments are missing.
        """
        if not self.free_indices:
            raise RuntimeError("Latent pool is full. Cannot allocate more latent vectors.")

        index = self.free_indices.pop()
        self.active_indices.add(index)
        self._initialize_new_latent(index, init_method, **kwargs)
        return index

    def deallocate_latent(self, index: int):
        """
        Deallocates a latent vector slot, marking it as free for reuse.
        The latent vector at the deallocated index is effectively cleared (zeroed out) for cleanliness.

        Args:
            index (int): The index of the latent vector to deallocate.

        Raises:
            ValueError: If the index is not active (not currently allocated).
            IndexError: If the index is out of the valid pool range.
        """
        if not (0 <= index < self.max_capacity):
            raise IndexError(f"Index {index} is out of bounds for latent pool (capacity: {self.max_capacity}).")
        if index not in self.active_indices:
            raise ValueError(f"Latent vector at index {index} is not currently active and cannot be deallocated.")

        self.active_indices.remove(index)
        self.free_indices.append(index)
        # Zero out the memory for security/cleanliness.
        self.latent_pool[index].zero_()

    def get_latent(self, index: int) -> torch.Tensor:
        """
        Retrieves the latent vector at the given index.

        Args:
            index (int): The index of the latent vector to retrieve.

        Returns:
            torch.Tensor: The latent vector.

        Raises:
            ValueError: If the index is not active.
            IndexError: If the index is out of the valid pool range.
        """
        if not (0 <= index < self.max_capacity):
            raise IndexError(f"Index {index} is out of bounds for latent pool (capacity: {self.max_capacity}).")
        if index not in self.active_indices:
            raise ValueError(f"Latent vector at index {index} is not active. It might be free or never allocated.")
        return self.latent_pool[index]

    def set_latent(self, index: int, vector: torch.Tensor):
        """
        Updates the latent vector at the given index with a new vector.

        Args:
            index (int): The index of the latent vector to update.
            vector (torch.Tensor): The new latent vector. Must match `self.latent_dim` shape.

        Raises:
            ValueError: If the index is not active or if the input vector's shape is incorrect.
            IndexError: If the index is out of the valid pool range.
        """
        if not (0 <= index < self.max_capacity):
            raise IndexError(f"Index {index} is out of bounds for latent pool (capacity: {self.max_capacity}).")
        if index not in self.active_indices:
            raise ValueError(f"Latent vector at index {index} is not active. It must be allocated before being set.")
        if vector.shape != (self.latent_dim,):
            raise ValueError(
                f"Input vector shape {vector.shape} does not match expected latent_dim ({self.latent_dim})."
            )

        self.latent_pool[index] = vector.to(self.device, self.dtype)

    def get_active_latents_tensor(self) -> torch.Tensor:
        """
        Returns a single tensor containing all currently active latent vectors.
        The order of latents in the returned tensor is sorted by their indices.

        Returns:
            torch.Tensor: A tensor of shape (num_active_latents, latent_dim).
                          Returns an empty tensor if no latents are active.
        """
        if not self.active_indices:
            return torch.empty(0, self.latent_dim, device=self.device, dtype=self.dtype)

        # Convert set to sorted list to ensure consistent ordering when returning
        sorted_active_indices = sorted(list(self.active_indices))
        return self.latent_pool[sorted_active_indices]

    def get_latents_by_indices(self, indices: list[int]) -> torch.Tensor:
        """
        Returns a tensor containing latent vectors for a specific list of indices.
        All provided indices must correspond to active latents.

        Args:
            indices (list[int]): A list of indices of the desired latent vectors.

        Returns:
            torch.Tensor: A tensor of shape (len(indices), latent_dim).
                          Returns an empty tensor if the input list is empty.

        Raises:
            ValueError: If any provided index is not active.
            IndexError: If any provided index is out of the valid pool range.
        """
        if not indices:
            return torch.empty(0, self.latent_dim, device=self.device, dtype=self.dtype)

        for idx in indices:
            if not (0 <= idx < self.max_capacity):
                raise IndexError(f"Requested index {idx} is out of bounds for latent pool (capacity: {self.max_capacity}).")
            if idx not in self.active_indices:
                raise ValueError(f"Latent vector at requested index {idx} is not active.")
        
        # Use advanced indexing to efficiently gather the latents
        return self.latent_pool[indices]

    @property
    def num_active_latents(self) -> int:
        """Returns the current number of active (allocated) latent vectors."""
        return len(self.active_indices)

    @property
    def current_capacity(self) -> int:
        """Returns the maximum capacity of the latent pool."""
        return self.max_capacity

    def __len__(self) -> int:
        """Returns the current number of active (allocated) latent vectors."""
        return self.num_active_latents

    def __repr__(self) -> str:
        """Provides a string representation of the LatentVectorPool."""
        return (f"LatentVectorPool(max_capacity={self.max_capacity}, latent_dim={self.latent_dim}, "
                f"num_active={self.num_active_latents}, device='{self.device}', dtype={self.dtype})")