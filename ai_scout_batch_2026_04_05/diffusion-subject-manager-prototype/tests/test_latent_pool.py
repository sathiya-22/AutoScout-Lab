```python
import pytest
import torch
import sys
import os

# --- Mock LatentVectorPool Class for Testing ---
# In a real project, this would be imported from `src.latent_pool`.
# For the purpose of this self-contained test file submission,
# we include a functional implementation here.
class LatentVectorPool:
    """
    Manages a pool of dedicated latent vectors (subject state tokens)
    for multi-subject diffusion models.
    """
    def __init__(self, latent_dim: int, max_subjects: int, device: str = "cpu", dtype: torch.dtype = torch.float32):
        if not isinstance(latent_dim, int) or latent_dim <= 0:
            raise ValueError("latent_dim must be a positive integer.")
        if not isinstance(max_subjects, int) or max_subjects < 0:
            raise ValueError("max_subjects must be a non-negative integer.")

        self.latent_dim = latent_dim
        self.max_subjects = max_subjects
        self.device = device
        self.dtype = dtype

        # The core pool of latent vectors, initialized to zeros
        self._pool = torch.zeros((max_subjects, latent_dim), device=self.device, dtype=self.dtype)

        # Track free and allocated indices using sets for O(1) average time complexity
        self._free_indices = set(range(max_subjects))
        self._allocated_indices = set()

    @property
    def num_allocated(self) -> int:
        """Returns the current number of allocated latent vectors."""
        return len(self._allocated_indices)

    @property
    def num_free(self) -> int:
        """Returns the current number of free latent vector slots."""
        return len(self._free_indices)

    def allocate_latent(self, initial_value: torch.Tensor = None) -> int:
        """
        Allocates a latent vector from the pool and returns its index.
        
        Args:
            initial_value (torch.Tensor, optional): A tensor to initialize the
                                                   newly allocated latent vector.
                                                   Must have shape (self.latent_dim,).
                                                   If None, the latent is initialized to zeros.
        Returns:
            int: The index of the allocated latent vector.
            
        Raises:
            RuntimeError: If the pool is full and no free slots are available.
            ValueError: If `initial_value` is provided but has an incorrect shape or type.
        """
        if not self._free_indices:
            raise RuntimeError("LatentVectorPool is full, no free slots available.")

        # Get an arbitrary free index and move it to allocated
        idx = self._free_indices.pop()
        self._allocated_indices.add(idx)

        # Initialize the latent vector
        if initial_value is not None:
            if not isinstance(initial_value, torch.Tensor) or initial_value.shape != (self.latent_dim,):
                raise ValueError(
                    f"initial_value must be a torch.Tensor of shape ({self.latent_dim},). "
                    f"Got shape {initial_value.shape if isinstance(initial_value, torch.Tensor) else type(initial_value)}."
                )
            self._pool[idx] = initial_value.to(self.device, self.dtype)
        else:
            self._pool[idx].zero_() # Default to zeros
        
        return idx

    def deallocate_latent(self, idx: int):
        """
        Deallocates a latent vector, making its slot available for reuse.
        The content of the deallocated slot is cleared (set to zeros).

        Args:
            idx (int): The index of the latent vector to deallocate.

        Raises:
            IndexError: If the index is out of bounds.
            ValueError: If the latent at the given index is not currently allocated.
        """
        if not isinstance(idx, int) or not (0 <= idx < self.max_subjects):
            raise IndexError(f"Invalid index: {idx}. Must be between 0 and {self.max_subjects - 1}.")
        
        if idx not in self._allocated_indices:
            raise ValueError(f"Latent at index {idx} is not currently allocated and cannot be deallocated.")
        
        self._allocated_indices.remove(idx)
        self._free_indices.add(idx)
        self._pool[idx].zero_() # Clear the deallocated latent vector

    def get_latent_vector(self, idx: int) -> torch.Tensor:
        """
        Retrieves a specific allocated latent vector from the pool.
        The returned tensor is a view into the internal pool, so modifications
        will directly affect the pooled latent vector.

        Args:
            idx (int): The index of the latent vector to retrieve.

        Returns:
            torch.Tensor: The latent vector at the specified index.

        Raises:
            IndexError: If the index is out of bounds.
            ValueError: If the latent at the given index is not currently allocated.
        """
        if not isinstance(idx, int) or not (0 <= idx < self.max_subjects):
            raise IndexError(f"Invalid index: {idx}. Must be between 0 and {self.max_subjects - 1}.")
        if idx not in self._allocated_indices:
            raise ValueError(f"Latent at index {idx} is not currently allocated and cannot be accessed.")
        return self._pool[idx]

    def get_all_active_latents(self) -> torch.Tensor:
        """
        Returns a tensor containing all currently active (allocated) latent vectors.
        The order of latents in the returned tensor corresponds to their sorted indices.

        Returns:
            torch.Tensor: A tensor of shape (num_allocated, latent_dim) containing
                          all active latent vectors. Returns an empty tensor if none are allocated.
        """
        if not self._allocated_indices:
            return torch.empty(0, self.latent_dim, device=self.device, dtype=self.dtype)
        
        # Sort indices to ensure consistent order in the returned tensor, useful for debugging/testing
        active_indices_list = sorted(list(self._allocated_indices)) 
        return self._pool[active_indices_list]

    def get_active_indices(self) -> set[int]:
        """
        Returns a copy of the set of currently allocated indices.
        """
        return self._allocated_indices.copy()

    def get_pool_tensor(self) -> torch.Tensor:
        """
        Returns the entire underlying latent vector pool tensor.
        This provides direct access for advanced operations or debugging.
        Modifications to this tensor will affect the individual latent vectors.
        """
        return self._pool

# --- Pytest Fixtures ---

@pytest.fixture
def latent_pool_params():
    """Provides standard parameters for LatentVectorPool initialization."""
    return {
        "latent_dim": 64,
        "max_subjects": 5,
        "device": "cpu", # Use CPU for general testing to avoid CUDA requirement
        "dtype": torch.float32
    }

@pytest.fixture
def empty_latent_pool(latent_pool_params):
    """Provides an initialized but empty LatentVectorPool."""
    return LatentVectorPool(**latent_pool_params)

@pytest.fixture
def full_latent_pool(latent_pool_params):
    """Provides a LatentVectorPool with all slots allocated."""
    pool = LatentVectorPool(**latent_pool_params)
    for _ in range(pool.max_subjects):
        pool.allocate_latent()
    return pool

# --- Test Cases for LatentVectorPool ---

class TestLatentVectorPool:

    def test_initialization(self, latent_pool_params):
        """Test basic initialization and properties."""
        pool = LatentVectorPool(**latent_pool_params)
        assert pool.latent_dim == latent_pool_params["latent_dim"]
        assert pool.max_subjects == latent_pool_params["max_subjects"]
        assert pool.device == latent_pool_params["device"]
        assert pool.dtype == latent_pool_params["dtype"]
        assert pool.num_allocated == 0
        assert pool.num_free == latent_pool_params["max_subjects"]
        assert pool.get_pool_tensor().shape == (pool.max_subjects, pool.latent_dim)
        assert torch.all(pool.get_pool_tensor() == 0) # Initially all zeros

    @pytest.mark.parametrize("latent_dim, max_subjects", [
        (0, 5), (-1, 5), (5, -1), (5, 0)
    ])
    def test_initialization_invalid_params(self, latent_dim, max_subjects):
        """Test initialization with invalid dimension or subject counts."""
        with pytest.raises(ValueError):
            LatentVectorPool(latent_dim=latent_dim, max_subjects=max_subjects)

    def test_allocate_single_latent(self, empty_latent_pool):
        """Test allocating a single latent vector."""
        idx = empty_latent_pool.allocate_latent()
        assert 0 <= idx < empty_latent_pool.max_subjects
        assert empty_latent_pool.num_allocated == 1
        assert empty_latent_pool.num_free == empty_latent_pool.max_subjects - 1
        assert idx in empty_latent_pool.get_active_indices()
        
        latent_vector = empty_latent_pool.get_latent_vector(idx)
        assert latent_vector.shape == (empty_latent_pool.latent_dim,)
        assert torch.all(latent_vector == 0) # Default initialization

    def test_allocate_with_initial_value(self, empty_latent_pool):
        """Test allocating a latent with a specific initial value."""
        initial_value = torch.randn(empty_latent_pool.latent_dim)
        idx = empty_latent_pool.allocate_latent(initial_value=initial_value)
        
        latent_vector = empty_latent_pool.get_latent_vector(idx)
        assert torch.allclose(latent_vector, initial_value)

    def test_allocate_with_invalid_initial_value(self, empty_latent_pool):
        """Test allocating with an initial value of incorrect shape or type."""
        # Invalid shape
        invalid_shape_value = torch.randn(empty_latent_pool.latent_dim + 1)
        with pytest.raises(ValueError, match="initial_value must be a torch.Tensor of shape"):
            empty_latent_pool.allocate_latent(initial_value=invalid_shape_value)
        
        # Invalid type
        invalid_type_value = [1, 2, 3] # Not a torch.Tensor
        with pytest.raises(ValueError, match="initial_value must be a torch.Tensor of shape"):
            empty_latent_pool.allocate_latent(initial_value=invalid_type_value)


    def test_allocate_multiple_latents(self, empty_latent_pool):
        """Test allocating all available latent vectors."""
        allocated_indices = []
        for i in range(empty_latent_pool.max_subjects):
            idx = empty_latent_pool.allocate_latent()
            allocated_indices.append(idx)
            assert empty_latent_pool.num_allocated == i + 1
            assert empty_latent_pool.num_free == empty_latent_pool.max_subjects - (i + 1)
        
        assert len(set(allocated_indices)) == empty_latent_pool.max_subjects # All unique
        assert empty_latent_pool.num_allocated == empty_latent_pool.max_subjects
        assert empty_latent_pool.num_free == 0
        assert empty_latent_pool.get_active_indices() == set(range(empty_latent_pool.max_subjects))

    def test_allocate_when_full(self, full_latent_pool):
        """Test that allocation fails when the pool is full."""
        with pytest.raises(RuntimeError, match="LatentVectorPool is full"):
            full_latent_pool.allocate_latent()

    def test_deallocate_latent(self, empty_latent_pool):
        """Test deallocating an allocated latent vector."""
        idx1 = empty_latent_pool.allocate_latent()
        _ = empty_latent_pool.allocate_latent() # Allocate another to ensure other latents exist

        assert empty_latent_pool.num_allocated == 2
        assert empty_latent_pool.num_free == empty_latent_pool.max_subjects - 2

        empty_latent_pool.deallocate_latent(idx1)
        assert empty_latent_pool.num_allocated == 1
        assert empty_latent_pool.num_free == empty_latent_pool.max_subjects - 1
        assert idx1 not in empty_latent_pool.get_active_indices()
        assert idx1 in empty_latent_pool._free_indices # Check if it's back in free list
        assert torch.all(empty_latent_pool.get_pool_tensor()[idx1] == 0) # Check cleared

    @pytest.mark.parametrize("invalid_idx", [-1, 999, "abc", None])
    def test_deallocate_invalid_index(self, empty_latent_pool, invalid_idx):
        """Test deallocating with an out-of-bounds or invalid type index."""
        # Allocate one to ensure the pool is not trivially empty and has allocated indices
        if empty_latent_pool.max_subjects > 0:
            empty_latent_pool.allocate_latent()
        with pytest.raises((IndexError, ValueError)):
            empty_latent_pool.deallocate_latent(invalid_idx)
    
    def test_deallocate_unallocated_index(self, empty_latent_pool):
        """Test deallocating an index that was never allocated or already freed."""
        if empty_latent_pool.max_subjects == 0:
            pytest.skip("Cannot test unallocated index if max_subjects is 0.")

        # Allocate and then deallocate, making `idx` free
        idx = empty_latent_pool.allocate_latent()
        empty_latent_pool.deallocate_latent(idx)
        with pytest.raises(ValueError, match="is not currently allocated"):
            empty_latent_pool.deallocate_latent(idx) # Try to deallocate again

        # Try to deallocate an index that exists but was never allocated
        unallocated_idx = (idx + 1) % empty_latent_pool.max_subjects
        if unallocated_idx in empty_latent_pool._free_indices: # Ensure it's truly a free slot
            with pytest.raises(ValueError, match="is not currently allocated"):
                empty_latent_pool.deallocate_latent(unallocated_idx)

    def test_reuse_deallocated_slot(self, empty_latent_pool):
        """Test that a deallocated slot can be reused for new allocations."""
        if empty_latent_pool.max_subjects < 2:
            pytest.skip("Requires at least 2 max_subjects for this test.")

        idx1 = empty_latent_pool.allocate_latent() # Allocate index 0 (assuming set.pop() behavior)
        idx2 = empty_latent_pool.allocate_latent() # Allocate index 1
        empty_latent_pool.deallocate_latent(idx1) # Free index 0

        # Now, allocate again. The pool should reuse a free slot, ideally idx1
        new_idx = empty_latent_pool.allocate_latent()
        assert new_idx in empty_latent_pool.get_active_indices()
        assert new_idx == idx1 # Should reuse the lowest available index first

    def test_get_latent_vector(self, empty_latent_pool):
        """Test retrieving an allocated latent vector."""
        initial_value = torch.ones(empty_latent_pool.latent_dim)
        idx = empty_latent_pool.allocate_latent(initial_value=initial_value)
        
        retrieved_latent = empty_latent_pool.get_latent_vector(idx)
        assert torch.allclose(retrieved_latent, initial_value)
        
        # Test modification through the retrieved tensor reflects in the pool
        test_val = 0.5
        retrieved_latent.fill_(test_val)
        assert torch.all(empty_latent_pool.get_pool_tensor()[idx] == test_val)

    @pytest.mark.parametrize("invalid_idx", [-1, 999, "abc", None])
    def test_get_latent_vector_invalid_index(self, empty_latent_pool, invalid_idx):
        """Test retrieving a latent with an out-of-bounds or invalid type index."""
        # Allocate one to ensure there's at least one allocated and pool isn't empty
        if empty_latent_pool.max_subjects > 0:
            empty_latent_pool.allocate_latent()
        with pytest.raises((IndexError, ValueError)):
            empty_latent_pool.get_latent_vector(invalid_idx)

    def test_get_latent_vector_unallocated(self, empty_latent_pool):
        """Test retrieving a latent at an existing but unallocated index."""
        if empty_latent_pool.max_subjects == 0:
            pytest.skip("Cannot test unallocated index if max_subjects is 0.")
        
        # Ensure there's at least one allocated for robust test, but we target an unallocated one
        if empty_latent_pool.max_subjects > 0:
            allocated_idx = empty_latent_pool.allocate_latent()
            # Find an unallocated index
            unallocated_idx = -1
            for i in range(empty_latent_pool.max_subjects):
                if i != allocated_idx:
                    unallocated_idx = i
                    break
            
            if unallocated_idx != -1: # Found an unallocated slot
                with pytest.raises(ValueError, match="is not currently allocated"):
                    empty_latent_pool.get_latent_vector(unallocated_idx)
            else:
                 pytest.skip("Could not find an unallocated index, likely max_subjects is 1 and it's allocated.")
        else:
            pytest.skip("Cannot test unallocated index if max_subjects is 0.")


    def test_get_all_active_latents_empty(self, empty_latent_pool):
        """Test retrieving all active latents when the pool is empty."""
        active_latents = empty_latent_pool.get_all_active_latents()
        assert active_latents.shape == (0, empty_latent_pool.latent_dim)
        assert active_latents.device == empty_latent_pool.device
        assert active_latents.dtype == empty_latent_pool.dtype

    def test_get_all_active_latents_some_allocated(self, empty_latent_pool):
        """Test retrieving all active latents with a mix of allocated/free slots."""
        if empty_latent_pool.max_subjects < 3:
            pytest.skip("Requires at least 3 max_subjects for this test.")

        latent1_val = torch.ones(empty_latent_pool.latent_dim)
        latent2_val = torch.full((empty_latent_pool.latent_dim,), 2.0)
        latent3_val = torch.full((empty_latent_pool.latent_dim,), 3.0)

        idx1 = empty_latent_pool.allocate_latent(initial_value=latent1_val)
        idx2 = empty_latent_pool.allocate_latent(initial_value=latent2_val)
        idx3 = empty_latent_pool.allocate_latent(initial_value=latent3_val)

        active_latents = empty_latent_pool.get_all_active_latents()
        assert active_latents.shape == (3, empty_latent_pool.latent_dim)
        
        # Collect expected and actual values for comparison, as order in tensor is sorted by index
        expected_values = {tuple(latent1_val.tolist()), tuple(latent2_val.tolist()), tuple(latent3_val.tolist())}
        actual_values = {tuple(lv.tolist()) for lv in active_latents}
        assert expected_values == actual_values
        
        # Deallocate one and check again
        empty_latent_pool.deallocate_latent(idx2)
        active_latents_after_dealloc = empty_latent_pool.get_all_active_latents()
        assert active_latents_after_dealloc.shape == (2, empty_latent_pool.latent_dim)
        
        expected_values_after_dealloc = {tuple(latent1_val.tolist()), tuple(latent3_val.tolist())}
        actual_values_after_dealloc = {tuple(lv.tolist()) for lv in active_latents_after_dealloc}
        assert expected_values_after_dealloc == actual_values_after_dealloc

    def test_get_active_indices(self, empty_latent_pool):
        """Test retrieving the set of active indices."""
        assert empty_latent_pool.get_active_indices() == set()
        
        idx1 = empty_latent_pool.allocate_latent()
        idx2 = empty_latent_pool.allocate_latent()
        
        assert empty_latent_pool.get_active_indices() == {idx1, idx2}
        
        empty_latent_pool.deallocate_latent(idx1)
        assert empty_latent_pool.get_active_indices() == {idx2}

    def test_get_pool_tensor(self, empty_latent_pool):
        """Test retrieving the full underlying pool tensor."""
        pool_tensor = empty_latent_pool.get_pool_tensor()
        assert isinstance(pool_tensor, torch.Tensor)
        assert pool_tensor.shape == (empty_latent_pool.max_subjects, empty_latent_pool.latent_dim)
        assert pool_tensor.device == empty_latent_pool.device
        assert pool_tensor.dtype == empty_latent_pool.dtype

        # Ensure it's the actual underlying tensor, not a copy, by modifying it
        if empty_latent_pool.max_subjects > 0:
            idx = empty_latent_pool.allocate_latent()
            test_val = 0.777
            pool_tensor[idx].fill_(test_val)
            assert torch.all(empty_latent_pool.get_latent_vector(idx) == test_val)
        else:
            # If max_subjects is 0, allocation will fail, so skip direct modification test
            pass # The initial shape and type assertions are sufficient for max_subjects=0
```