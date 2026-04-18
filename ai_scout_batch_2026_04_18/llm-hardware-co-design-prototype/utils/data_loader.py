import numpy as np

class LLMComponentDataLoader:
    """
    A utility class for generating synthetic data for various LLM components
    to be used in the hardware/software co-design prototype.

    This loader focuses on generating tensor shapes and numerical characteristics
    representative of LLM inference inputs, rather than actual textual data.
    """
    def __init__(self,
                 batch_size: int = 1,
                 sequence_length: int = 512,
                 hidden_dim: int = 768,
                 num_heads: int = 12,
                 vocab_size: int = 30000,
                 dtype=np.float32,
                 seed: int = None):
        """
        Initializes the data loader with common LLM component dimensions.

        Args:
            batch_size (int): The number of independent sequences in a batch.
            sequence_length (int): The length of each sequence.
            hidden_dim (int): The dimensionality of the hidden states/embeddings.
            num_heads (int): The number of attention heads (for attention components).
            vocab_size (int): The size of the vocabulary (for embedding inputs).
            dtype (np.dtype): The data type for generated tensors (e.g., np.float32, np.int32).
            seed (int, optional): Random seed for reproducibility.
        """
        if not all(isinstance(arg, int) and arg > 0 for arg in [batch_size, sequence_length, hidden_dim, num_heads, vocab_size]):
            raise ValueError("All dimension parameters (batch_size, sequence_length, hidden_dim, num_heads, vocab_size) must be positive integers.")

        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.dtype = dtype

        if seed is not None:
            np.random.seed(seed)

    def _generate_tensor(self, shape: tuple, data_type, sparsity: float = 0.0, low: float = -1.0, high: float = 1.0):
        """
        Helper method to generate a synthetic tensor with optional sparsity.

        Args:
            shape (tuple): The desired shape of the tensor.
            data_type (np.dtype): The data type for the tensor.
            sparsity (float): The proportion of zeros in the tensor (0.0 to 1.0).
                              Only applies to float types.
            low (float): Lower bound for random values.
            high (float): Upper bound for random values.

        Returns:
            np.ndarray: The generated synthetic tensor.

        Raises:
            ValueError: If sparsity is outside the range [0, 1].
        """
        if not (0.0 <= sparsity <= 1.0):
            raise ValueError("Sparsity must be between 0.0 and 1.0.")
        if any(d <= 0 for d in shape):
            raise ValueError(f"All dimensions in shape must be positive, got {shape}")

        if np.issubdtype(data_type, np.integer):
            # For integer types, sparsity might be less meaningful or handled differently
            # For now, generate random integers.
            tensor = np.random.randint(low=int(low), high=int(high), size=shape, dtype=data_type)
        else: # float types
            tensor = np.random.uniform(low=low, high=high, size=shape).astype(data_type)
            if sparsity > 0.0:
                num_elements = np.prod(shape)
                num_zeros = int(num_elements * sparsity)
                flat_indices = np.random.choice(num_elements, num_zeros, replace=False)
                tensor.ravel()[flat_indices] = 0.0
        return tensor

    def get_token_ids(self, sequence_length: int = None, vocab_size: int = None) -> np.ndarray:
        """
        Generates synthetic token IDs for an embedding layer.

        Args:
            sequence_length (int, optional): Override default sequence length.
            vocab_size (int, optional): Override default vocabulary size.

        Returns:
            np.ndarray: A tensor of shape (batch_size, sequence_length) with token IDs.
        """
        seq_len = sequence_length if sequence_length is not None else self.sequence_length
        vocab = vocab_size if vocab_size is not None else self.vocab_size
        return self._generate_tensor(
            shape=(self.batch_size, seq_len),
            data_type=np.int32,  # Token IDs are typically integers
            low=0,
            high=vocab
        )

    def get_attention_inputs(self,
                             sequence_length: int = None,
                             hidden_dim: int = None,
                             num_heads: int = None,
                             qkv_dim: int = None, # dimension of K/V per head or Q per head
                             sparsity: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates synthetic Query, Key, and Value tensors for an attention mechanism.

        Args:
            sequence_length (int, optional): Override default sequence length.
            hidden_dim (int, optional): Override default hidden dimension.
            num_heads (int, optional): Override default number of heads.
            qkv_dim (int, optional): Dimension per head. If None, defaults to hidden_dim // num_heads.
            sparsity (float): Sparsity percentage for Q, K, V tensors.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Q, K, V tensors.
                Each tensor has shape (batch_size, num_heads, sequence_length, head_dim).
        """
        seq_len = sequence_length if sequence_length is not None else self.sequence_length
        h_dim = hidden_dim if hidden_dim is not None else self.hidden_dim
        n_heads = num_heads if num_heads is not None else self.num_heads
        head_dim = qkv_dim if qkv_dim is not None else (h_dim // n_heads)

        if h_dim % n_heads != 0 and qkv_dim is None:
            raise ValueError(f"Hidden dimension ({h_dim}) must be divisible by number of heads ({n_heads}).")

        qkv_shape = (self.batch_size, n_heads, seq_len, head_dim)

        query = self._generate_tensor(qkv_shape, self.dtype, sparsity=sparsity)
        key = self._generate_tensor(qkv_shape, self.dtype, sparsity=sparsity)
        value = self._generate_tensor(qkv_shape, self.dtype, sparsity=sparsity)

        return query, key, value

    def get_feed_forward_inputs(self,
                                hidden_dim: int = None,
                                sequence_length: int = None,
                                sparsity: float = 0.0) -> np.ndarray:
        """
        Generates synthetic input for a feed-forward network layer.

        Args:
            hidden_dim (int, optional): Override default hidden dimension.
            sequence_length (int, optional): Override default sequence length.
            sparsity (float): Sparsity percentage for the input tensor.

        Returns:
            np.ndarray: An input tensor of shape (batch_size, sequence_length, hidden_dim).
        """
        h_dim = hidden_dim if hidden_dim is not None else self.hidden_dim
        seq_len = sequence_length if sequence_length is not None else self.sequence_length
        input_shape = (self.batch_size, seq_len, h_dim)
        return self._generate_tensor(input_shape, self.dtype, sparsity=sparsity)

    def get_sparse_feed_forward_inputs(self,
                                       hidden_dim: int = None,
                                       sequence_length: int = None,
                                       sparsity_percentage: float = 0.8) -> np.ndarray:
        """
        Generates synthetic input for a sparse feed-forward network layer,
        ensuring a high degree of sparsity.

        Args:
            hidden_dim (int, optional): Override default hidden dimension.
            sequence_length (int, optional): Override default sequence length.
            sparsity_percentage (float): The target sparsity for the input tensor (e.g., 0.8 for 80% zeros).

        Returns:
            np.ndarray: A sparse input tensor of shape (batch_size, sequence_length, hidden_dim).

        Raises:
            ValueError: If sparsity_percentage is outside the range [0, 1].
        """
        if not (0.0 <= sparsity_percentage <= 1.0):
            raise ValueError("Sparsity percentage must be between 0.0 and 1.0.")

        h_dim = hidden_dim if hidden_dim is not None else self.hidden_dim
        seq_len = sequence_length if sequence_length is not None else self.sequence_length
        input_shape = (self.batch_size, seq_len, h_dim)
        return self._generate_tensor(input_shape, self.dtype, sparsity=sparsity_percentage)

    def get_linear_layer_inputs(self,
                                in_features: int = None,
                                batch_size: int = None,
                                out_features: int = None,
                                sparsity: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates synthetic input and weights for a generic linear layer (matrix multiplication).

        Args:
            in_features (int, optional): Input features dimension. Defaults to hidden_dim.
            batch_size (int, optional): Override default batch size for the input.
            out_features (int, optional): Output features dimension.
            sparsity (float): Sparsity percentage for the input tensor.

        Returns:
            tuple[np.ndarray, np.ndarray]: Input tensor and weight matrix.
                Input shape: (batch_size, in_features)
                Weight shape: (in_features, out_features)

        Raises:
            ValueError: If out_features is not provided or non-positive.
        """
        if out_features is None or out_features <= 0:
            raise ValueError("out_features must be provided and positive for linear layer inputs.")

        b_size = batch_size if batch_size is not None else self.batch_size
        in_feat = in_features if in_features is not None else self.hidden_dim

        input_tensor = self._generate_tensor(
            shape=(b_size, in_feat),
            data_type=self.dtype,
            sparsity=sparsity
        )
        weight_matrix = self._generate_tensor(
            shape=(in_feat, out_features),
            data_type=self.dtype,
            sparsity=0.0 # Weights are typically dense unless specific sparse training is simulated
        )
        return input_tensor, weight_matrix