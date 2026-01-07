"""
Utilities for ensuring reproducible experiments with deterministic random number generation.
"""

import random
import warnings

import numpy as np
import torch
import os


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set random seeds for all libraries to ensure reproducibility.

    This function configures all random number generators (Python, NumPy, PyTorch)
    to use the same seed, ensuring that runs with the same seed produce identical results.

    Args:
        seed: The random seed to use across all libraries
        deterministic: If True, enables PyTorch's deterministic mode for CUDA operations.
                      This may impact performance but ensures full reproducibility.

    Note:
        When deterministic=True, some PyTorch operations may be slower and some
        operations may not have deterministic implementations. In such cases,
        PyTorch will raise an error.
    """
    # Python's built-in random module
    random.seed(seed)

    # NumPy random number generator
    np.random.seed(seed)

    # PyTorch random number generators
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    if deterministic:
        # show warning
        warnings.warn(
            "PyTorch deterministic mode is enabled. This may impact performance. "
            "Some operations may be slower or not have deterministic implementations and may raise errors."
        )

        # Enable deterministic algorithms in PyTorch
        # This ensures that CUDA operations are deterministic
        torch.use_deterministic_algorithms(True)

        # Some PyTorch operations (e.g., certain CuDNN operations) don't have
        # deterministic implementations. This flag forces them to be deterministic
        # at the cost of performance.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Set environment variable for deterministic operations
        # This is needed for some operations like torch.index_add
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def get_generator(seed: int, device: str = "cpu") -> torch.Generator:
    """
    Create a PyTorch random number generator with a specific seed.

    Useful for operations that accept a generator parameter to ensure
    local reproducibility without affecting the global RNG state.

    Args:
        seed: Seed for the generator
        device: Device for the generator ('cpu' or 'cuda')

    Returns:
        A seeded torch.Generator instance
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator
