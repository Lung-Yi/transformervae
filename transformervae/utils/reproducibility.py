"""
Reproducibility utilities for TransformerVAE.

This module provides functions for ensuring reproducible experiments
through proper random seed management.
"""

import random
import numpy as np
import torch
import os
from typing import Optional


def set_random_seeds(seed: int, use_deterministic: bool = True) -> None:
    """
    Set random seeds for reproducible results.

    Args:
        seed: Random seed value
        use_deterministic: Whether to use deterministic algorithms (slower but more reproducible)
    """
    # Python random
    random.seed(seed)

    # NumPy random
    np.random.seed(seed)

    # PyTorch random
    torch.manual_seed(seed)

    # CUDA random (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if use_deterministic:
            # Make operations deterministic (may impact performance)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # Set environment variable for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"Random seeds set to {seed} (deterministic={use_deterministic})")


def get_random_state() -> dict:
    """
    Get current random state for all random number generators.

    Returns:
        Dictionary containing random states
    """
    state = {
        'python_random': random.getstate(),
        'numpy_random': np.random.get_state(),
        'torch_random': torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state['torch_cuda_random'] = torch.cuda.get_rng_state()

    return state


def set_random_state(state: dict) -> None:
    """
    Restore random state for all random number generators.

    Args:
        state: Dictionary containing random states
    """
    if 'python_random' in state:
        random.setstate(state['python_random'])

    if 'numpy_random' in state:
        np.random.set_state(state['numpy_random'])

    if 'torch_random' in state:
        torch.set_rng_state(state['torch_random'])

    if 'torch_cuda_random' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state(state['torch_cuda_random'])


def create_reproducible_dataloader(dataset, batch_size: int, shuffle: bool = True,
                                 num_workers: int = 0, seed: Optional[int] = None) -> torch.utils.data.DataLoader:
    """
    Create a reproducible DataLoader.

    Args:
        dataset: Dataset to load from
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        seed: Random seed for worker initialization

    Returns:
        Reproducible DataLoader
    """
    def worker_init_fn(worker_id):
        if seed is not None:
            worker_seed = seed + worker_id
            random.seed(worker_seed)
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn if seed is not None else None,
        generator=torch.Generator().manual_seed(seed) if seed is not None and shuffle else None
    )


def save_experiment_state(filepath: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                         scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                         epoch: int = 0, additional_state: Optional[dict] = None) -> None:
    """
    Save complete experiment state for reproducibility.

    Args:
        filepath: Path to save state
        model: Model to save
        optimizer: Optimizer to save
        scheduler: Optional scheduler to save
        epoch: Current epoch
        additional_state: Additional state to save
    """
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'random_state': get_random_state(),
    }

    if scheduler is not None:
        state['scheduler_state_dict'] = scheduler.state_dict()

    if additional_state is not None:
        state['additional_state'] = additional_state

    torch.save(state, filepath)
    print(f"Experiment state saved to {filepath}")


def load_experiment_state(filepath: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                         scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                         device: Optional[torch.device] = None) -> tuple:
    """
    Load complete experiment state for reproducibility.

    Args:
        filepath: Path to load state from
        model: Model to load state into
        optimizer: Optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to map tensors to

    Returns:
        Tuple of (epoch, additional_state)
    """
    if device is None:
        device = next(model.parameters()).device

    state = torch.load(filepath, map_location=device)

    # Load model state
    model.load_state_dict(state['model_state_dict'])

    # Load optimizer state
    optimizer.load_state_dict(state['optimizer_state_dict'])

    # Load scheduler state if available
    if scheduler is not None and 'scheduler_state_dict' in state:
        scheduler.load_state_dict(state['scheduler_state_dict'])

    # Restore random state
    if 'random_state' in state:
        set_random_state(state['random_state'])

    epoch = state.get('epoch', 0)
    additional_state = state.get('additional_state', {})

    print(f"Experiment state loaded from {filepath}")
    print(f"Resumed from epoch {epoch}")

    return epoch, additional_state


def ensure_reproducibility(func):
    """
    Decorator to ensure function runs with reproducible random state.

    Usage:
        @ensure_reproducibility
        def my_function():
            # Function will run with saved random state
            pass
    """
    def wrapper(*args, **kwargs):
        # Save current random state
        state = get_random_state()

        try:
            # Run function
            result = func(*args, **kwargs)
        finally:
            # Restore random state
            set_random_state(state)

        return result

    return wrapper


class ReproducibilityContext:
    """Context manager for reproducible code blocks."""

    def __init__(self, seed: Optional[int] = None, use_deterministic: bool = True):
        """
        Initialize reproducibility context.

        Args:
            seed: Random seed to use (if None, saves and restores current state)
            use_deterministic: Whether to use deterministic algorithms
        """
        self.seed = seed
        self.use_deterministic = use_deterministic
        self.original_state = None
        self.original_deterministic = None
        self.original_benchmark = None

    def __enter__(self):
        # Save original state
        self.original_state = get_random_state()

        if torch.cuda.is_available():
            self.original_deterministic = torch.backends.cudnn.deterministic
            self.original_benchmark = torch.backends.cudnn.benchmark

        # Set new state
        if self.seed is not None:
            set_random_seeds(self.seed, self.use_deterministic)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original state
        set_random_state(self.original_state)

        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = self.original_deterministic
            torch.backends.cudnn.benchmark = self.original_benchmark


def validate_reproducibility(func, seed: int = 42, num_runs: int = 3) -> bool:
    """
    Validate that a function produces reproducible results.

    Args:
        func: Function to test (should return a deterministic value)
        seed: Seed to use for testing
        num_runs: Number of runs to compare

    Returns:
        True if results are reproducible
    """
    results = []

    for _ in range(num_runs):
        with ReproducibilityContext(seed=seed):
            result = func()
            results.append(result)

    # Check if all results are equal
    first_result = results[0]

    if isinstance(first_result, torch.Tensor):
        return all(torch.equal(first_result, result) for result in results[1:])
    elif isinstance(first_result, np.ndarray):
        return all(np.array_equal(first_result, result) for result in results[1:])
    else:
        return all(first_result == result for result in results[1:])


def get_system_info() -> dict:
    """
    Get system information for reproducibility tracking.

    Returns:
        Dictionary containing system information
    """
    import platform
    import sys

    info = {
        'python_version': sys.version,
        'platform': platform.platform(),
        'processor': platform.processor(),
        'torch_version': torch.__version__,
        'numpy_version': np.__version__,
    }

    if torch.cuda.is_available():
        info.update({
            'cuda_available': True,
            'cuda_version': torch.version.cuda,
            'cudnn_version': torch.backends.cudnn.version(),
            'gpu_name': torch.cuda.get_device_name(0),
            'gpu_count': torch.cuda.device_count(),
        })
    else:
        info['cuda_available'] = False

    return info


def log_reproducibility_info(seed: int, output_file: Optional[str] = None) -> None:
    """
    Log comprehensive reproducibility information.

    Args:
        seed: Random seed used
        output_file: Optional file to save information to
    """
    info = {
        'seed': seed,
        'system_info': get_system_info(),
        'random_state': get_random_state(),
    }

    # Format information
    output = f"""
Reproducibility Information
==========================
Random Seed: {seed}

System Information:
"""

    for key, value in info['system_info'].items():
        output += f"  {key}: {value}\n"

    print(output)

    if output_file:
        import json
        with open(output_file, 'w') as f:
            # Convert torch tensors to lists for JSON serialization
            serializable_info = {}
            for key, value in info.items():
                if key == 'random_state':
                    serializable_info[key] = 'saved_separately'
                else:
                    serializable_info[key] = value

            json.dump(serializable_info, f, indent=2)

        print(f"Reproducibility information saved to {output_file}")