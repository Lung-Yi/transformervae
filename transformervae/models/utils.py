"""
Model utility functions for TransformerVAE.

This module provides utility functions for model management including
parameter counting, device management, and model summaries.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Union, Optional
from transformervae.models.model import ModelInterface


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count the number of parameters in a model.

    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters

    Returns:
        Total number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def model_summary(model: ModelInterface) -> Dict[str, Any]:
    """
    Generate a detailed summary of the model architecture.

    Args:
        model: TransformerVAE model instance

    Returns:
        Dictionary containing model statistics
    """
    summary = {
        'total_parameters': count_parameters(model),
        'trainable_parameters': count_parameters(model, trainable_only=True),
        'encoder_parameters': 0,
        'decoder_parameters': 0,
        'sampler_parameters': 0,
        'head_parameters': 0,
        'latent_dim': model.latent_dim,
        'model_type': model.__class__.__name__
    }

    # Count parameters by component
    if hasattr(model, 'encoder_layers'):
        summary['encoder_parameters'] = sum(
            count_parameters(layer) for layer in model.encoder_layers
        )

    if hasattr(model, 'decoder_layers'):
        summary['decoder_parameters'] = sum(
            count_parameters(layer) for layer in model.decoder_layers
        )

    if hasattr(model, 'sampler'):
        summary['sampler_parameters'] = count_parameters(model.sampler)

    # Count head parameters
    head_params = 0
    if hasattr(model, 'regression_head') and model.regression_head is not None:
        head_params += sum(count_parameters(layer) for layer in model.regression_head)

    if hasattr(model, 'classification_head') and model.classification_head is not None:
        head_params += sum(count_parameters(layer) for layer in model.classification_head)

    summary['head_parameters'] = head_params

    return summary


def setup_device(prefer_cuda: bool = True) -> str:
    """
    Setup the computation device.

    Args:
        prefer_cuda: Whether to prefer CUDA if available

    Returns:
        Device string ('cuda' or 'cpu')
    """
    if prefer_cuda and torch.cuda.is_available():
        device = "cuda"
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = "cpu"
        print("Using CPU device")

    return device


def move_model_to_device(model: nn.Module, device: Union[str, torch.device]) -> nn.Module:
    """
    Move model to specified device.

    Args:
        model: PyTorch model
        device: Target device

    Returns:
        Model on target device
    """
    model = model.to(device)

    # Print device information
    device_name = next(model.parameters()).device
    print(f"Model moved to device: {device_name}")

    return model


def get_memory_usage(model: nn.Module, device: str = "cuda") -> Dict[str, float]:
    """
    Get memory usage statistics for the model.

    Args:
        model: PyTorch model
        device: Device to check memory for

    Returns:
        Dictionary with memory statistics in MB
    """
    if device == "cuda" and torch.cuda.is_available():
        # Get model parameter memory
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2

        # Get CUDA memory info
        allocated_memory = torch.cuda.memory_allocated() / 1024**2
        reserved_memory = torch.cuda.memory_reserved() / 1024**2

        return {
            'parameter_memory_mb': param_memory,
            'allocated_memory_mb': allocated_memory,
            'reserved_memory_mb': reserved_memory,
            'total_gpu_memory_mb': torch.cuda.get_device_properties(0).total_memory / 1024**2
        }
    else:
        # CPU memory estimation
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
        return {
            'parameter_memory_mb': param_memory,
            'device': 'cpu'
        }


def print_model_architecture(model: ModelInterface, verbose: bool = True):
    """
    Print a detailed model architecture overview.

    Args:
        model: TransformerVAE model instance
        verbose: Whether to include detailed layer information
    """
    summary = model_summary(model)

    print("=" * 70)
    print(f"Model Architecture: {summary['model_type']}")
    print("=" * 70)

    print(f"Total Parameters: {summary['total_parameters']:,}")
    print(f"Trainable Parameters: {summary['trainable_parameters']:,}")
    print(f"Latent Dimension: {summary['latent_dim']}")
    print()

    print("Component Parameter Breakdown:")
    print(f"  Encoder:        {summary['encoder_parameters']:,}")
    print(f"  Sampler:        {summary['sampler_parameters']:,}")
    print(f"  Decoder:        {summary['decoder_parameters']:,}")
    print(f"  Heads:          {summary['head_parameters']:,}")
    print()

    if verbose:
        print("Layer Details:")

        # Encoder layers
        if hasattr(model, 'encoder_layers'):
            print("  Encoder Layers:")
            for i, layer in enumerate(model.encoder_layers):
                layer_params = count_parameters(layer)
                print(f"    [{i}] {layer.__class__.__name__}: {layer_params:,} params")

        # Sampler
        if hasattr(model, 'sampler'):
            sampler_params = count_parameters(model.sampler)
            print(f"  Sampler: {model.sampler.__class__.__name__}: {sampler_params:,} params")

        # Decoder layers
        if hasattr(model, 'decoder_layers'):
            print("  Decoder Layers:")
            for i, layer in enumerate(model.decoder_layers):
                layer_params = count_parameters(layer)
                print(f"    [{i}] {layer.__class__.__name__}: {layer_params:,} params")

        # Heads
        if hasattr(model, 'regression_head') and model.regression_head is not None:
            print("  Regression Head:")
            for i, layer in enumerate(model.regression_head):
                layer_params = count_parameters(layer)
                print(f"    [{i}] {layer.__class__.__name__}: {layer_params:,} params")

        if hasattr(model, 'classification_head') and model.classification_head is not None:
            print("  Classification Head:")
            for i, layer in enumerate(model.classification_head):
                layer_params = count_parameters(layer)
                print(f"    [{i}] {layer.__class__.__name__}: {layer_params:,} params")

    print("=" * 70)


def validate_model_config(model: ModelInterface, input_shape: tuple) -> bool:
    """
    Validate that the model can process inputs of the specified shape.

    Args:
        model: Model to validate
        input_shape: Expected input shape (without batch dimension)

    Returns:
        True if model can process the input shape
    """
    try:
        model.eval()

        # Create dummy input
        batch_size = 2
        dummy_input = torch.randint(0, model.vocab_size, (batch_size,) + input_shape)

        # Try forward pass
        with torch.no_grad():
            output = model(dummy_input)

        # Validate output structure
        required_keys = ['reconstruction', 'mu', 'logvar']
        for key in required_keys:
            if key not in output:
                print(f"Missing required output key: {key}")
                return False

        # Validate output shapes
        if output['reconstruction'].shape != dummy_input.shape:
            print(f"Reconstruction shape mismatch: {output['reconstruction'].shape} != {dummy_input.shape}")
            return False

        if output['mu'].shape[0] != batch_size:
            print(f"Mu batch size mismatch: {output['mu'].shape[0]} != {batch_size}")
            return False

        if output['logvar'].shape != output['mu'].shape:
            print(f"Logvar shape mismatch: {output['logvar'].shape} != {output['mu'].shape}")
            return False

        print("Model validation passed!")
        return True

    except Exception as e:
        print(f"Model validation failed: {e}")
        return False


def initialize_weights(model: nn.Module, init_type: str = "normal", init_gain: float = 0.02):
    """
    Initialize model weights with specified strategy.

    Args:
        model: PyTorch model
        init_type: Initialization type ('normal', 'xavier', 'kaiming')
        init_gain: Initialization gain/std
    """
    def init_func(m):
        classname = m.__class__.__name__

        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            else:
                raise ValueError(f"Unknown initialization type: {init_type}")

            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    model.apply(init_func)
    print(f"Model weights initialized with {init_type} strategy")