"""
Model components for TransformerVAE.

This module provides the core neural network components including:
- LayerInterface: Abstract base class for all layers
- Individual layer implementations (encoder, decoder, sampler, etc.)
- LayerFactory: Factory for creating layers from configuration
- TransformerVAE: Main model implementation
"""

from transformervae.models.layer import (
    LayerInterface,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    LatentSampler,
    PoolingLayer,
    RegressionHead,
    ClassificationHead,
    LayerFactory,
    create_layer,
    register_layer,
    get_registered_layers,
)

try:
    from transformervae.models.model import (
        ModelInterface,
        TransformerVAE,
        PositionalEncoding,
        ModelFactory,
        create_model,
        register_model,
        get_registered_models,
    )
    _MODEL_AVAILABLE = True
except ImportError:
    _MODEL_AVAILABLE = False

__all__ = [
    # Layer interface and implementations
    "LayerInterface",
    "TransformerEncoderLayer",
    "TransformerDecoderLayer",
    "LatentSampler",
    "PoolingLayer",
    "RegressionHead",
    "ClassificationHead",
    # Factory system
    "LayerFactory",
    "create_layer",
    "register_layer",
    "get_registered_layers",
]

if _MODEL_AVAILABLE:
    __all__.extend([
        "ModelInterface",
        "TransformerVAE",
        "PositionalEncoding",
        "ModelFactory",
        "create_model",
        "register_model",
        "get_registered_models",
    ])