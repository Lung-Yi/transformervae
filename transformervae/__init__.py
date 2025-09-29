"""
TransformerVAE: Modular Configuration-Driven VAE for Molecular Generation.

A highly configurable implementation of TransformerVAE that enables researchers
to experiment with different model architectures, training parameters, and datasets
through YAML configuration files.
"""

__version__ = "0.1.0"
__author__ = "TransformerVAE Team"

# Import configuration components (available immediately)
from transformervae.config.basic_config import (
    LayerConfig,
    DetailedModelArchitecture,
    VAETrainingConfig,
    DatasetConfig,
    load_model_config,
    load_training_config,
    load_dataset_config,
    save_model_config,
    save_training_config,
    save_dataset_config,
)

# Models will be imported when available
try:
    from transformervae.models.model import TransformerVAE
    _MODELS_AVAILABLE = True
except ImportError:
    _MODELS_AVAILABLE = False
    TransformerVAE = None

__all__ = [
    # Configuration classes
    "LayerConfig",
    "DetailedModelArchitecture",
    "VAETrainingConfig",
    "DatasetConfig",
    # Configuration loading/saving
    "load_model_config",
    "load_training_config",
    "load_dataset_config",
    "save_model_config",
    "save_training_config",
    "save_dataset_config",
]

if _MODELS_AVAILABLE:
    __all__.append("TransformerVAE")