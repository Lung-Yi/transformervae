"""
Configuration module for TransformerVAE.

This module provides configuration classes and utilities for defining model
architectures, training parameters, and dataset configurations through
type-safe dataclasses and YAML files.
"""

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
    validate_layer_config,
    validate_model_architecture,
    validate_training_config,
    validate_dataset_config,
)

__all__ = [
    "LayerConfig",
    "DetailedModelArchitecture",
    "VAETrainingConfig",
    "DatasetConfig",
    "load_model_config",
    "load_training_config",
    "load_dataset_config",
    "save_model_config",
    "save_training_config",
    "save_dataset_config",
    "validate_layer_config",
    "validate_model_architecture",
    "validate_training_config",
    "validate_dataset_config",
]