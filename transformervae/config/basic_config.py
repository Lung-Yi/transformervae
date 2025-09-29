"""
Basic configuration classes and utilities for TransformerVAE.

This module defines the core configuration dataclasses for model architecture,
training parameters, and dataset settings, along with validation and
YAML loading/saving functionality.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
import yaml
import os
from pathlib import Path


@dataclass
class LayerConfig:
    """Configuration for individual neural network layers."""

    layer_type: str
    input_dim: int
    output_dim: int
    dropout: float
    activation: str
    layer_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate layer configuration after initialization."""
        if self.layer_params is None:
            self.layer_params = {}
        validate_layer_config(self)


@dataclass
class DetailedModelArchitecture:
    """Complete model architecture configuration."""

    encoder: List[LayerConfig]
    sampler: List[LayerConfig]
    decoder: List[LayerConfig]
    latent_regression_head: Optional[List[LayerConfig]] = None
    latent_classification_head: Optional[List[LayerConfig]] = None

    def __post_init__(self):
        """Validate model architecture after initialization."""
        validate_model_architecture(self)


@dataclass
class VAETrainingConfig:
    """Training configuration parameters."""

    learning_rate: float
    batch_size: int
    epochs: int
    beta: float
    scheduler_config: Dict[str, Any]
    optimizer_type: str = "adam"
    weight_decay: float = 0.0
    gradient_clip_norm: Optional[float] = None
    validation_freq: int = 1
    checkpoint_freq: int = 10
    random_seed: Optional[int] = None
    dataset_config: Optional[Dict[str, Any]] = None
    beta_schedule: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate training configuration after initialization."""
        validate_training_config(self)


@dataclass
class DatasetConfig:
    """Dataset configuration and preprocessing parameters."""

    dataset_type: str
    data_path: str
    max_sequence_length: int
    vocab_size: int
    train_split: float
    val_split: float
    test_split: float
    preprocessing_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate dataset configuration after initialization."""
        if self.preprocessing_config is None:
            self.preprocessing_config = {}
        validate_dataset_config(self)


# Validation functions

def validate_layer_config(config: LayerConfig) -> None:
    """Validate LayerConfig parameters."""
    supported_layer_types = {
        "embedding",
        "transformer_encoder",
        "transformer_decoder",
        "latent_sampler",
        "pooling",
        "regression_head",
        "classification_head",
        "linear"
    }

    supported_activations = {
        "relu", "gelu", "tanh", "sigmoid", "linear", "leaky_relu"
    }

    if config.layer_type not in supported_layer_types:
        raise ValueError(f"Unknown layer type: {config.layer_type}. "
                        f"Supported types: {sorted(supported_layer_types)}")

    if config.input_dim <= 0:
        raise ValueError("input_dim must be positive")

    if config.output_dim <= 0:
        raise ValueError("output_dim must be positive")

    if not (0.0 <= config.dropout <= 1.0):
        raise ValueError("dropout must be between 0.0 and 1.0")

    if config.activation not in supported_activations:
        raise ValueError(f"Unknown activation: {config.activation}. "
                        f"Supported activations: {sorted(supported_activations)}")


def validate_model_architecture(config: DetailedModelArchitecture) -> None:
    """Validate DetailedModelArchitecture for dimensional compatibility."""
    if not config.encoder:
        raise ValueError("Encoder cannot be empty")

    if not config.decoder:
        raise ValueError("Decoder cannot be empty")

    if not config.sampler:
        raise ValueError("Sampler cannot be empty")

    # Check dimensional compatibility
    encoder_output_dim = config.encoder[-1].output_dim
    sampler_input_dim = config.sampler[0].input_dim

    if encoder_output_dim != sampler_input_dim:
        raise ValueError(f"Dimension mismatch: encoder output ({encoder_output_dim}) "
                        f"!= sampler input ({sampler_input_dim})")

    sampler_output_dim = config.sampler[-1].output_dim
    decoder_input_dim = config.decoder[0].input_dim

    if sampler_output_dim != decoder_input_dim:
        raise ValueError(f"Dimension mismatch: sampler output ({sampler_output_dim}) "
                        f"!= decoder input ({decoder_input_dim})")

    # Validate optional heads
    if config.latent_regression_head:
        head_input_dim = config.latent_regression_head[0].input_dim
        if head_input_dim != sampler_output_dim:
            raise ValueError(f"Dimension mismatch: regression head input ({head_input_dim}) "
                           f"!= sampler output ({sampler_output_dim})")

    if config.latent_classification_head:
        head_input_dim = config.latent_classification_head[0].input_dim
        if head_input_dim != sampler_output_dim:
            raise ValueError(f"Dimension mismatch: classification head input ({head_input_dim}) "
                           f"!= sampler output ({sampler_output_dim})")


def validate_training_config(config: VAETrainingConfig) -> None:
    """Validate VAETrainingConfig parameters."""
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if config.epochs <= 0:
        raise ValueError("epochs must be positive")

    if config.beta < 0:
        raise ValueError("beta must be non-negative")

    if config.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")

    if config.validation_freq <= 0:
        raise ValueError("validation_freq must be positive")

    if config.checkpoint_freq <= 0:
        raise ValueError("checkpoint_freq must be positive")

    if config.random_seed is not None and config.random_seed < 0:
        raise ValueError("random_seed must be non-negative")

    supported_optimizers = {"adam", "sgd", "adamw"}
    if config.optimizer_type not in supported_optimizers:
        raise ValueError(f"Unsupported optimizer: {config.optimizer_type}. "
                        f"Supported optimizers: {sorted(supported_optimizers)}")


def validate_dataset_config(config: DatasetConfig) -> None:
    """Validate DatasetConfig parameters."""
    supported_datasets = {"moses", "zinc15", "chembl"}

    if config.dataset_type not in supported_datasets:
        raise ValueError(f"Unsupported dataset type: {config.dataset_type}. "
                        f"Supported datasets: {sorted(supported_datasets)}")

    if config.max_sequence_length <= 0:
        raise ValueError("max_sequence_length must be positive")

    if config.vocab_size <= 0:
        raise ValueError("vocab_size must be positive")

    # Validate splits sum to 1.0
    total_split = config.train_split + config.val_split + config.test_split
    if abs(total_split - 1.0) > 1e-6:
        raise ValueError(f"Data splits must sum to 1.0, got {total_split}")

    for split_name, split_value in [
        ("train_split", config.train_split),
        ("val_split", config.val_split),
        ("test_split", config.test_split)
    ]:
        if not (0.0 <= split_value <= 1.0):
            raise ValueError(f"{split_name} must be between 0.0 and 1.0")


# YAML loading and saving functions

def load_model_config(path: str) -> DetailedModelArchitecture:
    """Load model configuration from YAML file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")

    try:
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {path}: {e}")

    # Convert layer configurations
    encoder = [LayerConfig(**layer_data) for layer_data in data['encoder']]
    sampler = [LayerConfig(**layer_data) for layer_data in data['sampler']]
    decoder = [LayerConfig(**layer_data) for layer_data in data['decoder']]

    # Handle optional heads
    latent_regression_head = None
    if 'latent_regression_head' in data and data['latent_regression_head']:
        latent_regression_head = [LayerConfig(**layer_data)
                                for layer_data in data['latent_regression_head']]

    latent_classification_head = None
    if 'latent_classification_head' in data and data['latent_classification_head']:
        latent_classification_head = [LayerConfig(**layer_data)
                                    for layer_data in data['latent_classification_head']]

    return DetailedModelArchitecture(
        encoder=encoder,
        sampler=sampler,
        decoder=decoder,
        latent_regression_head=latent_regression_head,
        latent_classification_head=latent_classification_head
    )


def load_training_config(path: str) -> VAETrainingConfig:
    """Load training configuration from YAML file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")

    try:
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {path}: {e}")

    return VAETrainingConfig(**data)


def load_dataset_config(path: str) -> DatasetConfig:
    """Load dataset configuration from YAML file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")

    try:
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {path}: {e}")

    return DatasetConfig(**data)


def save_model_config(config: DetailedModelArchitecture, path: str) -> None:
    """Save model configuration to YAML file."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Convert to dictionary
    data = {
        'encoder': [_layer_config_to_dict(layer) for layer in config.encoder],
        'sampler': [_layer_config_to_dict(layer) for layer in config.sampler],
        'decoder': [_layer_config_to_dict(layer) for layer in config.decoder],
    }

    if config.latent_regression_head:
        data['latent_regression_head'] = [
            _layer_config_to_dict(layer) for layer in config.latent_regression_head
        ]

    if config.latent_classification_head:
        data['latent_classification_head'] = [
            _layer_config_to_dict(layer) for layer in config.latent_classification_head
        ]

    try:
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, indent=2)
    except Exception as e:
        raise IOError(f"Failed to save configuration to {path}: {e}")


def save_training_config(config: VAETrainingConfig, path: str) -> None:
    """Save training configuration to YAML file."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Convert to dictionary, handling None values
    data = {
        'learning_rate': config.learning_rate,
        'batch_size': config.batch_size,
        'epochs': config.epochs,
        'beta': config.beta,
        'scheduler_config': config.scheduler_config,
        'optimizer_type': config.optimizer_type,
        'weight_decay': config.weight_decay,
        'validation_freq': config.validation_freq,
        'checkpoint_freq': config.checkpoint_freq,
    }

    if config.gradient_clip_norm is not None:
        data['gradient_clip_norm'] = config.gradient_clip_norm

    if config.random_seed is not None:
        data['random_seed'] = config.random_seed

    try:
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, indent=2)
    except Exception as e:
        raise IOError(f"Failed to save configuration to {path}: {e}")


def save_dataset_config(config: DatasetConfig, path: str) -> None:
    """Save dataset configuration to YAML file."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    data = {
        'dataset_type': config.dataset_type,
        'data_path': config.data_path,
        'max_sequence_length': config.max_sequence_length,
        'vocab_size': config.vocab_size,
        'train_split': config.train_split,
        'val_split': config.val_split,
        'test_split': config.test_split,
        'preprocessing_config': config.preprocessing_config,
    }

    try:
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, indent=2)
    except Exception as e:
        raise IOError(f"Failed to save configuration to {path}: {e}")


def _layer_config_to_dict(layer_config: LayerConfig) -> Dict[str, Any]:
    """Convert LayerConfig to dictionary for YAML serialization."""
    data = {
        'layer_type': layer_config.layer_type,
        'input_dim': layer_config.input_dim,
        'output_dim': layer_config.output_dim,
        'dropout': layer_config.dropout,
        'activation': layer_config.activation,
    }

    if layer_config.layer_params:
        data['layer_params'] = layer_config.layer_params

    return data