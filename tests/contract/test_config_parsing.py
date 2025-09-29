"""
Contract tests for YAML configuration loading and parsing.
These tests define expected YAML parsing behavior and will initially fail.
"""

import pytest
import tempfile
import os
from pathlib import Path

# These imports will initially fail until implementation is complete
try:
    from transformervae.config.basic_config import (
        load_model_config,
        load_training_config,
        load_dataset_config,
        DetailedModelArchitecture,
        VAETrainingConfig,
        DatasetConfig,
    )
except ImportError:
    # Mock functions that will initially fail
    def load_model_config(path: str):
        raise NotImplementedError("YAML model config loading not implemented")

    def load_training_config(path: str):
        raise NotImplementedError("YAML training config loading not implemented")

    def load_dataset_config(path: str):
        raise NotImplementedError("YAML dataset config loading not implemented")

    # Mock classes from previous test
    from tests.contract.test_configuration_validation import (
        DetailedModelArchitecture,
        VAETrainingConfig,
        DatasetConfig,
    )


class TestYAMLConfigLoading:
    """Test YAML configuration file loading contracts."""

    def test_load_valid_model_config(self):
        """Should load valid model configuration from YAML file."""
        yaml_content = """
encoder:
  - layer_type: "transformer_encoder"
    input_dim: 100
    output_dim: 256
    dropout: 0.1
    activation: "relu"
    layer_params:
      num_heads: 8
      dim_feedforward: 512

sampler:
  - layer_type: "latent_sampler"
    input_dim: 256
    output_dim: 64
    dropout: 0.0
    activation: "linear"

decoder:
  - layer_type: "transformer_decoder"
    input_dim: 64
    output_dim: 100
    dropout: 0.1
    activation: "relu"
    layer_params:
      num_heads: 8
      dim_feedforward: 512
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            config = load_model_config(temp_path)
            assert isinstance(config, DetailedModelArchitecture)
            assert len(config.encoder) == 1
            assert len(config.sampler) == 1
            assert len(config.decoder) == 1
            assert config.encoder[0].layer_type == "transformer_encoder"
            assert config.encoder[0].input_dim == 100
            assert config.encoder[0].output_dim == 256
        finally:
            os.unlink(temp_path)

    def test_load_valid_training_config(self):
        """Should load valid training configuration from YAML file."""
        yaml_content = """
learning_rate: 0.001
batch_size: 32
epochs: 100
beta: 1.0
optimizer_type: "adam"
weight_decay: 0.0001
validation_freq: 5
checkpoint_freq: 10

scheduler_config:
  type: "reduce_on_plateau"
  patience: 10
  factor: 0.5
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            config = load_training_config(temp_path)
            assert isinstance(config, VAETrainingConfig)
            assert config.learning_rate == 0.001
            assert config.batch_size == 32
            assert config.epochs == 100
            assert config.beta == 1.0
            assert config.optimizer_type == "adam"
            assert config.scheduler_config["type"] == "reduce_on_plateau"
        finally:
            os.unlink(temp_path)

    def test_load_valid_dataset_config(self):
        """Should load valid dataset configuration from YAML file."""
        yaml_content = """
dataset_type: "moses"
data_path: "/path/to/moses/data"
max_sequence_length: 100
vocab_size: 1000
train_split: 0.8
val_split: 0.1
test_split: 0.1

preprocessing_config:
  augment_smiles: true
  canonical: false
  max_atoms: 50
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            config = load_dataset_config(temp_path)
            assert isinstance(config, DatasetConfig)
            assert config.dataset_type == "moses"
            assert config.data_path == "/path/to/moses/data"
            assert config.max_sequence_length == 100
            assert config.vocab_size == 1000
            assert config.preprocessing_config["augment_smiles"] is True
        finally:
            os.unlink(temp_path)

    def test_load_nonexistent_file_fails(self):
        """Loading nonexistent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_model_config("/nonexistent/path/config.yaml")

    def test_load_invalid_yaml_fails(self):
        """Loading invalid YAML should raise parsing error."""
        invalid_yaml = """
encoder:
  - layer_type: "transformer_encoder"
    input_dim: 100
    # Missing closing quote and invalid syntax
    activation: "relu
      num_heads: [invalid
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_yaml)
            temp_path = f.name

        try:
            with pytest.raises((ValueError, Exception)):  # YAML parsing error
                load_model_config(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_missing_required_fields_fails(self):
        """Loading YAML with missing required fields should fail validation."""
        incomplete_yaml = """
encoder:
  - layer_type: "transformer_encoder"
    input_dim: 100
    # Missing output_dim, dropout, activation

sampler:
  - layer_type: "latent_sampler"
    input_dim: 256
    output_dim: 64

# Missing decoder section
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(incomplete_yaml)
            temp_path = f.name

        try:
            with pytest.raises(ValueError):  # Validation error
                load_model_config(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_model_config_with_optional_heads(self):
        """Should load model config with optional prediction heads."""
        yaml_content = """
encoder:
  - layer_type: "transformer_encoder"
    input_dim: 100
    output_dim: 256
    dropout: 0.1
    activation: "relu"

sampler:
  - layer_type: "latent_sampler"
    input_dim: 256
    output_dim: 64
    dropout: 0.0
    activation: "linear"

decoder:
  - layer_type: "transformer_decoder"
    input_dim: 64
    output_dim: 100
    dropout: 0.1
    activation: "relu"

latent_regression_head:
  - layer_type: "regression_head"
    input_dim: 64
    output_dim: 5
    dropout: 0.1
    activation: "relu"

latent_classification_head:
  - layer_type: "classification_head"
    input_dim: 64
    output_dim: 10
    dropout: 0.1
    activation: "relu"
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            config = load_model_config(temp_path)
            assert isinstance(config, DetailedModelArchitecture)
            assert config.latent_regression_head is not None
            assert config.latent_classification_head is not None
            assert len(config.latent_regression_head) == 1
            assert len(config.latent_classification_head) == 1
            assert config.latent_regression_head[0].output_dim == 5
            assert config.latent_classification_head[0].output_dim == 10
        finally:
            os.unlink(temp_path)


class TestConfigFileDiscovery:
    """Test configuration file discovery and path resolution."""

    def test_load_from_relative_path(self):
        """Should load configuration from relative path."""
        # Create temporary directory structure
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "config" / "model_configs"
            config_dir.mkdir(parents=True)

            config_file = config_dir / "test_model.yaml"
            yaml_content = """
encoder:
  - layer_type: "transformer_encoder"
    input_dim: 100
    output_dim: 256
    dropout: 0.1
    activation: "relu"

sampler:
  - layer_type: "latent_sampler"
    input_dim: 256
    output_dim: 64
    dropout: 0.0
    activation: "linear"

decoder:
  - layer_type: "transformer_decoder"
    input_dim: 64
    output_dim: 100
    dropout: 0.1
    activation: "relu"
"""
            config_file.write_text(yaml_content)

            # Change to temp directory and load with relative path
            original_dir = os.getcwd()
            try:
                os.chdir(temp_dir)
                config = load_model_config("config/model_configs/test_model.yaml")
                assert isinstance(config, DetailedModelArchitecture)
            finally:
                os.chdir(original_dir)

    def test_load_from_absolute_path(self):
        """Should load configuration from absolute path."""
        yaml_content = """
learning_rate: 0.001
batch_size: 32
epochs: 100
beta: 1.0
scheduler_config:
  type: "constant"
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            # Use absolute path
            config = load_training_config(os.path.abspath(temp_path))
            assert isinstance(config, VAETrainingConfig)
            assert config.learning_rate == 0.001
        finally:
            os.unlink(temp_path)


class TestConfigValidationIntegration:
    """Test integration between YAML loading and validation."""

    def test_loaded_config_passes_validation(self):
        """Loaded configuration should automatically pass validation."""
        yaml_content = """ 
encoder:
  - layer_type: "transformer_encoder"
    input_dim: 100
    output_dim: 256
    dropout: 0.1
    activation: "relu"

sampler:
  - layer_type: "latent_sampler"
    input_dim: 256
    output_dim: 64
    dropout: 0.0
    activation: "linear"

decoder:
  - layer_type: "transformer_decoder"
    input_dim: 64
    output_dim: 100
    dropout: 0.1
    activation: "relu"
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            # Loading should include automatic validation
            config = load_model_config(temp_path)
            # If this doesn't raise an exception, validation passed
            assert isinstance(config, DetailedModelArchitecture)
        finally:
            os.unlink(temp_path)

    def test_loaded_invalid_config_fails_validation(self):
        """Loading invalid configuration should fail during validation."""
        invalid_yaml = """
encoder:
  - layer_type: "transformer_encoder"
    input_dim: -100  # Invalid negative dimension
    output_dim: 256
    dropout: 0.1
    activation: "relu"

sampler:
  - layer_type: "latent_sampler"
    input_dim: 256
    output_dim: 64
    dropout: 0.0
    activation: "linear"

decoder:
  - layer_type: "transformer_decoder"
    input_dim: 64
    output_dim: 100
    dropout: 0.1
    activation: "relu"
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_yaml)
            temp_path = f.name

        try:
            with pytest.raises(ValueError):  # Should fail validation
                load_model_config(temp_path)
        finally:
            os.unlink(temp_path)