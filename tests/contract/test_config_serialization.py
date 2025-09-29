"""
Contract tests for configuration serialization to YAML.
These tests define expected serialization behavior and will initially fail.
"""

import pytest
import tempfile
import os
from typing import Dict, Any

# These imports will initially fail until implementation is complete
try:
    from transformervae.config.basic_config import (
        save_model_config,
        save_training_config,
        save_dataset_config,
        load_model_config,
        load_training_config,
        load_dataset_config,
        LayerConfig,
        DetailedModelArchitecture,
        VAETrainingConfig,
        DatasetConfig,
    )
except ImportError:
    # Mock functions that will initially fail
    def save_model_config(config, path: str):
        raise NotImplementedError("YAML model config saving not implemented")

    def save_training_config(config, path: str):
        raise NotImplementedError("YAML training config saving not implemented")

    def save_dataset_config(config, path: str):
        raise NotImplementedError("YAML dataset config saving not implemented")

    # Import mock classes and load functions from previous tests
    from tests.contract.test_configuration_validation import (
        LayerConfig,
        DetailedModelArchitecture,
        VAETrainingConfig,
        DatasetConfig,
    )
    from tests.contract.test_config_parsing import (
        load_model_config,
        load_training_config,
        load_dataset_config,
    )


class TestModelConfigSerialization:
    """Test model configuration serialization contracts."""

    def test_save_and_load_model_config_roundtrip(self):
        """Should save and load model config without data loss."""
        # Create test configuration
        encoder = [LayerConfig(
            layer_type="transformer_encoder",
            input_dim=100,
            output_dim=256,
            dropout=0.1,
            activation="relu",
            layer_params={"num_heads": 8, "dim_feedforward": 512}
        )]

        sampler = [LayerConfig(
            layer_type="latent_sampler",
            input_dim=256,
            output_dim=64,
            dropout=0.0,
            activation="linear"
        )]

        decoder = [LayerConfig(
            layer_type="transformer_decoder",
            input_dim=64,
            output_dim=100,
            dropout=0.1,
            activation="relu",
            layer_params={"num_heads": 8, "dim_feedforward": 512}
        )]

        original_config = DetailedModelArchitecture(
            encoder=encoder,
            sampler=sampler,
            decoder=decoder
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name

        try:
            # Save configuration
            save_model_config(original_config, temp_path)

            # Load configuration back
            loaded_config = load_model_config(temp_path)

            # Verify data integrity
            assert isinstance(loaded_config, DetailedModelArchitecture)
            assert len(loaded_config.encoder) == len(original_config.encoder)
            assert len(loaded_config.sampler) == len(original_config.sampler)
            assert len(loaded_config.decoder) == len(original_config.decoder)

            # Verify encoder details
            assert loaded_config.encoder[0].layer_type == "transformer_encoder"
            assert loaded_config.encoder[0].input_dim == 100
            assert loaded_config.encoder[0].output_dim == 256
            assert loaded_config.encoder[0].dropout == 0.1
            assert loaded_config.encoder[0].activation == "relu"
            assert loaded_config.encoder[0].layer_params["num_heads"] == 8

            # Verify sampler details
            assert loaded_config.sampler[0].layer_type == "latent_sampler"
            assert loaded_config.sampler[0].input_dim == 256
            assert loaded_config.sampler[0].output_dim == 64

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_save_model_config_with_optional_heads(self):
        """Should save and load model config with optional prediction heads."""
        # Create configuration with optional heads
        encoder = [LayerConfig("transformer_encoder", 100, 256, 0.1, "relu")]
        sampler = [LayerConfig("latent_sampler", 256, 64, 0.0, "linear")]
        decoder = [LayerConfig("transformer_decoder", 64, 100, 0.1, "relu")]
        regression_head = [LayerConfig("regression_head", 64, 5, 0.1, "relu")]
        classification_head = [LayerConfig("classification_head", 64, 10, 0.1, "relu")]

        original_config = DetailedModelArchitecture(
            encoder=encoder,
            sampler=sampler,
            decoder=decoder,
            latent_regression_head=regression_head,
            latent_classification_head=classification_head
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name

        try:
            save_model_config(original_config, temp_path)
            loaded_config = load_model_config(temp_path)

            assert loaded_config.latent_regression_head is not None
            assert loaded_config.latent_classification_head is not None
            assert len(loaded_config.latent_regression_head) == 1
            assert len(loaded_config.latent_classification_head) == 1
            assert loaded_config.latent_regression_head[0].output_dim == 5
            assert loaded_config.latent_classification_head[0].output_dim == 10

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_save_to_nonexistent_directory(self):
        """Should create directory structure when saving to nonexistent path."""
        config = DetailedModelArchitecture(
            encoder=[LayerConfig("transformer_encoder", 100, 256, 0.1, "relu")],
            sampler=[LayerConfig("latent_sampler", 256, 64, 0.0, "linear")],
            decoder=[LayerConfig("transformer_decoder", 64, 100, 0.1, "relu")]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create path with nested directories that don't exist
            nested_path = os.path.join(temp_dir, "config", "models", "test.yaml")

            # Should create directories automatically
            save_model_config(config, nested_path)

            # Verify file was created and can be loaded
            assert os.path.exists(nested_path)
            loaded_config = load_model_config(nested_path)
            assert isinstance(loaded_config, DetailedModelArchitecture)


class TestTrainingConfigSerialization:
    """Test training configuration serialization contracts."""

    def test_save_and_load_training_config_roundtrip(self):
        """Should save and load training config without data loss."""
        original_config = VAETrainingConfig(
            learning_rate=0.001,
            batch_size=32,
            epochs=100,
            beta=1.0,
            scheduler_config={
                "type": "reduce_on_plateau",
                "patience": 10,
                "factor": 0.5,
                "min_lr": 0.00001
            },
            optimizer_type="adam",
            weight_decay=0.0001,
            gradient_clip_norm=1.0,
            validation_freq=5,
            checkpoint_freq=10
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name

        try:
            save_training_config(original_config, temp_path)
            loaded_config = load_training_config(temp_path)

            assert isinstance(loaded_config, VAETrainingConfig)
            assert loaded_config.learning_rate == 0.001
            assert loaded_config.batch_size == 32
            assert loaded_config.epochs == 100
            assert loaded_config.beta == 1.0
            assert loaded_config.optimizer_type == "adam"
            assert loaded_config.weight_decay == 0.0001
            assert loaded_config.gradient_clip_norm == 1.0
            assert loaded_config.validation_freq == 5
            assert loaded_config.checkpoint_freq == 10

            # Verify scheduler config
            assert loaded_config.scheduler_config["type"] == "reduce_on_plateau"
            assert loaded_config.scheduler_config["patience"] == 10
            assert loaded_config.scheduler_config["factor"] == 0.5
            assert loaded_config.scheduler_config["min_lr"] == 0.00001

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_save_training_config_with_none_values(self):
        """Should handle None values in training configuration."""
        config = VAETrainingConfig(
            learning_rate=0.001,
            batch_size=32,
            epochs=100,
            beta=1.0,
            scheduler_config={},
            gradient_clip_norm=None  # None value
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name

        try:
            save_training_config(config, temp_path)
            loaded_config = load_training_config(temp_path)

            assert loaded_config.gradient_clip_norm is None

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestDatasetConfigSerialization:
    """Test dataset configuration serialization contracts."""

    def test_save_and_load_dataset_config_roundtrip(self):
        """Should save and load dataset config without data loss."""
        original_config = DatasetConfig(
            dataset_type="moses",
            data_path="/path/to/moses/data",
            max_sequence_length=100,
            vocab_size=1000,
            train_split=0.8,
            val_split=0.1,
            test_split=0.1,
            preprocessing_config={
                "augment_smiles": True,
                "canonical": False,
                "max_atoms": 50,
                "add_explicit_hydrogens": False
            }
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name

        try:
            save_dataset_config(original_config, temp_path)
            loaded_config = load_dataset_config(temp_path)

            assert isinstance(loaded_config, DatasetConfig)
            assert loaded_config.dataset_type == "moses"
            assert loaded_config.data_path == "/path/to/moses/data"
            assert loaded_config.max_sequence_length == 100
            assert loaded_config.vocab_size == 1000
            assert loaded_config.train_split == 0.8
            assert loaded_config.val_split == 0.1
            assert loaded_config.test_split == 0.1

            # Verify preprocessing config
            assert loaded_config.preprocessing_config["augment_smiles"] is True
            assert loaded_config.preprocessing_config["canonical"] is False
            assert loaded_config.preprocessing_config["max_atoms"] == 50
            assert loaded_config.preprocessing_config["add_explicit_hydrogens"] is False

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestSerializationFormats:
    """Test serialization format and structure contracts."""

    def test_saved_yaml_is_human_readable(self):
        """Saved YAML should be human-readable and properly formatted."""
        config = DetailedModelArchitecture(
            encoder=[LayerConfig("transformer_encoder", 100, 256, 0.1, "relu")],
            sampler=[LayerConfig("latent_sampler", 256, 64, 0.0, "linear")],
            decoder=[LayerConfig("transformer_decoder", 64, 100, 0.1, "relu")]
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name

        try:
            save_model_config(config, temp_path)

            # Read file content and verify structure
            with open(temp_path, 'r') as f:
                content = f.read()

            # Verify key sections are present
            assert "encoder:" in content
            assert "sampler:" in content
            assert "decoder:" in content
            assert "layer_type:" in content
            assert "input_dim:" in content
            assert "output_dim:" in content

            # Verify proper YAML structure (indentation, etc.)
            lines = content.strip().split('\n')
            assert any(line.startswith('encoder:') for line in lines)
            assert any(line.startswith('  - layer_type:') for line in lines)

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_save_preserves_comments_and_structure(self):
        """Should preserve YAML structure and allow for comments."""
        # This test ensures that the serialization format is maintainable
        config = VAETrainingConfig(
            learning_rate=0.001,
            batch_size=32,
            epochs=100,
            beta=1.0,
            scheduler_config={"type": "constant"}
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name

        try:
            save_training_config(config, temp_path)

            with open(temp_path, 'r') as f:
                content = f.read()

            # Verify the YAML is structured and readable
            assert "learning_rate:" in content
            assert "batch_size:" in content
            assert "epochs:" in content
            assert "beta:" in content
            assert "scheduler_config:" in content

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestSerializationErrorHandling:
    """Test error handling in serialization contracts."""

    def test_save_to_readonly_location_fails(self):
        """Should raise appropriate error when saving to readonly location."""
        config = DetailedModelArchitecture(
            encoder=[LayerConfig("transformer_encoder", 100, 256, 0.1, "relu")],
            sampler=[LayerConfig("latent_sampler", 256, 64, 0.0, "linear")],
            decoder=[LayerConfig("transformer_decoder", 64, 100, 0.1, "relu")]
        )

        # Try to save to root directory (should be readonly)
        readonly_path = "/readonly_test_config.yaml"

        with pytest.raises((PermissionError, OSError)):
            save_model_config(config, readonly_path)

    def test_save_invalid_config_fails(self):
        """Should validate configuration before saving."""
        # Create invalid configuration
        invalid_config = DetailedModelArchitecture(
            encoder=[],  # Empty encoder (invalid)
            sampler=[LayerConfig("latent_sampler", 256, 64, 0.0, "linear")],
            decoder=[LayerConfig("transformer_decoder", 64, 100, 0.1, "relu")]
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name

        try:
            with pytest.raises(ValueError):  # Should fail validation
                save_model_config(invalid_config, temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)