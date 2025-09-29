"""
Integration tests for end-to-end configuration-to-training workflow.
These tests define expected workflow behavior and will initially fail.
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
        save_model_config,
        save_training_config,
    )
    from transformervae.models.model import TransformerVAE
    from transformervae.training.trainer import Trainer
    from transformervae.utils.reproducibility import set_random_seeds
except ImportError:
    # Mock functions that will initially fail
    def load_model_config(path):
        raise NotImplementedError("Configuration loading not implemented")

    def load_training_config(path):
        raise NotImplementedError("Training config loading not implemented")

    def save_model_config(config, path):
        raise NotImplementedError("Configuration saving not implemented")

    def save_training_config(config, path):
        raise NotImplementedError("Training config saving not implemented")

    class TransformerVAE:
        @classmethod
        def from_config(cls, config):
            raise NotImplementedError("TransformerVAE.from_config not implemented")

    class Trainer:
        @classmethod
        def from_config(cls, config):
            raise NotImplementedError("Trainer.from_config not implemented")

    def set_random_seeds(seed):
        raise NotImplementedError("Reproducibility utilities not implemented")


class TestConfigurationToTrainingWorkflow:
    """Test complete configuration-driven training workflow."""

    def test_full_workflow_from_yaml_configs(self):
        """Should support complete workflow from YAML configuration files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create model configuration
            model_config_path = os.path.join(temp_dir, "model_config.yaml")
            model_yaml = """
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

            # Create training configuration
            training_config_path = os.path.join(temp_dir, "training_config.yaml")
            training_yaml = """
learning_rate: 0.001
batch_size: 32
epochs: 10
beta: 1.0
optimizer_type: "adam"
weight_decay: 0.0001
validation_freq: 2
checkpoint_freq: 5

scheduler_config:
  type: "reduce_on_plateau"
  patience: 5
  factor: 0.5

random_seed: 42
"""

            # Write configuration files
            with open(model_config_path, 'w') as f:
                f.write(model_yaml)
            with open(training_config_path, 'w') as f:
                f.write(training_yaml)

            # Load configurations
            model_config = load_model_config(model_config_path)
            training_config = load_training_config(training_config_path)

            # Create model from configuration
            model = TransformerVAE.from_config(model_config)

            # Create trainer from configuration
            trainer = Trainer.from_config(training_config)

            # Setup model for training
            trainer.setup_model(model)

            # This workflow should complete without errors
            assert trainer.model is not None
            assert hasattr(trainer, 'optimizer')
            assert hasattr(trainer, 'scheduler')

    def test_configuration_validation_in_workflow(self):
        """Workflow should validate all configurations before proceeding."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create invalid model configuration (dimension mismatch)
            model_config_path = os.path.join(temp_dir, "invalid_model.yaml")
            invalid_yaml = """
encoder:
  - layer_type: "transformer_encoder"
    input_dim: 100
    output_dim: 256
    dropout: 0.1
    activation: "relu"

sampler:
  - layer_type: "latent_sampler"
    input_dim: 512  # Dimension mismatch!
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

            with open(model_config_path, 'w') as f:
                f.write(invalid_yaml)

            # Loading should fail validation
            with pytest.raises(ValueError, match="dimension mismatch"):
                load_model_config(model_config_path)

    def test_workflow_reproducibility(self):
        """Workflow should produce reproducible results with same configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create configuration files
            model_config_path = os.path.join(temp_dir, "model.yaml")
            training_config_path = os.path.join(temp_dir, "training.yaml")

            model_yaml = """
encoder:
  - layer_type: "transformer_encoder"
    input_dim: 64
    output_dim: 128
    dropout: 0.1
    activation: "relu"

sampler:
  - layer_type: "latent_sampler"
    input_dim: 128
    output_dim: 32
    dropout: 0.0
    activation: "linear"

decoder:
  - layer_type: "transformer_decoder"
    input_dim: 32
    output_dim: 64
    dropout: 0.1
    activation: "relu"
"""

            training_yaml = """
learning_rate: 0.001
batch_size: 16
epochs: 5
beta: 1.0
random_seed: 123
"""

            with open(model_config_path, 'w') as f:
                f.write(model_yaml)
            with open(training_config_path, 'w') as f:
                f.write(training_yaml)

            # First run
            set_random_seeds(123)
            model_config1 = load_model_config(model_config_path)
            training_config1 = load_training_config(training_config_path)
            model1 = TransformerVAE.from_config(model_config1)

            # Second run with same configs
            set_random_seeds(123)
            model_config2 = load_model_config(model_config_path)
            training_config2 = load_training_config(training_config_path)
            model2 = TransformerVAE.from_config(model_config2)

            # Models should have same initial parameters (deterministic initialization)
            import torch
            params1 = [p.clone() for p in model1.parameters()]
            params2 = [p.clone() for p in model2.parameters()]

            for p1, p2 in zip(params1, params2):
                assert torch.allclose(p1, p2), "Models should be initialized identically"

    def test_workflow_saves_and_loads_configs(self):
        """Workflow should support saving and loading configurations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Load base configurations
            model_config = load_model_config("config/model_configs/base_transformer.yaml")
            training_config = load_training_config("config/training_configs/moses_config.yaml")

            # Save configurations to new location
            saved_model_path = os.path.join(temp_dir, "saved_model.yaml")
            saved_training_path = os.path.join(temp_dir, "saved_training.yaml")

            save_model_config(model_config, saved_model_path)
            save_training_config(training_config, saved_training_path)

            # Load saved configurations
            loaded_model_config = load_model_config(saved_model_path)
            loaded_training_config = load_training_config(saved_training_path)

            # Should be identical to original
            assert loaded_model_config == model_config
            assert loaded_training_config == training_config

    def test_workflow_with_property_prediction(self):
        """Should support workflow with property prediction heads."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create model config with prediction heads
            model_config_path = os.path.join(temp_dir, "model_with_heads.yaml")
            model_yaml = """
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

            with open(model_config_path, 'w') as f:
                f.write(model_yaml)

            # Load and create model
            model_config = load_model_config(model_config_path)
            model = TransformerVAE.from_config(model_config)

            # Model should have prediction heads
            assert hasattr(model, 'latent_regression_head')
            assert hasattr(model, 'latent_classification_head')

            # Forward pass should include property predictions
            import torch
            x = torch.randint(0, 1000, (4, 15))
            output = model(x)

            assert "property_regression" in output
            assert "property_classification" in output
            assert output["property_regression"].shape == (4, 5)
            assert output["property_classification"].shape == (4, 10)


class TestWorkflowErrorHandling:
    """Test error handling in workflow integration."""

    def test_missing_config_file_fails(self):
        """Should fail gracefully when configuration files are missing."""
        with pytest.raises(FileNotFoundError):
            load_model_config("/nonexistent/config.yaml")

    def test_corrupted_config_file_fails(self):
        """Should handle corrupted configuration files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("corrupted: yaml: content: [invalid")
            temp_path = f.name

        try:
            with pytest.raises((ValueError, Exception)):  # YAML parsing error
                load_model_config(temp_path)
        finally:
            os.unlink(temp_path)

    def test_incompatible_configs_fail(self):
        """Should detect incompatible model and training configurations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create model config
            model_config_path = os.path.join(temp_dir, "model.yaml")
            model_yaml = """
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

            # Create incompatible training config
            training_config_path = os.path.join(temp_dir, "training.yaml")
            training_yaml = """
learning_rate: 0.001
batch_size: 32
epochs: 100
beta: 1.0

# Missing required fields or incompatible settings
dataset_config:
  dataset_type: "unsupported_dataset"
  vocab_size: -1  # Invalid
"""

            with open(model_config_path, 'w') as f:
                f.write(model_yaml)
            with open(training_config_path, 'w') as f:
                f.write(training_yaml)

            model_config = load_model_config(model_config_path)

            # Training config should fail validation
            with pytest.raises(ValueError):
                training_config = load_training_config(training_config_path)

    def test_model_trainer_compatibility_validation(self):
        """Should validate model-trainer compatibility."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create valid configurations
            model_config_path = os.path.join(temp_dir, "model.yaml")
            training_config_path = os.path.join(temp_dir, "training.yaml")

            # Minimal configs
            with open(model_config_path, 'w') as f:
                f.write("""
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
""")

            with open(training_config_path, 'w') as f:
                f.write("""
learning_rate: 0.001
batch_size: 32
epochs: 10
beta: 1.0
""")

            model_config = load_model_config(model_config_path)
            training_config = load_training_config(training_config_path)

            model = TransformerVAE.from_config(model_config)
            trainer = Trainer.from_config(training_config)

            # Setup should validate compatibility
            trainer.setup_model(model)

            # Should succeed if compatible
            assert trainer.model is model


class TestWorkflowPerformance:
    """Test workflow performance characteristics."""

    def test_config_loading_performance(self):
        """Configuration loading should be reasonably fast."""
        import time

        # Create test configuration
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
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
""")
            temp_path = f.name

        try:
            # Time configuration loading
            start_time = time.time()
            for _ in range(10):  # Load multiple times
                config = load_model_config(temp_path)
            end_time = time.time()

            avg_time = (end_time - start_time) / 10
            assert avg_time < 1.0, f"Config loading too slow: {avg_time:.3f}s average"

        finally:
            os.unlink(temp_path)

    def test_model_creation_performance(self):
        """Model creation from config should be reasonably fast."""
        import time

        # Load test configuration
        config = load_model_config("config/model_configs/base_transformer.yaml")

        # Time model creation
        start_time = time.time()
        for _ in range(5):  # Create multiple models
            model = TransformerVAE.from_config(config)
        end_time = time.time()

        avg_time = (end_time - start_time) / 5
        assert avg_time < 5.0, f"Model creation too slow: {avg_time:.3f}s average"

    def test_memory_usage_reasonable(self):
        """Workflow should not have excessive memory usage."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Load configurations and create models
        config = load_model_config("config/model_configs/base_transformer.yaml")
        models = []

        for _ in range(3):  # Create a few models
            model = TransformerVAE.from_config(config)
            models.append(model)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Should not use excessive memory (threshold depends on model size)
        assert memory_increase < 500, f"Excessive memory usage: {memory_increase:.1f}MB"


class TestWorkflowCompatibility:
    """Test workflow compatibility with different configurations."""

    def test_different_model_sizes(self):
        """Should support different model size configurations."""
        model_configs = [
            # Small model
            {
                "encoder_dim": 64,
                "latent_dim": 16,
                "decoder_dim": 64
            },
            # Medium model
            {
                "encoder_dim": 256,
                "latent_dim": 64,
                "decoder_dim": 256
            },
            # Large model
            {
                "encoder_dim": 512,
                "latent_dim": 128,
                "decoder_dim": 512
            }
        ]

        for config_params in model_configs:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write(f"""
encoder:
  - layer_type: "transformer_encoder"
    input_dim: 100
    output_dim: {config_params["encoder_dim"]}
    dropout: 0.1
    activation: "relu"

sampler:
  - layer_type: "latent_sampler"
    input_dim: {config_params["encoder_dim"]}
    output_dim: {config_params["latent_dim"]}
    dropout: 0.0
    activation: "linear"

decoder:
  - layer_type: "transformer_decoder"
    input_dim: {config_params["latent_dim"]}
    output_dim: 100
    dropout: 0.1
    activation: "relu"
""")
                temp_path = f.name

            try:
                config = load_model_config(temp_path)
                model = TransformerVAE.from_config(config)

                # Should create successfully for all sizes
                assert model is not None

                # Test forward pass
                import torch
                x = torch.randint(0, 1000, (2, 10))
                output = model(x)
                assert "reconstruction" in output

            finally:
                os.unlink(temp_path)

    def test_different_activation_functions(self):
        """Should support different activation functions."""
        activations = ["relu", "gelu", "tanh", "sigmoid"]

        for activation in activations:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write(f"""
encoder:
  - layer_type: "transformer_encoder"
    input_dim: 100
    output_dim: 128
    dropout: 0.1
    activation: "{activation}"

sampler:
  - layer_type: "latent_sampler"
    input_dim: 128
    output_dim: 32
    dropout: 0.0
    activation: "linear"

decoder:
  - layer_type: "transformer_decoder"
    input_dim: 32
    output_dim: 100
    dropout: 0.1
    activation: "{activation}"
""")
                temp_path = f.name

            try:
                config = load_model_config(temp_path)
                model = TransformerVAE.from_config(config)

                # Should work with all activation functions
                assert model is not None

            finally:
                os.unlink(temp_path)