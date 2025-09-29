"""
Integration tests for model creation from configuration.
These tests validate the model instantiation process and will initially fail.
"""

import pytest
import torch
import tempfile
import os

# These imports will initially fail until implementation is complete
try:
    from transformervae.config.basic_config import (
        DetailedModelArchitecture,
        LayerConfig,
        load_model_config,
    )
    from transformervae.models.model import TransformerVAE
    from transformervae.models.layer import create_layer
except ImportError:
    # Mock classes and functions
    from tests.contract.test_configuration_validation import (
        DetailedModelArchitecture,
        LayerConfig,
    )
    from tests.contract.test_config_parsing import load_model_config

    class TransformerVAE:
        @classmethod
        def from_config(cls, config):
            raise NotImplementedError("TransformerVAE.from_config not implemented")

    def create_layer(config):
        raise NotImplementedError("create_layer function not implemented")


class TestBasicModelCreation:
    """Test basic model creation from configuration."""

    def test_minimal_model_creation(self):
        """Should create model with minimal configuration."""
        config = DetailedModelArchitecture(
            encoder=[LayerConfig("transformer_encoder", 100, 256, 0.1, "relu")],
            sampler=[LayerConfig("latent_sampler", 256, 64, 0.0, "linear")],
            decoder=[LayerConfig("transformer_decoder", 64, 100, 0.1, "relu")]
        )

        model = TransformerVAE.from_config(config)

        # Basic checks
        assert model is not None
        assert hasattr(model, 'encoder')
        assert hasattr(model, 'sampler')
        assert hasattr(model, 'decoder')

    def test_model_creation_from_yaml_file(self):
        """Should create model from YAML configuration file."""
        yaml_config = """
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
            f.write(yaml_config)
            temp_path = f.name

        try:
            config = load_model_config(temp_path)
            model = TransformerVAE.from_config(config)

            assert model is not None
            # Test that configuration was properly applied
            # (specific tests depend on implementation details)

        finally:
            os.unlink(temp_path)

    def test_multi_layer_model_creation(self):
        """Should create model with multiple layers in each component."""
        config = DetailedModelArchitecture(
            encoder=[
                LayerConfig("transformer_encoder", 100, 256, 0.1, "relu"),
                LayerConfig("transformer_encoder", 256, 512, 0.1, "relu"),
                LayerConfig("pooling", 512, 512, 0.0, "linear", {"pooling_type": "mean"})
            ],
            sampler=[
                LayerConfig("latent_sampler", 512, 128, 0.0, "linear")
            ],
            decoder=[
                LayerConfig("transformer_decoder", 128, 512, 0.1, "relu"),
                LayerConfig("transformer_decoder", 512, 256, 0.1, "relu"),
                LayerConfig("transformer_decoder", 256, 100, 0.1, "relu")
            ]
        )

        model = TransformerVAE.from_config(config)

        # Should handle multi-layer configurations
        assert model is not None

        # Test forward pass with multi-layer model
        x = torch.randint(0, 1000, (4, 15))
        output = model(x)

        assert "reconstruction" in output
        assert "mu" in output
        assert "logvar" in output


class TestModelCreationWithHeads:
    """Test model creation with optional prediction heads."""

    def test_model_with_regression_head(self):
        """Should create model with regression head for property prediction."""
        config = DetailedModelArchitecture(
            encoder=[LayerConfig("transformer_encoder", 100, 256, 0.1, "relu")],
            sampler=[LayerConfig("latent_sampler", 256, 64, 0.0, "linear")],
            decoder=[LayerConfig("transformer_decoder", 64, 100, 0.1, "relu")],
            latent_regression_head=[
                LayerConfig("regression_head", 64, 5, 0.1, "relu")
            ]
        )

        model = TransformerVAE.from_config(config)

        assert hasattr(model, 'latent_regression_head')

        # Test forward pass includes regression output
        x = torch.randint(0, 1000, (3, 10))
        output = model(x)

        assert "property_regression" in output
        assert output["property_regression"].shape == (3, 5)

    def test_model_with_classification_head(self):
        """Should create model with classification head for molecular classes."""
        config = DetailedModelArchitecture(
            encoder=[LayerConfig("transformer_encoder", 100, 256, 0.1, "relu")],
            sampler=[LayerConfig("latent_sampler", 256, 64, 0.0, "linear")],
            decoder=[LayerConfig("transformer_decoder", 64, 100, 0.1, "relu")],
            latent_classification_head=[
                LayerConfig("classification_head", 64, 10, 0.1, "relu")
            ]
        )

        model = TransformerVAE.from_config(config)

        assert hasattr(model, 'latent_classification_head')

        # Test forward pass includes classification output
        x = torch.randint(0, 1000, (3, 10))
        output = model(x)

        assert "property_classification" in output
        assert output["property_classification"].shape == (3, 10)

    def test_model_with_both_heads(self):
        """Should create model with both regression and classification heads."""
        config = DetailedModelArchitecture(
            encoder=[LayerConfig("transformer_encoder", 100, 256, 0.1, "relu")],
            sampler=[LayerConfig("latent_sampler", 256, 64, 0.0, "linear")],
            decoder=[LayerConfig("transformer_decoder", 64, 100, 0.1, "relu")],
            latent_regression_head=[
                LayerConfig("regression_head", 64, 5, 0.1, "relu")
            ],
            latent_classification_head=[
                LayerConfig("classification_head", 64, 10, 0.1, "relu")
            ]
        )

        model = TransformerVAE.from_config(config)

        assert hasattr(model, 'latent_regression_head')
        assert hasattr(model, 'latent_classification_head')

        # Test forward pass includes both outputs
        x = torch.randint(0, 1000, (2, 12))
        output = model(x)

        assert "property_regression" in output
        assert "property_classification" in output
        assert output["property_regression"].shape == (2, 5)
        assert output["property_classification"].shape == (2, 10)

    def test_model_without_heads(self):
        """Should create model without optional heads when not specified."""
        config = DetailedModelArchitecture(
            encoder=[LayerConfig("transformer_encoder", 100, 256, 0.1, "relu")],
            sampler=[LayerConfig("latent_sampler", 256, 64, 0.0, "linear")],
            decoder=[LayerConfig("transformer_decoder", 64, 100, 0.1, "relu")]
            # No optional heads specified
        )

        model = TransformerVAE.from_config(config)

        # Should not have prediction heads
        assert not hasattr(model, 'latent_regression_head') or model.latent_regression_head is None
        assert not hasattr(model, 'latent_classification_head') or model.latent_classification_head is None

        # Forward pass should not include property predictions
        x = torch.randint(0, 1000, (2, 10))
        output = model(x)

        assert "property_regression" not in output
        assert "property_classification" not in output


class TestModelCreationValidation:
    """Test validation during model creation."""

    def test_dimension_mismatch_validation(self):
        """Should validate layer dimension compatibility."""
        # Encoder output (256) doesn't match sampler input (512)
        invalid_config = DetailedModelArchitecture(
            encoder=[LayerConfig("transformer_encoder", 100, 256, 0.1, "relu")],
            sampler=[LayerConfig("latent_sampler", 512, 64, 0.0, "linear")],  # Mismatch!
            decoder=[LayerConfig("transformer_decoder", 64, 100, 0.1, "relu")]
        )

        with pytest.raises(ValueError, match="dimension mismatch"):
            TransformerVAE.from_config(invalid_config)

    def test_empty_component_validation(self):
        """Should validate that required components are not empty."""
        # Empty encoder
        with pytest.raises(ValueError, match="encoder.*empty"):
            DetailedModelArchitecture(
                encoder=[],  # Empty!
                sampler=[LayerConfig("latent_sampler", 256, 64, 0.0, "linear")],
                decoder=[LayerConfig("transformer_decoder", 64, 100, 0.1, "relu")]
            )

        # Empty decoder
        with pytest.raises(ValueError, match="decoder.*empty"):
            DetailedModelArchitecture(
                encoder=[LayerConfig("transformer_encoder", 100, 256, 0.1, "relu")],
                sampler=[LayerConfig("latent_sampler", 256, 64, 0.0, "linear")],
                decoder=[]  # Empty!
            )

    def test_invalid_layer_type_validation(self):
        """Should validate layer types during model creation."""
        invalid_config = DetailedModelArchitecture(
            encoder=[LayerConfig("nonexistent_layer", 100, 256, 0.1, "relu")],
            sampler=[LayerConfig("latent_sampler", 256, 64, 0.0, "linear")],
            decoder=[LayerConfig("transformer_decoder", 64, 100, 0.1, "relu")]
        )

        with pytest.raises(ValueError, match="unknown layer type"):
            TransformerVAE.from_config(invalid_config)

    def test_prediction_head_dimension_validation(self):
        """Should validate prediction head dimensions match latent dimension."""
        # Regression head input (128) doesn't match sampler output (64)
        invalid_config = DetailedModelArchitecture(
            encoder=[LayerConfig("transformer_encoder", 100, 256, 0.1, "relu")],
            sampler=[LayerConfig("latent_sampler", 256, 64, 0.0, "linear")],
            decoder=[LayerConfig("transformer_decoder", 64, 100, 0.1, "relu")],
            latent_regression_head=[
                LayerConfig("regression_head", 128, 5, 0.1, "relu")  # Wrong input dim!
            ]
        )

        with pytest.raises(ValueError, match="dimension mismatch"):
            TransformerVAE.from_config(invalid_config)


class TestModelCreationParameterization:
    """Test model creation with different parameters."""

    def test_different_activation_functions(self):
        """Should support different activation functions."""
        activations = ["relu", "gelu", "tanh", "sigmoid"]

        for activation in activations:
            config = DetailedModelArchitecture(
                encoder=[LayerConfig("transformer_encoder", 100, 256, 0.1, activation)],
                sampler=[LayerConfig("latent_sampler", 256, 64, 0.0, "linear")],
                decoder=[LayerConfig("transformer_decoder", 64, 100, 0.1, activation)]
            )

            model = TransformerVAE.from_config(config)
            assert model is not None

            # Test forward pass works with different activations
            x = torch.randint(0, 1000, (2, 8))
            output = model(x)
            assert "reconstruction" in output

    def test_different_dropout_rates(self):
        """Should support different dropout rates."""
        dropout_rates = [0.0, 0.1, 0.3, 0.5]

        for dropout in dropout_rates:
            config = DetailedModelArchitecture(
                encoder=[LayerConfig("transformer_encoder", 100, 256, dropout, "relu")],
                sampler=[LayerConfig("latent_sampler", 256, 64, 0.0, "linear")],
                decoder=[LayerConfig("transformer_decoder", 64, 100, dropout, "relu")]
            )

            model = TransformerVAE.from_config(config)
            assert model is not None

    def test_different_model_sizes(self):
        """Should support different model sizes."""
        size_configs = [
            # Small model
            {"encoder_dim": 128, "latent_dim": 32},
            # Medium model
            {"encoder_dim": 256, "latent_dim": 64},
            # Large model
            {"encoder_dim": 512, "latent_dim": 128},
        ]

        for size_config in size_configs:
            config = DetailedModelArchitecture(
                encoder=[LayerConfig("transformer_encoder", 100, size_config["encoder_dim"], 0.1, "relu")],
                sampler=[LayerConfig("latent_sampler", size_config["encoder_dim"], size_config["latent_dim"], 0.0, "linear")],
                decoder=[LayerConfig("transformer_decoder", size_config["latent_dim"], 100, 0.1, "relu")]
            )

            model = TransformerVAE.from_config(config)
            assert model is not None

            # Test forward pass
            x = torch.randint(0, 1000, (3, 10))
            output = model(x)
            assert output["mu"].shape == (3, size_config["latent_dim"])

    def test_layer_specific_parameters(self):
        """Should pass layer-specific parameters correctly."""
        config = DetailedModelArchitecture(
            encoder=[
                LayerConfig("transformer_encoder", 100, 256, 0.1, "relu", {
                    "num_heads": 16,
                    "dim_feedforward": 1024,
                    "num_layers": 6
                })
            ],
            sampler=[LayerConfig("latent_sampler", 256, 64, 0.0, "linear")],
            decoder=[
                LayerConfig("transformer_decoder", 64, 100, 0.1, "relu", {
                    "num_heads": 8,
                    "dim_feedforward": 512,
                    "num_layers": 4
                })
            ]
        )

        model = TransformerVAE.from_config(config)
        assert model is not None

        # Verify that layer parameters were applied
        # (Implementation-specific checks would go here)


class TestModelCreationDevicePlacement:
    """Test model creation with different device placements."""

    def test_model_creation_cpu(self):
        """Should create model on CPU by default."""
        config = DetailedModelArchitecture(
            encoder=[LayerConfig("transformer_encoder", 100, 256, 0.1, "relu")],
            sampler=[LayerConfig("latent_sampler", 256, 64, 0.0, "linear")],
            decoder=[LayerConfig("transformer_decoder", 64, 100, 0.1, "relu")]
        )

        model = TransformerVAE.from_config(config)

        # Should be on CPU by default
        device = next(model.parameters()).device
        assert device.type == "cpu"

    def test_model_device_movement(self):
        """Should support moving model to different devices."""
        config = DetailedModelArchitecture(
            encoder=[LayerConfig("transformer_encoder", 100, 256, 0.1, "relu")],
            sampler=[LayerConfig("latent_sampler", 256, 64, 0.0, "linear")],
            decoder=[LayerConfig("transformer_decoder", 64, 100, 0.1, "relu")]
        )

        model = TransformerVAE.from_config(config)

        # Move to CPU explicitly
        model = model.to("cpu")
        assert next(model.parameters()).device.type == "cpu"

        # Test CUDA if available
        if torch.cuda.is_available():
            model = model.to("cuda")
            assert next(model.parameters()).device.type == "cuda"

    def test_model_creation_with_device_specification(self):
        """Should support device specification during creation."""
        config = DetailedModelArchitecture(
            encoder=[LayerConfig("transformer_encoder", 100, 256, 0.1, "relu")],
            sampler=[LayerConfig("latent_sampler", 256, 64, 0.0, "linear")],
            decoder=[LayerConfig("transformer_decoder", 64, 100, 0.1, "relu")]
        )

        # Create on CPU
        model_cpu = TransformerVAE.from_config(config, device="cpu")
        assert next(model_cpu.parameters()).device.type == "cpu"

        # Create on CUDA if available
        if torch.cuda.is_available():
            model_cuda = TransformerVAE.from_config(config, device="cuda")
            assert next(model_cuda.parameters()).device.type == "cuda"


class TestModelCreationMemoryEfficiency:
    """Test memory efficiency of model creation."""

    def test_model_creation_memory_usage(self):
        """Model creation should not use excessive memory."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create several models
        configs = []
        models = []

        for i in range(5):
            config = DetailedModelArchitecture(
                encoder=[LayerConfig("transformer_encoder", 100, 256, 0.1, "relu")],
                sampler=[LayerConfig("latent_sampler", 256, 64, 0.0, "linear")],
                decoder=[LayerConfig("transformer_decoder", 64, 100, 0.1, "relu")]
            )
            configs.append(config)
            models.append(TransformerVAE.from_config(config))

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Should not use excessive memory for model creation
        assert memory_increase < 200, f"Excessive memory usage: {memory_increase:.1f}MB"

    def test_model_creation_timing(self):
        """Model creation should be reasonably fast."""
        import time

        config = DetailedModelArchitecture(
            encoder=[LayerConfig("transformer_encoder", 100, 256, 0.1, "relu")],
            sampler=[LayerConfig("latent_sampler", 256, 64, 0.0, "linear")],
            decoder=[LayerConfig("transformer_decoder", 64, 100, 0.1, "relu")]
        )

        # Time model creation
        start_time = time.time()
        for _ in range(3):
            model = TransformerVAE.from_config(config)
        end_time = time.time()

        avg_time = (end_time - start_time) / 3
        assert avg_time < 2.0, f"Model creation too slow: {avg_time:.3f}s average"