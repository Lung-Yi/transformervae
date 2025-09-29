"""
Unit tests for TransformerEncoderLayer.
"""

import pytest
import torch
from transformervae.config.basic_config import LayerConfig
from transformervae.models.layer import TransformerEncoderLayer, LayerInterface


class TestTransformerEncoderLayer:
    """Test TransformerEncoderLayer implementation."""

    def test_encoder_creation(self):
        """Should create encoder with valid configuration."""
        config = LayerConfig(
            layer_type="transformer_encoder",
            input_dim=100,
            output_dim=256,
            dropout=0.1,
            activation="relu",
            layer_params={"num_heads": 8, "dim_feedforward": 512}
        )

        encoder = TransformerEncoderLayer(config)

        assert isinstance(encoder, LayerInterface)
        assert encoder.input_dim == 100
        assert encoder.output_dim == 256
        assert encoder.num_heads == 8
        assert encoder.dim_feedforward == 512

    def test_encoder_forward_pass(self):
        """Should perform forward pass with correct shapes."""
        config = LayerConfig(
            layer_type="transformer_encoder",
            input_dim=64,
            output_dim=128,
            dropout=0.1,
            activation="relu"
        )

        encoder = TransformerEncoderLayer(config)
        encoder.eval()  # Set to eval mode for deterministic behavior

        batch_size, seq_len = 4, 10
        x = torch.randn(batch_size, seq_len, 64)

        with torch.no_grad():
            output = encoder(x)

        assert output.shape == (batch_size, seq_len, 128)
        assert output.dtype == x.dtype

    def test_encoder_with_same_input_output_dim(self):
        """Should handle case where input_dim == output_dim."""
        config = LayerConfig(
            layer_type="transformer_encoder",
            input_dim=256,
            output_dim=256,
            dropout=0.1,
            activation="gelu"
        )

        encoder = TransformerEncoderLayer(config)

        # Should not have input projection
        assert encoder.input_projection is None

        x = torch.randn(2, 8, 256)
        output = encoder(x)
        assert output.shape == (2, 8, 256)

    def test_encoder_with_attention_mask(self):
        """Should support attention masking."""
        config = LayerConfig(
            layer_type="transformer_encoder",
            input_dim=64,
            output_dim=64,
            dropout=0.0,
            activation="relu"
        )

        encoder = TransformerEncoderLayer(config)
        encoder.eval()

        batch_size, seq_len = 3, 12
        x = torch.randn(batch_size, seq_len, 64)

        # Create mask (True for padding positions)
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        mask[:, 8:] = True  # Mask last 4 positions

        with torch.no_grad():
            output = encoder(x, mask=mask)

        assert output.shape == (batch_size, seq_len, 64)

    def test_encoder_different_activations(self):
        """Should support different activation functions."""
        activations = ["relu", "gelu", "tanh"]

        for activation in activations:
            config = LayerConfig(
                layer_type="transformer_encoder",
                input_dim=32,
                output_dim=64,
                dropout=0.1,
                activation=activation
            )

            encoder = TransformerEncoderLayer(config)
            x = torch.randn(2, 5, 32)
            output = encoder(x)

            assert output.shape == (2, 5, 64)

    def test_encoder_multiple_layers(self):
        """Should support multiple transformer layers."""
        config = LayerConfig(
            layer_type="transformer_encoder",
            input_dim=64,
            output_dim=64,
            dropout=0.1,
            activation="relu",
            layer_params={"num_heads": 4, "num_layers": 3}
        )

        encoder = TransformerEncoderLayer(config)

        assert encoder.num_layers == 3

        x = torch.randn(2, 6, 64)
        output = encoder(x)
        assert output.shape == (2, 6, 64)

    def test_encoder_different_head_numbers(self):
        """Should support different numbers of attention heads."""
        for num_heads in [1, 2, 4, 8]:
            config = LayerConfig(
                layer_type="transformer_encoder",
                input_dim=64,  # Must be divisible by num_heads
                output_dim=64,
                dropout=0.1,
                activation="relu",
                layer_params={"num_heads": num_heads}
            )

            encoder = TransformerEncoderLayer(config)
            assert encoder.num_heads == num_heads

            x = torch.randn(2, 4, 64)
            output = encoder(x)
            assert output.shape == (2, 4, 64)

    def test_encoder_invalid_head_dimension(self):
        """Should handle case where embedding dimension is not divisible by num_heads."""
        config = LayerConfig(
            layer_type="transformer_encoder",
            input_dim=100,
            output_dim=100,
            dropout=0.1,
            activation="relu",
            layer_params={"num_heads": 7}  # 100 not divisible by 7
        )

        # Should raise error during layer creation
        with pytest.raises(Exception):  # ValueError or similar
            encoder = TransformerEncoderLayer(config)
            x = torch.randn(2, 4, 100)
            encoder(x)

    def test_encoder_dropout_behavior(self):
        """Should apply dropout in training mode."""
        config = LayerConfig(
            layer_type="transformer_encoder",
            input_dim=64,
            output_dim=64,
            dropout=0.5,  # High dropout for observable effect
            activation="relu"
        )

        encoder = TransformerEncoderLayer(config)
        x = torch.randn(2, 4, 64)

        # Training mode - should have different outputs
        encoder.train()
        output1 = encoder(x)
        output2 = encoder(x)

        # Outputs should be different due to dropout
        assert not torch.allclose(output1, output2, atol=1e-6)

        # Eval mode - should have same outputs
        encoder.eval()
        with torch.no_grad():
            output3 = encoder(x)
            output4 = encoder(x)

        assert torch.allclose(output3, output4)

    def test_encoder_layer_params_defaults(self):
        """Should use default values when layer_params not provided."""
        config = LayerConfig(
            layer_type="transformer_encoder",
            input_dim=64,
            output_dim=128,
            dropout=0.1,
            activation="relu"
            # No layer_params provided
        )

        encoder = TransformerEncoderLayer(config)

        # Should use defaults
        assert encoder.num_heads == 8  # Default
        assert encoder.dim_feedforward == 2048  # Default
        assert encoder.num_layers == 1  # Default

    def test_encoder_gradient_flow(self):
        """Should allow gradient flow in training mode."""
        config = LayerConfig(
            layer_type="transformer_encoder",
            input_dim=32,
            output_dim=64,
            dropout=0.1,
            activation="relu"
        )

        encoder = TransformerEncoderLayer(config)
        encoder.train()

        x = torch.randn(2, 4, 32, requires_grad=True)
        output = encoder(x)

        # Compute dummy loss and backpropagate
        loss = output.sum()
        loss.backward()

        # Check that gradients exist
        assert x.grad is not None
        assert any(p.grad is not None for p in encoder.parameters())

    def test_encoder_device_compatibility(self):
        """Should work with different devices."""
        config = LayerConfig(
            layer_type="transformer_encoder",
            input_dim=32,
            output_dim=64,
            dropout=0.1,
            activation="relu"
        )

        encoder = TransformerEncoderLayer(config)

        # Test CPU
        x_cpu = torch.randn(2, 4, 32)
        output_cpu = encoder(x_cpu)
        assert output_cpu.device == x_cpu.device

        # Test CUDA if available
        if torch.cuda.is_available():
            encoder_cuda = encoder.cuda()
            x_cuda = x_cpu.cuda()
            output_cuda = encoder_cuda(x_cuda)
            assert output_cuda.device == x_cuda.device