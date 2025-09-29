"""
Unit tests for TransformerDecoderLayer.
"""

import pytest
import torch
from transformervae.config.basic_config import LayerConfig
from transformervae.models.layer import TransformerDecoderLayer, LayerInterface


class TestTransformerDecoderLayer:
    """Test TransformerDecoderLayer implementation."""

    def test_decoder_creation(self):
        """Should create decoder with valid configuration."""
        config = LayerConfig(
            layer_type="transformer_decoder",
            input_dim=64,
            output_dim=128,
            dropout=0.1,
            activation="gelu",
            layer_params={"num_heads": 4, "dim_feedforward": 256}
        )

        decoder = TransformerDecoderLayer(config)

        assert isinstance(decoder, LayerInterface)
        assert decoder.input_dim == 64
        assert decoder.output_dim == 128
        assert decoder.num_heads == 4
        assert decoder.dim_feedforward == 256

    def test_decoder_forward_pass_with_memory(self):
        """Should perform forward pass with encoder memory."""
        config = LayerConfig(
            layer_type="transformer_decoder",
            input_dim=64,
            output_dim=64,
            dropout=0.1,
            activation="relu"
        )

        decoder = TransformerDecoderLayer(config)
        decoder.eval()

        batch_size, tgt_len, src_len = 3, 8, 10
        tgt = torch.randn(batch_size, tgt_len, 64)
        memory = torch.randn(batch_size, src_len, 64)

        with torch.no_grad():
            output = decoder(tgt, memory=memory)

        assert output.shape == (batch_size, tgt_len, 64)

    def test_decoder_forward_pass_without_memory(self):
        """Should work without external memory (self-attention only)."""
        config = LayerConfig(
            layer_type="transformer_decoder",
            input_dim=32,
            output_dim=64,
            dropout=0.1,
            activation="relu"
        )

        decoder = TransformerDecoderLayer(config)
        decoder.eval()

        batch_size, seq_len = 2, 6
        x = torch.randn(batch_size, seq_len, 32)

        with torch.no_grad():
            output = decoder(x)  # No memory provided

        assert output.shape == (batch_size, seq_len, 64)

    def test_decoder_with_causal_mask(self):
        """Should support causal masking for autoregressive generation."""
        config = LayerConfig(
            layer_type="transformer_decoder",
            input_dim=64,
            output_dim=64,
            dropout=0.0,
            activation="relu"
        )

        decoder = TransformerDecoderLayer(config)
        decoder.eval()

        batch_size, seq_len = 2, 5
        tgt = torch.randn(batch_size, seq_len, 64)

        # Create causal mask (upper triangular)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

        with torch.no_grad():
            output = decoder(tgt, tgt_mask=causal_mask)

        assert output.shape == (batch_size, seq_len, 64)

    def test_decoder_different_input_output_dim(self):
        """Should handle different input and output dimensions."""
        config = LayerConfig(
            layer_type="transformer_decoder",
            input_dim=32,
            output_dim=128,
            dropout=0.1,
            activation="gelu"
        )

        decoder = TransformerDecoderLayer(config)

        # Should have input projection
        assert decoder.input_projection is not None

        x = torch.randn(2, 4, 32)
        output = decoder(x)
        assert output.shape == (2, 4, 128)

    def test_decoder_same_input_output_dim(self):
        """Should handle case where input_dim == output_dim."""
        config = LayerConfig(
            layer_type="transformer_decoder",
            input_dim=64,
            output_dim=64,
            dropout=0.1,
            activation="relu"
        )

        decoder = TransformerDecoderLayer(config)

        # Should not have input projection
        assert decoder.input_projection is None

        x = torch.randn(2, 4, 64)
        output = decoder(x)
        assert output.shape == (2, 4, 64)

    def test_decoder_multiple_layers(self):
        """Should support multiple decoder layers."""
        config = LayerConfig(
            layer_type="transformer_decoder",
            input_dim=64,
            output_dim=64,
            dropout=0.1,
            activation="relu",
            layer_params={"num_heads": 8, "num_layers": 4}
        )

        decoder = TransformerDecoderLayer(config)
        assert decoder.num_layers == 4

        x = torch.randn(2, 6, 64)
        output = decoder(x)
        assert output.shape == (2, 6, 64)

    def test_decoder_different_activations(self):
        """Should support different activation functions."""
        activations = ["relu", "gelu", "tanh"]

        for activation in activations:
            config = LayerConfig(
                layer_type="transformer_decoder",
                input_dim=32,
                output_dim=32,
                dropout=0.1,
                activation=activation
            )

            decoder = TransformerDecoderLayer(config)
            x = torch.randn(2, 4, 32)
            output = decoder(x)

            assert output.shape == (2, 4, 32)

    def test_decoder_with_memory_mask(self):
        """Should support memory masking."""
        config = LayerConfig(
            layer_type="transformer_decoder",
            input_dim=64,
            output_dim=64,
            dropout=0.0,
            activation="relu"
        )

        decoder = TransformerDecoderLayer(config)
        decoder.eval()

        batch_size, tgt_len, src_len = 2, 6, 8
        tgt = torch.randn(batch_size, tgt_len, 64)
        memory = torch.randn(batch_size, src_len, 64)

        # Create memory mask
        memory_mask = torch.zeros(batch_size, src_len, dtype=torch.bool)
        memory_mask[:, 5:] = True  # Mask last 3 positions

        with torch.no_grad():
            output = decoder(tgt, memory=memory, memory_mask=memory_mask)

        assert output.shape == (batch_size, tgt_len, 64)

    def test_decoder_dropout_behavior(self):
        """Should apply dropout in training mode."""
        config = LayerConfig(
            layer_type="transformer_decoder",
            input_dim=64,
            output_dim=64,
            dropout=0.5,  # High dropout
            activation="relu"
        )

        decoder = TransformerDecoderLayer(config)
        x = torch.randn(2, 4, 64)

        # Training mode - different outputs
        decoder.train()
        output1 = decoder(x)
        output2 = decoder(x)
        assert not torch.allclose(output1, output2, atol=1e-6)

        # Eval mode - same outputs
        decoder.eval()
        with torch.no_grad():
            output3 = decoder(x)
            output4 = decoder(x)
        assert torch.allclose(output3, output4)

    def test_decoder_gradient_flow(self):
        """Should allow gradient flow."""
        config = LayerConfig(
            layer_type="transformer_decoder",
            input_dim=32,
            output_dim=64,
            dropout=0.1,
            activation="relu"
        )

        decoder = TransformerDecoderLayer(config)
        decoder.train()

        x = torch.randn(2, 4, 32, requires_grad=True)
        output = decoder(x)

        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert any(p.grad is not None for p in decoder.parameters())

    def test_decoder_layer_params_defaults(self):
        """Should use default values when layer_params not specified."""
        config = LayerConfig(
            layer_type="transformer_decoder",
            input_dim=64,
            output_dim=64,
            dropout=0.1,
            activation="relu"
            # No layer_params
        )

        decoder = TransformerDecoderLayer(config)

        assert decoder.num_heads == 8  # Default
        assert decoder.dim_feedforward == 2048  # Default
        assert decoder.num_layers == 1  # Default

    def test_decoder_cross_attention(self):
        """Should perform cross-attention with encoder memory."""
        config = LayerConfig(
            layer_type="transformer_decoder",
            input_dim=64,
            output_dim=64,
            dropout=0.0,
            activation="relu"
        )

        decoder = TransformerDecoderLayer(config)
        decoder.eval()

        batch_size = 2
        tgt_len, src_len = 5, 7
        tgt = torch.randn(batch_size, tgt_len, 64)
        memory = torch.randn(batch_size, src_len, 64)

        with torch.no_grad():
            output_with_memory = decoder(tgt, memory=memory)
            output_without_memory = decoder(tgt)  # Self-attention only

        # Outputs should be different (cross-attention vs self-attention)
        assert not torch.allclose(output_with_memory, output_without_memory, atol=1e-4)

    def test_decoder_device_compatibility(self):
        """Should work with different devices."""
        config = LayerConfig(
            layer_type="transformer_decoder",
            input_dim=32,
            output_dim=64,
            dropout=0.1,
            activation="relu"
        )

        decoder = TransformerDecoderLayer(config)

        # Test CPU
        x_cpu = torch.randn(2, 4, 32)
        output_cpu = decoder(x_cpu)
        assert output_cpu.device == x_cpu.device

        # Test CUDA if available
        if torch.cuda.is_available():
            decoder_cuda = decoder.cuda()
            x_cuda = x_cpu.cuda()
            output_cuda = decoder_cuda(x_cuda)
            assert output_cuda.device == x_cuda.device