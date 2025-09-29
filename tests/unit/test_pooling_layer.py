"""
Unit tests for PoolingLayer.
"""

import pytest
import torch
from transformervae.config.basic_config import LayerConfig
from transformervae.models.layer import PoolingLayer, LayerInterface


class TestPoolingLayer:
    """Test PoolingLayer implementation."""

    def test_pooling_creation(self):
        """Should create pooling layer with valid configuration."""
        config = LayerConfig(
            layer_type="pooling",
            input_dim=128,
            output_dim=64,
            dropout=0.1,
            activation="relu",
            layer_params={"pooling_type": "mean"}
        )

        pooling = PoolingLayer(config)

        assert isinstance(pooling, LayerInterface)
        assert pooling.input_dim == 128
        assert pooling.output_dim == 64
        assert pooling.pooling_type == "mean"

    def test_mean_pooling(self):
        """Should perform mean pooling correctly."""
        config = LayerConfig(
            layer_type="pooling",
            input_dim=64,
            output_dim=64,
            dropout=0.0,
            activation="relu",
            layer_params={"pooling_type": "mean"}
        )

        pooling = PoolingLayer(config)
        pooling.eval()

        batch_size, seq_len = 3, 8
        x = torch.randn(batch_size, seq_len, 64)

        with torch.no_grad():
            output = pooling(x)

        assert output.shape == (batch_size, 64)

        # Check that mean pooling is correct
        expected = x.mean(dim=1)
        assert torch.allclose(output, expected, atol=1e-6)

    def test_max_pooling(self):
        """Should perform max pooling correctly."""
        config = LayerConfig(
            layer_type="pooling",
            input_dim=64,
            output_dim=64,
            dropout=0.0,
            activation="relu",
            layer_params={"pooling_type": "max"}
        )

        pooling = PoolingLayer(config)
        pooling.eval()

        batch_size, seq_len = 2, 6
        x = torch.randn(batch_size, seq_len, 64)

        with torch.no_grad():
            output = pooling(x)

        assert output.shape == (batch_size, 64)

        # Check that max pooling is correct
        expected = x.max(dim=1)[0]
        assert torch.allclose(output, expected, atol=1e-6)

    def test_attention_pooling(self):
        """Should perform attention-based pooling."""
        config = LayerConfig(
            layer_type="pooling",
            input_dim=64,
            output_dim=64,
            dropout=0.0,
            activation="relu",
            layer_params={"pooling_type": "attention", "num_heads": 4}
        )

        pooling = PoolingLayer(config)
        pooling.eval()

        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, 64)

        with torch.no_grad():
            output = pooling(x)

        assert output.shape == (batch_size, 64)
        assert hasattr(pooling, 'attention')
        assert hasattr(pooling, 'query')

    def test_masked_mean_pooling(self):
        """Should handle masking for mean pooling."""
        config = LayerConfig(
            layer_type="pooling",
            input_dim=32,
            output_dim=32,
            dropout=0.0,
            activation="relu",
            layer_params={"pooling_type": "mean"}
        )

        pooling = PoolingLayer(config)
        pooling.eval()

        batch_size, seq_len = 3, 8
        x = torch.randn(batch_size, seq_len, 32)

        # Create mask (True for valid positions)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        mask[:, 5:] = False  # Mask last 3 positions

        with torch.no_grad():
            output = pooling(x, mask=mask)

        assert output.shape == (batch_size, 32)

        # Manually compute masked mean
        mask_expanded = mask.unsqueeze(-1).float()
        x_masked = x * mask_expanded
        expected = x_masked.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        assert torch.allclose(output, expected, atol=1e-6)

    def test_masked_max_pooling(self):
        """Should handle masking for max pooling."""
        config = LayerConfig(
            layer_type="pooling",
            input_dim=32,
            output_dim=32,
            dropout=0.0,
            activation="relu",
            layer_params={"pooling_type": "max"}
        )

        pooling = PoolingLayer(config)
        pooling.eval()

        batch_size, seq_len = 2, 6
        x = torch.randn(batch_size, seq_len, 32)

        # Create mask (True for valid positions)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        mask[:, 4:] = False  # Mask last 2 positions

        with torch.no_grad():
            output = pooling(x, mask=mask)

        assert output.shape == (batch_size, 32)

    def test_masked_attention_pooling(self):
        """Should handle masking for attention pooling."""
        config = LayerConfig(
            layer_type="pooling",
            input_dim=64,
            output_dim=64,
            dropout=0.0,
            activation="relu",
            layer_params={"pooling_type": "attention", "num_heads": 2}
        )

        pooling = PoolingLayer(config)
        pooling.eval()

        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, 64)

        # Create mask (True for valid positions)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        mask[:, 6:] = False  # Mask last 2 positions

        with torch.no_grad():
            output = pooling(x, mask=mask)

        assert output.shape == (batch_size, 64)

    def test_pooling_with_projection(self):
        """Should apply output projection when dimensions differ."""
        config = LayerConfig(
            layer_type="pooling",
            input_dim=128,
            output_dim=64,
            dropout=0.1,
            activation="relu",
            layer_params={"pooling_type": "mean"}
        )

        pooling = PoolingLayer(config)

        # Should have output projection
        assert pooling.output_projection is not None

        x = torch.randn(2, 5, 128)
        output = pooling(x)
        assert output.shape == (2, 64)

    def test_pooling_without_projection(self):
        """Should not use projection when dimensions match."""
        config = LayerConfig(
            layer_type="pooling",
            input_dim=64,
            output_dim=64,
            dropout=0.1,
            activation="relu",
            layer_params={"pooling_type": "max"}
        )

        pooling = PoolingLayer(config)

        # Should not have output projection
        assert pooling.output_projection is None

        x = torch.randn(2, 5, 64)
        output = pooling(x)
        assert output.shape == (2, 64)

    def test_pooling_dropout_behavior(self):
        """Should apply dropout in training mode."""
        config = LayerConfig(
            layer_type="pooling",
            input_dim=64,
            output_dim=64,
            dropout=0.5,
            activation="relu",
            layer_params={"pooling_type": "mean"}
        )

        pooling = PoolingLayer(config)
        x = torch.randn(3, 4, 64)

        # Training mode - different outputs
        pooling.train()
        output1 = pooling(x)
        output2 = pooling(x)
        assert not torch.allclose(output1, output2, atol=1e-6)

        # Eval mode - same outputs
        pooling.eval()
        with torch.no_grad():
            output3 = pooling(x)
            output4 = pooling(x)
        assert torch.allclose(output3, output4)

    def test_pooling_different_types(self):
        """Should support all pooling types."""
        pooling_types = ["mean", "max", "attention"]

        for pooling_type in pooling_types:
            config = LayerConfig(
                layer_type="pooling",
                input_dim=32,
                output_dim=32,
                dropout=0.1,
                activation="relu",
                layer_params={"pooling_type": pooling_type}
            )

            pooling = PoolingLayer(config)
            x = torch.randn(2, 6, 32)
            output = pooling(x)

            assert output.shape == (2, 32)

    def test_pooling_invalid_type(self):
        """Should raise error for invalid pooling type."""
        config = LayerConfig(
            layer_type="pooling",
            input_dim=64,
            output_dim=64,
            dropout=0.1,
            activation="relu",
            layer_params={"pooling_type": "invalid"}
        )

        pooling = PoolingLayer(config)
        x = torch.randn(2, 4, 64)

        with pytest.raises(ValueError, match="Unknown pooling type"):
            pooling(x)

    def test_pooling_batch_independence(self):
        """Should process batch items independently."""
        config = LayerConfig(
            layer_type="pooling",
            input_dim=32,
            output_dim=32,
            dropout=0.0,
            activation="relu",
            layer_params={"pooling_type": "mean"}
        )

        pooling = PoolingLayer(config)
        pooling.eval()

        # Process individually
        x1 = torch.randn(1, 5, 32)
        x2 = torch.randn(1, 5, 32)

        with torch.no_grad():
            output1 = pooling(x1)
            output2 = pooling(x2)

        # Process as batch
        x_batch = torch.cat([x1, x2], dim=0)
        with torch.no_grad():
            output_batch = pooling(x_batch)

        # Results should be identical
        assert torch.allclose(output_batch[0:1], output1)
        assert torch.allclose(output_batch[1:2], output2)

    def test_pooling_gradient_flow(self):
        """Should allow gradient flow."""
        config = LayerConfig(
            layer_type="pooling",
            input_dim=32,
            output_dim=64,
            dropout=0.1,
            activation="relu",
            layer_params={"pooling_type": "mean"}
        )

        pooling = PoolingLayer(config)
        pooling.train()

        x = torch.randn(2, 4, 32, requires_grad=True)
        output = pooling(x)

        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert any(p.grad is not None for p in pooling.parameters())

    def test_pooling_device_compatibility(self):
        """Should work with different devices."""
        config = LayerConfig(
            layer_type="pooling",
            input_dim=32,
            output_dim=64,
            dropout=0.1,
            activation="relu",
            layer_params={"pooling_type": "max"}
        )

        pooling = PoolingLayer(config)

        # Test CPU
        x_cpu = torch.randn(2, 4, 32)
        output_cpu = pooling(x_cpu)
        assert output_cpu.device == x_cpu.device

        # Test CUDA if available
        if torch.cuda.is_available():
            pooling_cuda = pooling.cuda()
            x_cuda = x_cpu.cuda()
            output_cuda = pooling_cuda(x_cuda)
            assert output_cuda.device == x_cuda.device

    def test_attention_pooling_parameters(self):
        """Should create learnable parameters for attention pooling."""
        config = LayerConfig(
            layer_type="pooling",
            input_dim=64,
            output_dim=64,
            dropout=0.0,
            activation="relu",
            layer_params={"pooling_type": "attention", "num_heads": 4}
        )

        pooling = PoolingLayer(config)

        # Check attention parameters exist
        assert hasattr(pooling, 'attention')
        assert hasattr(pooling, 'query')
        assert pooling.query.shape == (1, 1, 64)
        assert pooling.query.requires_grad

    def test_pooling_layer_params_defaults(self):
        """Should use default values when layer_params not specified."""
        config = LayerConfig(
            layer_type="pooling",
            input_dim=64,
            output_dim=64,
            dropout=0.1,
            activation="relu"
            # No layer_params
        )

        pooling = PoolingLayer(config)

        # Should use default pooling type
        assert pooling.pooling_type == "mean"

    def test_pooling_edge_cases(self):
        """Should handle edge cases correctly."""
        config = LayerConfig(
            layer_type="pooling",
            input_dim=16,
            output_dim=16,
            dropout=0.0,
            activation="relu",
            layer_params={"pooling_type": "mean"}
        )

        pooling = PoolingLayer(config)
        pooling.eval()

        # Single sequence element
        x_single = torch.randn(2, 1, 16)
        with torch.no_grad():
            output_single = pooling(x_single)
        assert output_single.shape == (2, 16)

        # Long sequence
        x_long = torch.randn(1, 100, 16)
        with torch.no_grad():
            output_long = pooling(x_long)
        assert output_long.shape == (1, 16)

    def test_pooling_numerical_stability(self):
        """Should maintain numerical stability."""
        config = LayerConfig(
            layer_type="pooling",
            input_dim=32,
            output_dim=32,
            dropout=0.0,
            activation="relu",
            layer_params={"pooling_type": "mean"}
        )

        pooling = PoolingLayer(config)
        pooling.eval()

        # Test with very small values
        x_small = torch.randn(2, 5, 32) * 1e-8
        with torch.no_grad():
            output_small = pooling(x_small)
        assert torch.isfinite(output_small).all()

        # Test with large values
        x_large = torch.randn(2, 5, 32) * 1e8
        with torch.no_grad():
            output_large = pooling(x_large)
        assert torch.isfinite(output_large).all()