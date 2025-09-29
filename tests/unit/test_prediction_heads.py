"""
Unit tests for prediction heads (RegressionHead and ClassificationHead).
"""

import pytest
import torch
from transformervae.config.basic_config import LayerConfig
from transformervae.models.layer import RegressionHead, ClassificationHead, LayerInterface


class TestRegressionHead:
    """Test RegressionHead implementation."""

    def test_regression_head_creation(self):
        """Should create regression head with valid configuration."""
        config = LayerConfig(
            layer_type="regression_head",
            input_dim=128,
            output_dim=1,
            dropout=0.1,
            activation="relu",
            layer_params={"hidden_dims": [64, 32]}
        )

        head = RegressionHead(config)

        assert isinstance(head, LayerInterface)
        assert head.input_dim == 128
        assert head.output_dim == 1
        assert hasattr(head, 'mlp')

    def test_regression_head_simple_architecture(self):
        """Should work with no hidden layers."""
        config = LayerConfig(
            layer_type="regression_head",
            input_dim=64,
            output_dim=1,
            dropout=0.0,
            activation="relu",
            layer_params={"hidden_dims": []}
        )

        head = RegressionHead(config)

        x = torch.randn(4, 64)
        output = head(x)

        assert output.shape == (4, 1)

    def test_regression_head_multi_hidden(self):
        """Should work with multiple hidden layers."""
        config = LayerConfig(
            layer_type="regression_head",
            input_dim=256,
            output_dim=1,
            dropout=0.1,
            activation="gelu",
            layer_params={"hidden_dims": [128, 64, 32]}
        )

        head = RegressionHead(config)

        x = torch.randn(3, 256)
        output = head(x)

        assert output.shape == (3, 1)

    def test_regression_head_multi_output(self):
        """Should support multiple output dimensions."""
        config = LayerConfig(
            layer_type="regression_head",
            input_dim=128,
            output_dim=5,
            dropout=0.1,
            activation="relu",
            layer_params={"hidden_dims": [64]}
        )

        head = RegressionHead(config)

        x = torch.randn(2, 128)
        output = head(x)

        assert output.shape == (2, 5)

    def test_regression_head_different_activations(self):
        """Should support different activation functions."""
        activations = ["relu", "gelu", "tanh", "sigmoid", "leaky_relu"]

        for activation in activations:
            config = LayerConfig(
                layer_type="regression_head",
                input_dim=64,
                output_dim=1,
                dropout=0.1,
                activation=activation,
                layer_params={"hidden_dims": [32]}
            )

            head = RegressionHead(config)
            x = torch.randn(2, 64)
            output = head(x)

            assert output.shape == (2, 1)

    def test_regression_head_dropout_behavior(self):
        """Should apply dropout in training mode."""
        config = LayerConfig(
            layer_type="regression_head",
            input_dim=64,
            output_dim=1,
            dropout=0.5,
            activation="relu",
            layer_params={"hidden_dims": [32]}
        )

        head = RegressionHead(config)
        x = torch.randn(3, 64)

        # Training mode - different outputs
        head.train()
        output1 = head(x)
        output2 = head(x)
        assert not torch.allclose(output1, output2, atol=1e-6)

        # Eval mode - same outputs
        head.eval()
        with torch.no_grad():
            output3 = head(x)
            output4 = head(x)
        assert torch.allclose(output3, output4)

    def test_regression_head_gradient_flow(self):
        """Should allow gradient flow."""
        config = LayerConfig(
            layer_type="regression_head",
            input_dim=32,
            output_dim=1,
            dropout=0.1,
            activation="relu",
            layer_params={"hidden_dims": [16]}
        )

        head = RegressionHead(config)
        head.train()

        x = torch.randn(2, 32, requires_grad=True)
        output = head(x)

        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert any(p.grad is not None for p in head.parameters())

    def test_regression_head_batch_independence(self):
        """Should process batch items independently."""
        config = LayerConfig(
            layer_type="regression_head",
            input_dim=32,
            output_dim=1,
            dropout=0.0,
            activation="relu",
            layer_params={"hidden_dims": [16]}
        )

        head = RegressionHead(config)
        head.eval()

        # Process individually
        x1 = torch.randn(1, 32)
        x2 = torch.randn(1, 32)

        with torch.no_grad():
            output1 = head(x1)
            output2 = head(x2)

        # Process as batch
        x_batch = torch.cat([x1, x2], dim=0)
        with torch.no_grad():
            output_batch = head(x_batch)

        # Results should be identical
        assert torch.allclose(output_batch[0:1], output1)
        assert torch.allclose(output_batch[1:2], output2)


class TestClassificationHead:
    """Test ClassificationHead implementation."""

    def test_classification_head_creation(self):
        """Should create classification head with valid configuration."""
        config = LayerConfig(
            layer_type="classification_head",
            input_dim=128,
            output_dim=10,
            dropout=0.1,
            activation="relu",
            layer_params={"hidden_dims": [64, 32]}
        )

        head = ClassificationHead(config)

        assert isinstance(head, LayerInterface)
        assert head.input_dim == 128
        assert head.output_dim == 10
        assert hasattr(head, 'mlp')

    def test_classification_head_binary(self):
        """Should work for binary classification."""
        config = LayerConfig(
            layer_type="classification_head",
            input_dim=64,
            output_dim=2,
            dropout=0.1,
            activation="relu",
            layer_params={"hidden_dims": [32]}
        )

        head = ClassificationHead(config)

        x = torch.randn(4, 64)
        output = head(x)

        assert output.shape == (4, 2)

    def test_classification_head_multiclass(self):
        """Should work for multiclass classification."""
        config = LayerConfig(
            layer_type="classification_head",
            input_dim=128,
            output_dim=50,
            dropout=0.1,
            activation="gelu",
            layer_params={"hidden_dims": [64]}
        )

        head = ClassificationHead(config)

        x = torch.randn(8, 128)
        output = head(x)

        assert output.shape == (8, 50)

    def test_classification_head_no_hidden(self):
        """Should work with no hidden layers."""
        config = LayerConfig(
            layer_type="classification_head",
            input_dim=32,
            output_dim=5,
            dropout=0.0,
            activation="relu",
            layer_params={"hidden_dims": []}
        )

        head = ClassificationHead(config)

        x = torch.randn(3, 32)
        output = head(x)

        assert output.shape == (3, 5)

    def test_classification_head_deep_architecture(self):
        """Should work with deep architectures."""
        config = LayerConfig(
            layer_type="classification_head",
            input_dim=512,
            output_dim=100,
            dropout=0.2,
            activation="relu",
            layer_params={"hidden_dims": [256, 128, 64, 32]}
        )

        head = ClassificationHead(config)

        x = torch.randn(2, 512)
        output = head(x)

        assert output.shape == (2, 100)

    def test_classification_head_different_activations(self):
        """Should support different activation functions."""
        activations = ["relu", "gelu", "tanh", "sigmoid", "leaky_relu"]

        for activation in activations:
            config = LayerConfig(
                layer_type="classification_head",
                input_dim=64,
                output_dim=10,
                dropout=0.1,
                activation=activation,
                layer_params={"hidden_dims": [32]}
            )

            head = ClassificationHead(config)
            x = torch.randn(2, 64)
            output = head(x)

            assert output.shape == (2, 10)

    def test_classification_head_logits_output(self):
        """Should output raw logits (no softmax)."""
        config = LayerConfig(
            layer_type="classification_head",
            input_dim=64,
            output_dim=5,
            dropout=0.0,
            activation="relu",
            layer_params={"hidden_dims": [32]}
        )

        head = ClassificationHead(config)
        head.eval()

        x = torch.randn(3, 64)
        with torch.no_grad():
            logits = head(x)

        # Should output raw logits (can be negative, positive, any range)
        assert logits.shape == (3, 5)
        # Logits should not be normalized (no softmax applied)
        probs = torch.softmax(logits, dim=-1)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(3))

    def test_classification_head_dropout_behavior(self):
        """Should apply dropout in training mode."""
        config = LayerConfig(
            layer_type="classification_head",
            input_dim=64,
            output_dim=10,
            dropout=0.5,
            activation="relu",
            layer_params={"hidden_dims": [32]}
        )

        head = ClassificationHead(config)
        x = torch.randn(3, 64)

        # Training mode - different outputs
        head.train()
        output1 = head(x)
        output2 = head(x)
        assert not torch.allclose(output1, output2, atol=1e-6)

        # Eval mode - same outputs
        head.eval()
        with torch.no_grad():
            output3 = head(x)
            output4 = head(x)
        assert torch.allclose(output3, output4)

    def test_classification_head_gradient_flow(self):
        """Should allow gradient flow."""
        config = LayerConfig(
            layer_type="classification_head",
            input_dim=32,
            output_dim=5,
            dropout=0.1,
            activation="relu",
            layer_params={"hidden_dims": [16]}
        )

        head = ClassificationHead(config)
        head.train()

        x = torch.randn(2, 32, requires_grad=True)
        output = head(x)

        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert any(p.grad is not None for p in head.parameters())

    def test_classification_head_batch_independence(self):
        """Should process batch items independently."""
        config = LayerConfig(
            layer_type="classification_head",
            input_dim=32,
            output_dim=5,
            dropout=0.0,
            activation="relu",
            layer_params={"hidden_dims": [16]}
        )

        head = ClassificationHead(config)
        head.eval()

        # Process individually
        x1 = torch.randn(1, 32)
        x2 = torch.randn(1, 32)

        with torch.no_grad():
            output1 = head(x1)
            output2 = head(x2)

        # Process as batch
        x_batch = torch.cat([x1, x2], dim=0)
        with torch.no_grad():
            output_batch = head(x_batch)

        # Results should be identical
        assert torch.allclose(output_batch[0:1], output1)
        assert torch.allclose(output_batch[1:2], output2)

    def test_classification_head_device_compatibility(self):
        """Should work with different devices."""
        config = LayerConfig(
            layer_type="classification_head",
            input_dim=32,
            output_dim=10,
            dropout=0.1,
            activation="relu",
            layer_params={"hidden_dims": [16]}
        )

        head = ClassificationHead(config)

        # Test CPU
        x_cpu = torch.randn(2, 32)
        output_cpu = head(x_cpu)
        assert output_cpu.device == x_cpu.device

        # Test CUDA if available
        if torch.cuda.is_available():
            head_cuda = head.cuda()
            x_cuda = x_cpu.cuda()
            output_cuda = head_cuda(x_cuda)
            assert output_cuda.device == x_cuda.device

    def test_prediction_heads_layer_params_defaults(self):
        """Should use default values when layer_params not specified."""
        # Test regression head
        config_reg = LayerConfig(
            layer_type="regression_head",
            input_dim=64,
            output_dim=1,
            dropout=0.1,
            activation="relu"
            # No layer_params
        )

        head_reg = RegressionHead(config_reg)
        x = torch.randn(2, 64)
        output_reg = head_reg(x)
        assert output_reg.shape == (2, 1)

        # Test classification head
        config_cls = LayerConfig(
            layer_type="classification_head",
            input_dim=64,
            output_dim=5,
            dropout=0.1,
            activation="relu"
            # No layer_params
        )

        head_cls = ClassificationHead(config_cls)
        output_cls = head_cls(x)
        assert output_cls.shape == (2, 5)

    def test_prediction_heads_unknown_activation(self):
        """Should fall back to default activation for unknown activations."""
        config = LayerConfig(
            layer_type="regression_head",
            input_dim=32,
            output_dim=1,
            dropout=0.1,
            activation="unknown_activation",
            layer_params={"hidden_dims": [16]}
        )

        head = RegressionHead(config)
        x = torch.randn(2, 32)
        output = head(x)

        # Should still work (fall back to ReLU)
        assert output.shape == (2, 1)

    def test_prediction_heads_numerical_stability(self):
        """Should maintain numerical stability."""
        config = LayerConfig(
            layer_type="classification_head",
            input_dim=32,
            output_dim=10,
            dropout=0.0,
            activation="relu",
            layer_params={"hidden_dims": [16]}
        )

        head = ClassificationHead(config)
        head.eval()

        # Test with very small values
        x_small = torch.randn(2, 32) * 1e-8
        with torch.no_grad():
            output_small = head(x_small)
        assert torch.isfinite(output_small).all()

        # Test with large values
        x_large = torch.randn(2, 32) * 1e8
        with torch.no_grad():
            output_large = head(x_large)
        assert torch.isfinite(output_large).all()

    def test_prediction_heads_edge_cases(self):
        """Should handle edge cases correctly."""
        # Single output dimension
        config_single = LayerConfig(
            layer_type="regression_head",
            input_dim=16,
            output_dim=1,
            dropout=0.0,
            activation="relu",
            layer_params={"hidden_dims": []}
        )

        head_single = RegressionHead(config_single)
        x_single = torch.randn(1, 16)
        output_single = head_single(x_single)
        assert output_single.shape == (1, 1)

        # Large number of classes
        config_large = LayerConfig(
            layer_type="classification_head",
            input_dim=64,
            output_dim=1000,
            dropout=0.0,
            activation="relu",
            layer_params={"hidden_dims": [32]}
        )

        head_large = ClassificationHead(config_large)
        x_large = torch.randn(2, 64)
        output_large = head_large(x_large)
        assert output_large.shape == (2, 1000)