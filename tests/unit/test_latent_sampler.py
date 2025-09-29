"""
Unit tests for LatentSampler (VAE reparameterization layer).
"""

import pytest
import torch
from transformervae.config.basic_config import LayerConfig
from transformervae.models.layer import LatentSampler, LayerInterface


class TestLatentSampler:
    """Test LatentSampler implementation."""

    def test_sampler_creation(self):
        """Should create sampler with valid configuration."""
        config = LayerConfig(
            layer_type="latent_sampler",
            input_dim=256,
            output_dim=64,
            dropout=0.0,
            activation="linear"
        )

        sampler = LatentSampler(config)

        assert isinstance(sampler, LayerInterface)
        assert sampler.input_dim == 256
        assert sampler.output_dim == 64
        assert hasattr(sampler, 'mu_layer')
        assert hasattr(sampler, 'logvar_layer')

    def test_sampler_forward_pass(self):
        """Should perform forward pass with reparameterization."""
        config = LayerConfig(
            layer_type="latent_sampler",
            input_dim=128,
            output_dim=32,
            dropout=0.0,
            activation="linear"
        )

        sampler = LatentSampler(config)
        sampler.train()  # Enable training mode for stochastic sampling

        batch_size = 4
        x = torch.randn(batch_size, 128)

        mu, logvar, z = sampler(x)

        # Check output shapes
        assert mu.shape == (batch_size, 32)
        assert logvar.shape == (batch_size, 32)
        assert z.shape == (batch_size, 32)

        # Check output types
        assert mu.dtype == torch.float32
        assert logvar.dtype == torch.float32
        assert z.dtype == torch.float32

    def test_sampler_deterministic_mode(self):
        """Should return mu in deterministic mode."""
        config = LayerConfig(
            layer_type="latent_sampler",
            input_dim=64,
            output_dim=16,
            dropout=0.0,
            activation="linear"
        )

        sampler = LatentSampler(config)
        x = torch.randn(3, 64)

        # Deterministic mode
        mu1, logvar1, z1 = sampler(x, deterministic=True)
        mu2, logvar2, z2 = sampler(x, deterministic=True)

        # mu and logvar should be identical
        assert torch.allclose(mu1, mu2)
        assert torch.allclose(logvar1, logvar2)

        # z should equal mu in deterministic mode
        assert torch.allclose(z1, mu1)
        assert torch.allclose(z2, mu2)

    def test_sampler_stochastic_mode(self):
        """Should sample differently in stochastic mode."""
        config = LayerConfig(
            layer_type="latent_sampler",
            input_dim=64,
            output_dim=16,
            dropout=0.0,
            activation="linear"
        )

        sampler = LatentSampler(config)
        sampler.train()  # Enable training mode
        x = torch.randn(3, 64)

        # Stochastic mode (default)
        mu1, logvar1, z1 = sampler(x, deterministic=False)
        mu2, logvar2, z2 = sampler(x, deterministic=False)

        # mu and logvar should be identical (deterministic)
        assert torch.allclose(mu1, mu2)
        assert torch.allclose(logvar1, logvar2)

        # z should be different (stochastic)
        assert not torch.allclose(z1, z2, atol=1e-6)

    def test_sampler_eval_mode_behavior(self):
        """Should behave deterministically in eval mode."""
        config = LayerConfig(
            layer_type="latent_sampler",
            input_dim=64,
            output_dim=16,
            dropout=0.0,
            activation="linear"
        )

        sampler = LatentSampler(config)
        sampler.eval()  # Set to eval mode
        x = torch.randn(3, 64)

        # In eval mode, should be deterministic even without explicit flag
        mu1, logvar1, z1 = sampler(x)
        mu2, logvar2, z2 = sampler(x)

        assert torch.allclose(mu1, mu2)
        assert torch.allclose(logvar1, logvar2)
        assert torch.allclose(z1, z2)  # Should be deterministic in eval mode

    def test_sampler_reparameterization_trick(self):
        """Should implement proper reparameterization trick."""
        config = LayerConfig(
            layer_type="latent_sampler",
            input_dim=64,
            output_dim=16,
            dropout=0.0,
            activation="linear"
        )

        sampler = LatentSampler(config)
        sampler.train()

        x = torch.randn(100, 64)  # Larger batch for statistical properties

        mu, logvar, z = sampler(x)

        # Check that z follows proper distribution properties
        # z should have similar mean to mu (approximately)
        z_mean = z.mean(dim=0)
        mu_mean = mu.mean(dim=0)
        assert torch.allclose(z_mean, mu_mean, atol=0.5)  # Allow some variance

        # Check that logvar affects the variance of z
        std = torch.exp(0.5 * logvar)
        z_std = z.std(dim=0)
        std_mean = std.mean(dim=0)
        assert torch.allclose(z_std, std_mean, atol=0.5)  # Approximate check

    def test_sampler_gradient_flow(self):
        """Should allow gradient flow through reparameterization."""
        config = LayerConfig(
            layer_type="latent_sampler",
            input_dim=32,
            output_dim=8,
            dropout=0.0,
            activation="linear"
        )

        sampler = LatentSampler(config)
        sampler.train()

        x = torch.randn(4, 32, requires_grad=True)
        mu, logvar, z = sampler(x)

        # Compute loss and backpropagate
        loss = z.sum() + mu.sum() + logvar.sum()
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        assert sampler.mu_layer.weight.grad is not None
        assert sampler.logvar_layer.weight.grad is not None

    def test_sampler_with_dropout(self):
        """Should apply dropout to input."""
        config = LayerConfig(
            layer_type="latent_sampler",
            input_dim=64,
            output_dim=16,
            dropout=0.5,  # High dropout
            activation="linear"
        )

        sampler = LatentSampler(config)
        x = torch.randn(2, 64)

        # Training mode - dropout active
        sampler.train()
        mu1, _, _ = sampler(x)
        mu2, _, _ = sampler(x)
        assert not torch.allclose(mu1, mu2, atol=1e-6)

        # Eval mode - no dropout
        sampler.eval()
        with torch.no_grad():
            mu3, _, _ = sampler(x)
            mu4, _, _ = sampler(x)
        assert torch.allclose(mu3, mu4)

    def test_sampler_different_dimensions(self):
        """Should work with different input/output dimensions."""
        test_cases = [
            (512, 128),
            (256, 64),
            (128, 32),
            (64, 16),
        ]

        for input_dim, output_dim in test_cases:
            config = LayerConfig(
                layer_type="latent_sampler",
                input_dim=input_dim,
                output_dim=output_dim,
                dropout=0.0,
                activation="linear"
            )

            sampler = LatentSampler(config)
            x = torch.randn(2, input_dim)
            mu, logvar, z = sampler(x)

            assert mu.shape == (2, output_dim)
            assert logvar.shape == (2, output_dim)
            assert z.shape == (2, output_dim)

    def test_sampler_kl_divergence_computation(self):
        """Should enable KL divergence computation from outputs."""
        config = LayerConfig(
            layer_type="latent_sampler",
            input_dim=64,
            output_dim=16,
            dropout=0.0,
            activation="linear"
        )

        sampler = LatentSampler(config)
        x = torch.randn(4, 64)

        mu, logvar, z = sampler(x)

        # Compute KL divergence: KL(q(z|x) || p(z)) where p(z) = N(0,I)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

        assert kl_div.shape == (4,)  # One KL value per batch item
        assert torch.all(kl_div >= 0)  # KL divergence should be non-negative

    def test_sampler_reconstruction_capability(self):
        """Should maintain information for reconstruction."""
        config = LayerConfig(
            layer_type="latent_sampler",
            input_dim=128,
            output_dim=64,
            dropout=0.0,
            activation="linear"
        )

        sampler = LatentSampler(config)
        sampler.eval()  # Deterministic mode

        x = torch.randn(1, 128)

        # Sample multiple times
        samples = []
        for _ in range(10):
            mu, logvar, z = sampler(x, deterministic=True)
            samples.append(z)

        # All samples should be identical in deterministic mode
        for i in range(1, len(samples)):
            assert torch.allclose(samples[0], samples[i])

    def test_sampler_device_compatibility(self):
        """Should work with different devices."""
        config = LayerConfig(
            layer_type="latent_sampler",
            input_dim=32,
            output_dim=8,
            dropout=0.0,
            activation="linear"
        )

        sampler = LatentSampler(config)

        # Test CPU
        x_cpu = torch.randn(2, 32)
        mu_cpu, logvar_cpu, z_cpu = sampler(x_cpu)
        assert mu_cpu.device == x_cpu.device
        assert logvar_cpu.device == x_cpu.device
        assert z_cpu.device == x_cpu.device

        # Test CUDA if available
        if torch.cuda.is_available():
            sampler_cuda = sampler.cuda()
            x_cuda = x_cpu.cuda()
            mu_cuda, logvar_cuda, z_cuda = sampler_cuda(x_cuda)
            assert mu_cuda.device == x_cuda.device
            assert logvar_cuda.device == x_cuda.device
            assert z_cuda.device == x_cuda.device

    def test_sampler_layer_dimensions(self):
        """Should have correctly sized linear layers."""
        config = LayerConfig(
            layer_type="latent_sampler",
            input_dim=100,
            output_dim=50,
            dropout=0.0,
            activation="linear"
        )

        sampler = LatentSampler(config)

        # Check layer dimensions
        assert sampler.mu_layer.in_features == 100
        assert sampler.mu_layer.out_features == 50
        assert sampler.logvar_layer.in_features == 100
        assert sampler.logvar_layer.out_features == 50

    def test_sampler_batch_independence(self):
        """Should process batch items independently."""
        config = LayerConfig(
            layer_type="latent_sampler",
            input_dim=32,
            output_dim=8,
            dropout=0.0,
            activation="linear"
        )

        sampler = LatentSampler(config)
        sampler.eval()

        # Process individually
        x1 = torch.randn(1, 32)
        x2 = torch.randn(1, 32)

        mu1, logvar1, z1 = sampler(x1, deterministic=True)
        mu2, logvar2, z2 = sampler(x2, deterministic=True)

        # Process as batch
        x_batch = torch.cat([x1, x2], dim=0)
        mu_batch, logvar_batch, z_batch = sampler(x_batch, deterministic=True)

        # Results should be identical
        assert torch.allclose(mu_batch[0:1], mu1)
        assert torch.allclose(mu_batch[1:2], mu2)
        assert torch.allclose(logvar_batch[0:1], logvar1)
        assert torch.allclose(logvar_batch[1:2], logvar2)
        assert torch.allclose(z_batch[0:1], z1)
        assert torch.allclose(z_batch[1:2], z2)