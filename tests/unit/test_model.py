"""
Unit tests for TransformerVAE model implementation.
"""

import pytest
import torch
from transformervae.config.basic_config import ModelConfig, LayerConfig
from transformervae.models.model import ModelInterface, TransformerVAE, PositionalEncoding


class TestModelInterface:
    """Test ModelInterface abstract base class."""

    def test_model_interface_is_abstract(self):
        """Should not be able to instantiate ModelInterface directly."""
        config = ModelConfig(
            model_type="test",
            vocab_size=1000,
            embedding_dim=128,
            latent_dim=64,
            max_sequence_length=100,
            pad_token_id=0,
            encoder_layers=[],
            decoder_layers=[],
            latent_layer=LayerConfig(
                layer_type="latent_sampler",
                input_dim=128,
                output_dim=64,
                dropout=0.1,
                activation="linear"
            )
        )

        with pytest.raises(TypeError):
            ModelInterface(config)


class TestPositionalEncoding:
    """Test PositionalEncoding component."""

    def test_positional_encoding_creation(self):
        """Should create positional encoding with valid parameters."""
        pe = PositionalEncoding(d_model=128, max_len=512, dropout=0.1)

        assert hasattr(pe, 'pe')
        assert pe.pe.shape == (512, 1, 128)

    def test_positional_encoding_forward(self):
        """Should add positional encoding to input."""
        pe = PositionalEncoding(d_model=64, max_len=100, dropout=0.0)
        pe.eval()

        batch_size, seq_len = 4, 20
        x = torch.randn(batch_size, seq_len, 64)

        with torch.no_grad():
            output = pe(x)

        assert output.shape == (batch_size, seq_len, 64)

    def test_positional_encoding_different_lengths(self):
        """Should work with different sequence lengths."""
        pe = PositionalEncoding(d_model=32, max_len=1000, dropout=0.0)
        pe.eval()

        for seq_len in [10, 50, 100, 500]:
            x = torch.randn(2, seq_len, 32)
            with torch.no_grad():
                output = pe(x)
            assert output.shape == (2, seq_len, 32)

    def test_positional_encoding_dropout(self):
        """Should apply dropout in training mode."""
        pe = PositionalEncoding(d_model=64, max_len=100, dropout=0.5)
        x = torch.randn(3, 10, 64)

        # Training mode - different outputs
        pe.train()
        output1 = pe(x)
        output2 = pe(x)
        assert not torch.allclose(output1, output2, atol=1e-6)

        # Eval mode - same outputs (deterministic)
        pe.eval()
        with torch.no_grad():
            output3 = pe(x)
            output4 = pe(x)
        assert torch.allclose(output3, output4)


class TestTransformerVAE:
    """Test TransformerVAE model implementation."""

    def create_basic_config(self):
        """Create a basic model configuration for testing."""
        return ModelConfig(
            model_type="transformer_vae",
            vocab_size=1000,
            embedding_dim=128,
            latent_dim=64,
            max_sequence_length=100,
            pad_token_id=0,
            positional_dropout=0.1,
            encoder_layers=[
                LayerConfig(
                    layer_type="transformer_encoder",
                    input_dim=128,
                    output_dim=128,
                    dropout=0.1,
                    activation="relu",
                    layer_params={"num_heads": 4, "dim_feedforward": 256}
                ),
                LayerConfig(
                    layer_type="pooling",
                    input_dim=128,
                    output_dim=128,
                    dropout=0.1,
                    activation="relu",
                    layer_params={"pooling_type": "mean"}
                )
            ],
            decoder_layers=[
                LayerConfig(
                    layer_type="transformer_decoder",
                    input_dim=64 + 128,  # latent_dim + embedding_dim
                    output_dim=128,
                    dropout=0.1,
                    activation="relu",
                    layer_params={"num_heads": 4, "dim_feedforward": 256}
                )
            ],
            latent_layer=LayerConfig(
                layer_type="latent_sampler",
                input_dim=128,
                output_dim=64,
                dropout=0.1,
                activation="linear"
            )
        )

    def test_model_creation(self):
        """Should create TransformerVAE with valid configuration."""
        config = self.create_basic_config()
        model = TransformerVAE(config)

        assert isinstance(model, ModelInterface)
        assert model.vocab_size == 1000
        assert model.latent_dim == 64
        assert model.max_sequence_length == 100

        # Check components exist
        assert hasattr(model, 'token_embedding')
        assert hasattr(model, 'positional_encoding')
        assert hasattr(model, 'encoder_layers')
        assert hasattr(model, 'decoder_layers')
        assert hasattr(model, 'latent_sampler')
        assert hasattr(model, 'output_projection')

    def test_model_forward_pass(self):
        """Should perform complete forward pass."""
        config = self.create_basic_config()
        model = TransformerVAE(config)
        model.eval()

        batch_size, seq_len = 4, 20
        x = torch.randint(0, 1000, (batch_size, seq_len))

        with torch.no_grad():
            outputs = model(x)

        # Check output structure
        assert isinstance(outputs, dict)
        assert 'logits' in outputs
        assert 'mu' in outputs
        assert 'logvar' in outputs
        assert 'z' in outputs

        # Check shapes
        assert outputs['logits'].shape == (batch_size, seq_len, 1000)
        assert outputs['mu'].shape == (batch_size, 64)
        assert outputs['logvar'].shape == (batch_size, 64)
        assert outputs['z'].shape == (batch_size, 64)

    def test_model_encode(self):
        """Should encode sequences to latent space."""
        config = self.create_basic_config()
        model = TransformerVAE(config)
        model.eval()

        batch_size, seq_len = 3, 15
        x = torch.randint(0, 1000, (batch_size, seq_len))

        with torch.no_grad():
            encoding_results = model.encode(x)

        assert isinstance(encoding_results, dict)
        assert 'mu' in encoding_results
        assert 'logvar' in encoding_results
        assert 'z' in encoding_results

        assert encoding_results['mu'].shape == (batch_size, 64)
        assert encoding_results['logvar'].shape == (batch_size, 64)
        assert encoding_results['z'].shape == (batch_size, 64)

    def test_model_decode_teacher_forcing(self):
        """Should decode with teacher forcing during training."""
        config = self.create_basic_config()
        model = TransformerVAE(config)
        model.train()  # Training mode for teacher forcing

        batch_size, seq_len = 2, 10
        z = torch.randn(batch_size, 64)
        target_sequence = torch.randint(0, 1000, (batch_size, seq_len))

        logits = model.decode(z, target_sequence=target_sequence)

        assert logits.shape == (batch_size, seq_len, 1000)

    def test_model_decode_autoregressive(self):
        """Should decode autoregressively during inference."""
        config = self.create_basic_config()
        model = TransformerVAE(config)
        model.eval()  # Eval mode for autoregressive generation

        batch_size = 2
        z = torch.randn(batch_size, 64)

        with torch.no_grad():
            logits = model.decode(z)

        # Check that we get some output (length may vary due to early stopping)
        assert logits.shape[0] == batch_size
        assert logits.shape[2] == 1000  # vocab_size

    def test_model_sample(self):
        """Should generate samples from the model."""
        config = self.create_basic_config()
        model = TransformerVAE(config)

        num_samples = 5
        samples = model.sample(num_samples)

        assert samples.shape[0] == num_samples
        assert samples.dtype == torch.long
        assert samples.min() >= 0
        assert samples.max() < 1000

    def test_model_with_lengths_masking(self):
        """Should handle variable length sequences with masking."""
        config = self.create_basic_config()
        model = TransformerVAE(config)
        model.eval()

        batch_size, max_seq_len = 3, 20
        x = torch.randint(0, 1000, (batch_size, max_seq_len))
        lengths = torch.tensor([15, 12, 18])

        with torch.no_grad():
            outputs = model(x, lengths=lengths)

        # Should still produce correct shapes
        assert outputs['logits'].shape == (batch_size, max_seq_len, 1000)
        assert outputs['mu'].shape == (batch_size, 64)

    def test_model_compute_loss(self):
        """Should compute VAE loss correctly."""
        config = self.create_basic_config()
        model = TransformerVAE(config)

        batch_size, seq_len = 4, 15
        sequences = torch.randint(0, 1000, (batch_size, seq_len))
        batch = {'sequences': sequences}

        loss_dict = model.compute_loss(batch, beta=0.5)

        assert isinstance(loss_dict, dict)
        assert 'total_loss' in loss_dict
        assert 'recon_loss' in loss_dict
        assert 'kl_loss' in loss_dict
        assert 'beta' in loss_dict

        # Check that losses are positive scalars
        assert loss_dict['total_loss'].ndim == 0
        assert loss_dict['recon_loss'] >= 0
        assert loss_dict['kl_loss'] >= 0
        assert loss_dict['beta'] == 0.5

    def test_model_get_latent_representations(self):
        """Should extract latent representations."""
        config = self.create_basic_config()
        model = TransformerVAE(config)

        batch_size, seq_len = 3, 12
        sequences = torch.randint(0, 1000, (batch_size, seq_len))

        latent_repr = model.get_latent_representations(sequences)

        assert latent_repr.shape == (batch_size, 64)
        assert latent_repr.dtype == torch.float32

    def test_model_gradient_flow(self):
        """Should allow gradient flow through the model."""
        config = self.create_basic_config()
        model = TransformerVAE(config)
        model.train()

        batch_size, seq_len = 2, 10
        x = torch.randint(0, 1000, (batch_size, seq_len))
        x.requires_grad_(False)  # Input tokens don't need gradients

        outputs = model(x)
        loss = outputs['logits'].sum() + outputs['mu'].sum() + outputs['logvar'].sum()
        loss.backward()

        # Check that model parameters have gradients
        param_count = 0
        grad_count = 0
        for param in model.parameters():
            param_count += 1
            if param.grad is not None:
                grad_count += 1

        assert param_count > 0
        assert grad_count > 0

    def test_model_device_compatibility(self):
        """Should work with different devices."""
        config = self.create_basic_config()
        model = TransformerVAE(config)

        # Test CPU
        x_cpu = torch.randint(0, 1000, (2, 10))
        outputs_cpu = model(x_cpu)
        assert outputs_cpu['logits'].device == x_cpu.device

        # Test CUDA if available
        if torch.cuda.is_available():
            model_cuda = model.cuda()
            x_cuda = x_cpu.cuda()
            outputs_cuda = model_cuda(x_cuda)
            assert outputs_cuda['logits'].device == x_cuda.device

    def test_model_different_configurations(self):
        """Should work with different model configurations."""
        # Test with different encoder/decoder architectures
        configs = [
            # Single encoder layer
            ModelConfig(
                model_type="transformer_vae",
                vocab_size=500,
                embedding_dim=64,
                latent_dim=32,
                max_sequence_length=50,
                pad_token_id=0,
                encoder_layers=[
                    LayerConfig(
                        layer_type="pooling",
                        input_dim=64,
                        output_dim=64,
                        dropout=0.1,
                        activation="relu",
                        layer_params={"pooling_type": "mean"}
                    )
                ],
                decoder_layers=[
                    LayerConfig(
                        layer_type="transformer_decoder",
                        input_dim=32 + 64,
                        output_dim=64,
                        dropout=0.1,
                        activation="relu"
                    )
                ],
                latent_layer=LayerConfig(
                    layer_type="latent_sampler",
                    input_dim=64,
                    output_dim=32,
                    dropout=0.1,
                    activation="linear"
                )
            ),
            # Multiple encoder layers
            ModelConfig(
                model_type="transformer_vae",
                vocab_size=2000,
                embedding_dim=256,
                latent_dim=128,
                max_sequence_length=150,
                pad_token_id=0,
                encoder_layers=[
                    LayerConfig(
                        layer_type="transformer_encoder",
                        input_dim=256,
                        output_dim=256,
                        dropout=0.1,
                        activation="gelu",
                        layer_params={"num_heads": 8}
                    ),
                    LayerConfig(
                        layer_type="transformer_encoder",
                        input_dim=256,
                        output_dim=256,
                        dropout=0.1,
                        activation="gelu"
                    ),
                    LayerConfig(
                        layer_type="pooling",
                        input_dim=256,
                        output_dim=256,
                        dropout=0.1,
                        activation="relu",
                        layer_params={"pooling_type": "attention"}
                    )
                ],
                decoder_layers=[
                    LayerConfig(
                        layer_type="transformer_decoder",
                        input_dim=128 + 256,
                        output_dim=256,
                        dropout=0.1,
                        activation="gelu"
                    ),
                    LayerConfig(
                        layer_type="transformer_decoder",
                        input_dim=256,
                        output_dim=256,
                        dropout=0.1,
                        activation="gelu"
                    )
                ],
                latent_layer=LayerConfig(
                    layer_type="latent_sampler",
                    input_dim=256,
                    output_dim=128,
                    dropout=0.1,
                    activation="linear"
                )
            )
        ]

        for config in configs:
            model = TransformerVAE(config)
            x = torch.randint(0, config.vocab_size, (2, 20))

            # Should create and run without errors
            outputs = model(x)
            assert outputs['logits'].shape[2] == config.vocab_size
            assert outputs['mu'].shape[1] == config.latent_dim

    def test_model_stochastic_vs_deterministic(self):
        """Should behave differently in training vs eval modes."""
        config = self.create_basic_config()
        model = TransformerVAE(config)

        batch_size, seq_len = 2, 10
        x = torch.randint(0, 1000, (batch_size, seq_len))

        # Training mode - stochastic
        model.train()
        outputs1 = model(x)
        outputs2 = model(x)

        # mu and logvar should be same (deterministic)
        assert torch.allclose(outputs1['mu'], outputs2['mu'])
        assert torch.allclose(outputs1['logvar'], outputs2['logvar'])

        # z should be different (stochastic)
        assert not torch.allclose(outputs1['z'], outputs2['z'], atol=1e-6)

        # Eval mode - deterministic
        model.eval()
        with torch.no_grad():
            outputs3 = model(x)
            outputs4 = model(x)

        # All outputs should be identical in eval mode
        assert torch.allclose(outputs3['mu'], outputs4['mu'])
        assert torch.allclose(outputs3['logvar'], outputs4['logvar'])
        assert torch.allclose(outputs3['z'], outputs4['z'])

    def test_model_padding_token_handling(self):
        """Should properly handle padding tokens."""
        config = self.create_basic_config()
        model = TransformerVAE(config)

        batch_size, seq_len = 2, 15
        x = torch.randint(1, 1000, (batch_size, seq_len))  # No padding tokens
        x_with_padding = x.clone()
        x_with_padding[:, 10:] = 0  # Add padding

        lengths = torch.tensor([10, 10])  # Actual sequence lengths

        # Test loss computation with padding
        batch_no_pad = {'sequences': x}
        batch_with_pad = {'sequences': x_with_padding, 'lengths': lengths}

        loss_no_pad = model.compute_loss(batch_no_pad)
        loss_with_pad = model.compute_loss(batch_with_pad)

        # Both should compute valid losses
        assert torch.isfinite(loss_no_pad['total_loss'])
        assert torch.isfinite(loss_with_pad['total_loss'])

    def test_model_edge_cases(self):
        """Should handle edge cases gracefully."""
        config = self.create_basic_config()
        model = TransformerVAE(config)
        model.eval()

        # Single sequence
        x_single = torch.randint(0, 1000, (1, 5))
        with torch.no_grad():
            outputs_single = model(x_single)
        assert outputs_single['logits'].shape == (1, 5, 1000)

        # Very short sequence
        x_short = torch.randint(0, 1000, (2, 2))
        with torch.no_grad():
            outputs_short = model(x_short)
        assert outputs_short['logits'].shape == (2, 2, 1000)

        # Batch size of 1
        x_batch1 = torch.randint(0, 1000, (1, 10))
        with torch.no_grad():
            outputs_batch1 = model(x_batch1)
        assert outputs_batch1['mu'].shape == (1, 64)