"""
Main model implementations for TransformerVAE.

This module provides the complete TransformerVAE model architecture that can be
instantiated from configuration files.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List, Union
from transformervae.config.basic_config import DetailedModelArchitecture, LayerConfig
from transformervae.models.layer import create_layer, LayerInterface


class ModelInterface(ABC, nn.Module):
    """Abstract base class for all model types."""

    def __init__(self, config: DetailedModelArchitecture):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, x: torch.Tensor, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        pass

    @abstractmethod
    def encode(self, x: torch.Tensor, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """Encode input to latent space."""
        pass

    @abstractmethod
    def decode(self, z: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Decode from latent space to output."""
        pass

    @abstractmethod
    def sample(self, num_samples: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Generate samples from the model."""
        pass

    @property
    def latent_dim(self) -> int:
        """Latent space dimension."""
        return self.config.sampler[-1].output_dim

    @property
    def vocab_size(self) -> int:
        """Vocabulary size for sequence models."""
        # Default vocab size, can be overridden
        return 1000


class TransformerVAE(ModelInterface):
    """TransformerVAE model implementation."""

    def __init__(self, config: DetailedModelArchitecture):
        super().__init__(config)

        # Build encoder layers
        self.encoder_layers = nn.ModuleList()
        for layer_config in config.encoder:
            layer = create_layer(layer_config)
            self.encoder_layers.append(layer)

        # Positional encoding for encoder (as per paper)
        # Should be applied after embedding, so use encoder[0].output_dim (embedding dim)
        embed_dim = config.encoder[0].output_dim
        self.encoder_positional_encoding = PositionalEncoding(d_model=embed_dim, dropout=0.0)

        # Latent sampler
        self.sampler = create_layer(config.sampler[0])  # Assume single sampler layer

        # Build decoder layers (these will be used for sequential decoding)
        self.decoder_layers = nn.ModuleList()
        for layer_config in config.decoder:
            layer = create_layer(layer_config)
            self.decoder_layers.append(layer)

        # Embedding layer for decoder input tokens
        vocab_size = config.decoder[-1].output_dim
        embed_dim = config.decoder[0].input_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Positional encoding for decoder (as per paper)
        self.decoder_positional_encoding = PositionalEncoding(d_model=embed_dim, dropout=0.0)

        # Latent projection layer to match embedding dimension
        latent_dim = config.sampler[0].output_dim
        self.latent_projection = nn.Linear(latent_dim, embed_dim)

        # Optional property prediction heads
        self.regression_head = None
        if config.latent_regression_head:
            self.regression_head = nn.ModuleList()
            for layer_config in config.latent_regression_head:
                layer = create_layer(layer_config)
                self.regression_head.append(layer)

        self.classification_head = None
        if config.latent_classification_head:
            self.classification_head = nn.ModuleList()
            for layer_config in config.latent_classification_head:
                layer = create_layer(layer_config)
                self.classification_head.append(layer)

        # Initialize weights
        self._init_weights()

    @classmethod
    def from_config(cls, config: DetailedModelArchitecture) -> "TransformerVAE":
        """Create model from configuration."""
        return cls(config)

    def _init_weights(self):
        """Initialize model weights."""
        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, target_sequence: Optional[torch.Tensor] = None,
                training: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete model.

        Args:
            x: Input tensor for encoding
            target_sequence: Target sequence for teacher forcing (training only)
            training: Whether in training mode (teacher forcing) or inference mode

        Returns:
            Dictionary containing reconstruction and latent info
        """
        # Encode to latent space
        mu, logvar = self.encode(x)

        # Sample from latent distribution (reparameterization trick)
        z = self._reparameterize(mu, logvar)

        if training and target_sequence is not None:
            # Training mode: use teacher forcing with sampled latent variables
            reconstruction = self.decode_teacher_forcing(z, target_sequence)
        else:
            # Inference mode: autoregressive generation
            reconstruction = self.decode_autoregressive(z, max_length=x.size(1))

        outputs = {
            'reconstruction': reconstruction,
            'mu': mu,
            'logvar': logvar,
            'z': z
        }

        # Add property predictions if heads exist
        if self.regression_head is not None:
            z = mu  # Use mean for property prediction
            for layer in self.regression_head:
                z = layer(z)
            outputs['property_regression'] = z

        if self.classification_head is not None:
            z = mu  # Use mean for property prediction
            for layer in self.classification_head:
                z = layer(z)
            outputs['property_classification'] = z

        return outputs

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent space.

        Args:
            x: Input tensor (batch_size, seq_len) - token indices

        Returns:
            Tuple of (mu, logvar) tensors
        """
        # Pass through encoder layers
        # First layer is embedding: (batch, seq_len) -> (batch, seq_len, embed_dim)
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x)

            # Add positional encoding after embedding layer (as per paper)
            if i == 0:  # After embedding layer
                x = self.encoder_positional_encoding(x)

        # x is now the memory output from encoder
        # Note: If config has pooling layer, it's already applied in encoder_layers
        # Otherwise, we need to pool here as described in paper

        # Sample latent variables using sampler
        mu, logvar, z = self.sampler(x)

        return mu, logvar

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE latent sampling.

        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution

        Returns:
            Sampled latent variables
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # During inference, use mean
            return mu

    def decode_teacher_forcing(self, z: torch.Tensor, target_sequence: torch.Tensor) -> torch.Tensor:
        """
        Decode using teacher forcing for training.

        Args:
            z: Latent codes (batch_size, latent_dim)
            target_sequence: Target sequence for teacher forcing (batch_size, seq_len)

        Returns:
            Decoded logits tensor (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = target_sequence.shape
        device = z.device

        # Create shifted input sequence: [SOS, token_0, token_1, ..., token_{n-2}]
        # This is what the model sees when predicting [token_0, token_1, ..., token_{n-1}]
        sos_token = torch.ones(batch_size, 1, dtype=torch.long, device=device)  # SOS = 1
        decoder_input = torch.cat([sos_token, target_sequence[:, :-1]], dim=1)  # (batch, seq_len)

        # Get embeddings for the entire sequence at once
        embeddings = self.embedding(decoder_input)  # (batch, seq_len, embed_dim)

        # Add positional encoding (as per paper)
        embeddings = self.decoder_positional_encoding(embeddings)

        # Project latent and add to all positions (broadcast)
        # As per paper: "the latent representation is added to the embedding of each token during decoding"
        latent_features = self.latent_projection(z)  # (batch, embed_dim)
        embeddings = embeddings + latent_features.unsqueeze(1)  # (batch, seq_len, embed_dim)

        # Pass through decoder layers - processes entire sequence in parallel
        x = embeddings
        for layer in self.decoder_layers:
            x = layer(x)

        # x is now (batch, seq_len, vocab_size) - logits for each position
        return x

    def decode_autoregressive(self, z: torch.Tensor, max_length: int,
                            use_beam_search: bool = False, beam_size: int = 5) -> torch.Tensor:
        """
        Decode autoregressively for inference.

        Args:
            z: Latent codes (batch_size, latent_dim)
            max_length: Maximum sequence length to generate
            use_beam_search: Whether to use beam search
            beam_size: Beam size for beam search

        Returns:
            Generated sequences (batch_size, seq_len, vocab_size) or (batch_size, seq_len) for beam search
        """
        if use_beam_search:
            return self._beam_search_decode(z, max_length, beam_size)
        else:
            return self._greedy_decode(z, max_length)

    def _greedy_decode(self, z: torch.Tensor, max_length: int) -> torch.Tensor:
        """Greedy autoregressive decoding."""
        batch_size = z.size(0)
        device = z.device

        # Initialize with SOS token
        sequences = torch.ones(batch_size, 1, dtype=torch.long, device=device)  # SOS = 1
        logits_list = []

        # Pre-compute latent projection (will be added to ALL positions as per paper)
        latent_features = self.latent_projection(z)  # (batch, embed_dim)

        for step in range(max_length):
            # Get embeddings for current sequence
            embeddings = self.embedding(sequences)

            # Add positional encoding (as per paper)
            embeddings = self.decoder_positional_encoding(embeddings)

            # Add latent variables to ALL positions (as per paper)
            # "the latent representation is added to the embedding of each token during decoding"
            embeddings = embeddings + latent_features.unsqueeze(1)  # (batch, seq_len, embed_dim)

            # Pass through decoder layers
            x = embeddings
            for layer in self.decoder_layers:
                x = layer(x)

            # Get logits for next token (last position)
            next_token_logits = x[:, -1, :]  # (batch_size, vocab_size)
            logits_list.append(next_token_logits.unsqueeze(1))

            # Get next token (greedy)
            next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Stop if all sequences generated EOS token (assuming EOS = 2)
            if (next_tokens.squeeze(-1) == 2).all():
                break

            # Append next token to sequences
            sequences = torch.cat([sequences, next_tokens], dim=1)

        # Return logits for all positions
        return torch.cat(logits_list, dim=1) if logits_list else torch.zeros(batch_size, 0, self.config.decoder[-1].output_dim, device=device)

    def _beam_search_decode(self, z: torch.Tensor, max_length: int, beam_size: int) -> torch.Tensor:
        """Beam search decoding for better generation quality."""
        batch_size = z.size(0)
        device = z.device
        vocab_size = self.config.decoder[-1].output_dim

        # Initialize beams
        # Each beam: (sequence, log_prob)
        sequences = torch.ones(batch_size, beam_size, 1, dtype=torch.long, device=device)  # SOS
        log_probs = torch.zeros(batch_size, beam_size, device=device)

        # Pre-compute latent features for all beams
        z_expanded = z.unsqueeze(1).expand(-1, beam_size, -1).reshape(batch_size * beam_size, -1)
        latent_features = self.latent_projection(z_expanded)  # (batch*beam, embed_dim)

        finished_sequences = []

        for step in range(max_length):
            sequences_flat = sequences.reshape(batch_size * beam_size, -1)

            # Get embeddings
            embeddings = self.embedding(sequences_flat)

            # Add positional encoding (as per paper)
            embeddings = self.decoder_positional_encoding(embeddings)

            # Add latent variables to ALL positions (as per paper)
            # "the latent representation is added to the embedding of each token during decoding"
            embeddings = embeddings + latent_features.unsqueeze(1)  # (batch*beam, seq_len, embed_dim)

            # Pass through decoder
            x = embeddings
            for layer in self.decoder_layers:
                x = layer(x)

            # Get next token logits
            next_token_logits = x[:, -1, :]  # (batch_size * beam_size, vocab_size)
            next_token_log_probs = torch.log_softmax(next_token_logits, dim=-1)

            # Reshape back to beam format
            next_token_log_probs = next_token_log_probs.reshape(batch_size, beam_size, vocab_size)

            # Compute new scores
            new_log_probs = log_probs.unsqueeze(-1) + next_token_log_probs  # (batch_size, beam_size, vocab_size)
            new_log_probs = new_log_probs.reshape(batch_size, -1)  # (batch_size, beam_size * vocab_size)

            # Select top beam_size candidates
            top_log_probs, top_indices = torch.topk(new_log_probs, beam_size, dim=-1)

            # Convert indices back to beam and token indices
            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size

            # Update sequences and probabilities
            new_sequences = []
            new_log_probs_list = []

            for b in range(batch_size):
                batch_sequences = []
                batch_log_probs = []

                for i in range(beam_size):
                    beam_idx = beam_indices[b, i]
                    token_idx = token_indices[b, i]

                    old_sequence = sequences[b, beam_idx]
                    new_sequence = torch.cat([old_sequence, token_idx.unsqueeze(0)])

                    batch_sequences.append(new_sequence)
                    batch_log_probs.append(top_log_probs[b, i])

                # Pad sequences to same length
                max_seq_len = max(len(seq) for seq in batch_sequences)
                padded_sequences = []
                for seq in batch_sequences:
                    if len(seq) < max_seq_len:
                        padded_seq = torch.cat([seq, torch.zeros(max_seq_len - len(seq), dtype=seq.dtype, device=device)])
                    else:
                        padded_seq = seq
                    padded_sequences.append(padded_seq)

                new_sequences.append(torch.stack(padded_sequences))
                new_log_probs_list.append(torch.stack(batch_log_probs))

            sequences = torch.stack(new_sequences)
            log_probs = torch.stack(new_log_probs_list)

            # Check for EOS tokens and move finished sequences
            # For simplicity, we continue until max_length

        # Return best sequence for each batch (convert to logits format for compatibility)
        best_sequences = sequences[:, 0, 1:]  # Remove SOS token, take best beam

        # Convert back to logits format (one-hot encoding)
        seq_length = best_sequences.size(1)
        logits = torch.zeros(batch_size, seq_length, vocab_size, device=device)
        for b in range(batch_size):
            for t in range(seq_length):
                if best_sequences[b, t] < vocab_size:
                    logits[b, t, best_sequences[b, t]] = 1.0

        return logits

    def decode(self, z: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Concrete implementation of abstract decode method.
        Routes to autoregressive decoding by default.

        Args:
            z: Latent codes
            *args, **kwargs: Additional arguments

        Returns:
            Decoded output tensor
        """
        max_length = kwargs.get('max_length', 100)
        return self.decode_autoregressive(z, max_length=max_length, use_beam_search=False)

    def sample(self, num_samples: int, device: str = "cpu") -> torch.Tensor:
        """
        Generate samples from the model.

        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on

        Returns:
            Generated samples
        """
        self.eval()

        with torch.no_grad():
            # Sample from prior distribution
            z = torch.randn(num_samples, self.latent_dim, device=device)

            # Decode samples using autoregressive generation
            samples = self.decode_autoregressive(z, max_length=100, use_beam_search=False)

            return samples



class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings."""
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


# Model Factory System

class ModelFactory:
    """Factory for creating models from configuration."""

    _instance = None
    _registry = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._register_builtin_models()

    def _register_builtin_models(self):
        """Register all built-in model types."""
        self._registry = {
            "transformer_vae": TransformerVAE,
        }

    def register_model(self, model_type: str, model_class: type):
        """Register a new model type."""
        if model_type in self._registry:
            raise ValueError(f"Model type '{model_type}' is already registered")

        if not issubclass(model_class, ModelInterface):
            raise TypeError("Model class must implement ModelInterface")

        self._registry[model_type] = model_class

    def create_model(self, config: DetailedModelArchitecture) -> ModelInterface:
        """Create a model from configuration."""
        # For now, always create TransformerVAE
        return TransformerVAE(config)

    @property
    def registered_models(self) -> Dict[str, type]:
        """Get dictionary of registered model types."""
        return self._registry.copy()


# Global factory instance
_model_factory = ModelFactory()


def register_model(model_type: str):
    """Decorator for registering model types."""
    def decorator(model_class: type):
        _model_factory.register_model(model_type, model_class)
        return model_class
    return decorator


def create_model(config: DetailedModelArchitecture) -> ModelInterface:
    """Create a model from configuration using the global factory."""
    return _model_factory.create_model(config)


def get_registered_models() -> List[str]:
    """Get list of registered model types."""
    return list(_model_factory.registered_models.keys())