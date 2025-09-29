"""
Layer implementations for TransformerVAE.

This module provides all the neural network layer implementations including
transformer components, VAE sampling, pooling, and prediction heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
from transformervae.config.basic_config import LayerConfig


class LayerInterface(ABC, nn.Module):
    """Abstract base class for all layer types."""

    def __init__(self, config: LayerConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass through the layer."""
        pass

    @property
    def input_dim(self) -> int:
        """Input dimension of the layer."""
        return self.config.input_dim

    @property
    def output_dim(self) -> int:
        """Output dimension of the layer."""
        return self.config.output_dim

class EmbeddingLayer(LayerInterface):
    """Embedding layer that maps token ids to dense vectors."""
    def __init__(self, config: LayerConfig):
        super().__init__(config)
        # Interpret input_dim as vocab_size; output_dim as embedding_dim
        self.embedding = nn.Embedding(num_embeddings=config.input_dim, embedding_dim=config.output_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq) long -> (batch, seq, emb_dim) float
        x = self.embedding(x)
        return self.dropout(x)

# class TransformerEncoderLayer(LayerInterface):
    # """Custom Transformer encoder layer."""

    # def __init__(self, config: LayerConfig):
    #     super().__init__(config)

    #     # Get layer-specific parameters
    #     self.num_heads = config.layer_params.get("num_heads", 8)
    #     self.dim_feedforward = config.layer_params.get("dim_feedforward", 2048)
    #     self.num_layers = config.layer_params.get("num_layers", 1)

    #     # Input projection if needed
    #     if config.input_dim != config.output_dim:
    #         self.input_projection = nn.Linear(config.input_dim, config.output_dim)
    #     else:
    #         self.input_projection = None

    #     # Transformer encoder layers
    #     encoder_layer = nn.TransformerEncoderLayer(
    #         d_model=config.output_dim,
    #         nhead=self.num_heads,
    #         dim_feedforward=self.dim_feedforward,
    #         dropout=config.dropout,
    #         activation=self._get_activation(config.activation),
    #         batch_first=True
    #     )

    #     self.transformer_encoder = nn.TransformerEncoder(
    #         encoder_layer,
    #         num_layers=self.num_layers
    #     )

    #     self.dropout = nn.Dropout(config.dropout)

    # def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    #     """
    #     Forward pass through transformer encoder.

    #     Args:
    #         x: Input tensor of shape (batch_size, seq_len, input_dim)
    #         mask: Optional attention mask

    #     Returns:
    #         Encoded tensor of shape (batch_size, seq_len, output_dim)
    #     """
    #     # Project input if dimensions don't match
    #     if self.input_projection is not None:
    #         x = self.input_projection(x)

    #     # Apply transformer encoder
    #     x = self.transformer_encoder(x, src_key_padding_mask=mask)

    #     return self.dropout(x)

    # def _get_activation(self, activation: str):
    #     """Convert activation string to PyTorch activation function."""
    #     activations = {
    #         "relu": "relu",
    #         "gelu": "gelu",
    #         "tanh": "tanh",
    #         "sigmoid": "sigmoid"
    #     }
    #     return activations.get(activation, "relu")

class PreLNTransformerEncoderLayer(nn.Module):
    """Transformer Encoder Layer with Pre-LN structure."""
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, 
                 activation="relu", batch_first=True):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.batch_first = batch_first
        
        # Multi-head attention
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Pre-LN: LayerNorm在attention和FF之前
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout for residual connections
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Activation
        self.activation = self._get_activation_fn(activation)
    
    def _get_activation_fn(self, activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu
        elif activation == "tanh":
            return torch.tanh
        elif activation == "sigmoid":
            return torch.sigmoid
        else:
            return F.relu
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Pre-LN for self-attention
        src_norm = self.norm1(src)
        src2, _ = self.self_attn(
            src_norm, src_norm, src_norm,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        src = src + self.dropout1(src2)  # Residual connection
        
        # Pre-LN for feedforward
        src_norm = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src_norm))))
        src = src + self.dropout2(src2)  # Residual connection
        
        return src


class TransformerEncoderLayer(LayerInterface):
    """Custom Transformer encoder layer with Pre-LN structure."""

    def __init__(self, config: LayerConfig):
        super().__init__(config)

        # Get layer-specific parameters
        self.num_heads = config.layer_params.get("num_heads", 8)
        self.dim_feedforward = config.layer_params.get("dim_feedforward", 2048)
        self.num_layers = config.layer_params.get("num_layers", 1)

        # Input projection if needed
        if config.input_dim != config.output_dim:
            self.input_projection = nn.Linear(config.input_dim, config.output_dim)
        else:
            self.input_projection = None

        # 使用Pre-LN encoder layers
        self.encoder_layers = nn.ModuleList([
            PreLNTransformerEncoderLayer(
                d_model=config.output_dim,
                nhead=self.num_heads,
                dim_feedforward=self.dim_feedforward,
                dropout=config.dropout,
                activation=config.activation,
                batch_first=True
            ) for _ in range(self.num_layers)
        ])

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through transformer encoder with Pre-LN structure.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional attention mask

        Returns:
            Encoded tensor of shape (batch_size, seq_len, output_dim)
        """
        # Project input if dimensions don't match
        if self.input_projection is not None:
            x = self.input_projection(x)

        # Apply Pre-LN transformer encoder layers
        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=mask)

        return self.dropout(x)


class TransformerDecoderLayer(LayerInterface):
    """Custom Transformer decoder layer."""

    def __init__(self, config: LayerConfig):
        super().__init__(config)

        # Get layer-specific parameters
        self.num_heads = config.layer_params.get("num_heads", 8)
        self.dim_feedforward = config.layer_params.get("dim_feedforward", 2048)
        self.num_layers = config.layer_params.get("num_layers", 1)

        # Input projection if needed
        if config.input_dim != config.output_dim:
            self.input_projection = nn.Linear(config.input_dim, config.output_dim)
        else:
            self.input_projection = None

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.output_dim,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=config.dropout,
            activation=self._get_activation(config.activation),
            batch_first=True
        )

        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=self.num_layers
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, memory: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through transformer decoder.

        Args:
            x: Target tensor of shape (batch_size, tgt_len, input_dim)
            memory: Memory tensor from encoder (batch_size, src_len, d_model)
            tgt_mask: Target attention mask
            memory_mask: Memory attention mask

        Returns:
            Decoded tensor of shape (batch_size, tgt_len, output_dim)
        """
        # Project input if dimensions don't match
        if self.input_projection is not None:
            x = self.input_projection(x)

        # If no memory provided, use self-attention only
        if memory is None:
            memory = x

        # Apply transformer decoder
        x = self.transformer_decoder(
            x, memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask
        )

        return self.dropout(x)

    def _get_activation(self, activation: str):
        """Convert activation string to PyTorch activation function."""
        activations = {
            "relu": "relu",
            "gelu": "gelu",
            "tanh": "tanh",
            "sigmoid": "sigmoid"
        }
        return activations.get(activation, "relu")


class LatentSampler(LayerInterface):
    """VAE reparameterization layer for sampling latent variables."""

    def __init__(self, config: LayerConfig):
        super().__init__(config)

        # Linear layers for mean and log variance
        self.mu_layer = nn.Linear(config.input_dim, config.output_dim)
        self.logvar_layer = nn.Linear(config.input_dim, config.output_dim)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with reparameterization trick.

        Args:
            x: Input tensor of shape (batch_size, input_dim)
            deterministic: If True, return mu instead of sampling

        Returns:
            Tuple of (mu, logvar, z) tensors
        """
        x = self.dropout(x)

        # Compute mean and log variance
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)

        if deterministic or not self.training:
            # In deterministic mode, return mean
            z = mu
        else:
            # Reparameterization trick: z = mu + sigma * epsilon
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + std * eps

        return mu, logvar, z


class PoolingLayer(LayerInterface):
    """Pooling layer with multiple pooling strategies."""

    def __init__(self, config: LayerConfig):
        super().__init__(config)

        self.pooling_type = config.layer_params.get("pooling_type", "mean")

        if self.pooling_type == "attention":
            # Attention-based pooling
            self.attention = nn.MultiheadAttention(
                embed_dim=config.input_dim,
                num_heads=config.layer_params.get("num_heads", 1),
                dropout=config.dropout,
                batch_first=True
            )
            self.query = nn.Parameter(torch.randn(1, 1, config.input_dim))

        # Output projection if needed
        if config.input_dim != config.output_dim:
            self.output_projection = nn.Linear(config.input_dim, config.output_dim)
        else:
            self.output_projection = None

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through pooling layer.

        Args:
            x: Input tensor of shape (batch_size, seq_len, features)
            mask: Optional mask for valid positions

        Returns:
            Pooled tensor of shape (batch_size, features)
        """
        if self.pooling_type == "mean":
            if mask is not None:
                # Masked mean pooling
                mask_expanded = mask.unsqueeze(-1).float()
                x_masked = x * mask_expanded
                pooled = x_masked.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
            else:
                pooled = x.mean(dim=1)

        elif self.pooling_type == "max":
            if mask is not None:
                # Masked max pooling
                x_masked = x.masked_fill(~mask.unsqueeze(-1), float('-inf'))
                pooled = x_masked.max(dim=1)[0]
            else:
                pooled = x.max(dim=1)[0]

        elif self.pooling_type == "attention":
            # Attention-based pooling
            batch_size = x.size(0)
            query = self.query.expand(batch_size, -1, -1)

            pooled, _ = self.attention(
                query, x, x,
                key_padding_mask=~mask if mask is not None else None
            )
            pooled = pooled.squeeze(1)  # Remove query dimension

        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")

        # Apply output projection if needed
        if self.output_projection is not None:
            pooled = self.output_projection(pooled)

        return self.dropout(pooled)


class RegressionHead(LayerInterface):
    """Regression head for molecular property prediction."""

    def __init__(self, config: LayerConfig):
        super().__init__(config)

        # Get hidden dimensions
        hidden_dims = config.layer_params.get("hidden_dims", [])

        # Build MLP layers
        layers = []
        prev_dim = config.input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self._get_activation_layer(config.activation))
            layers.append(nn.Dropout(config.dropout))
            prev_dim = hidden_dim

        # Final output layer
        layers.append(nn.Linear(prev_dim, config.output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for regression prediction.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Regression predictions of shape (batch_size, output_dim)
        """
        return self.mlp(x)

    def _get_activation_layer(self, activation: str):
        """Get activation layer from string."""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(),
        }
        return activations.get(activation, nn.ReLU())


class ClassificationHead(LayerInterface):
    """Classification head for molecular class prediction."""

    def __init__(self, config: LayerConfig):
        super().__init__(config)

        # Get hidden dimensions
        hidden_dims = config.layer_params.get("hidden_dims", [])

        # Build MLP layers
        layers = []
        prev_dim = config.input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self._get_activation_layer(config.activation))
            layers.append(nn.Dropout(config.dropout))
            prev_dim = hidden_dim

        # Final output layer (logits)
        layers.append(nn.Linear(prev_dim, config.output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for classification prediction.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Classification logits of shape (batch_size, output_dim)
        """
        return self.mlp(x)

    def _get_activation_layer(self, activation: str):
        """Get activation layer from string."""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(),
        }
        return activations.get(activation, nn.ReLU())


class LinearLayer(LayerInterface):
    """Simple linear layer."""

    def __init__(self, config: LayerConfig):
        super().__init__(config)

        self.linear = nn.Linear(config.input_dim, config.output_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = self._get_activation_layer(config.activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through linear layer."""
        x = self.linear(x)
        if self.config.activation != "linear":
            x = self.activation(x)
        return self.dropout(x)

    def _get_activation_layer(self, activation: str):
        """Get activation layer from string."""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(),
            "linear": nn.Identity(),
        }
        return activations.get(activation, nn.ReLU())


# Layer Factory System

class LayerFactory:
    """Factory for creating layers from configuration."""

    _instance = None
    _registry = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._register_builtin_layers()

    def _register_builtin_layers(self):
        """Register all built-in layer types."""
        self._registry = {
            "embedding": EmbeddingLayer,
            "transformer_encoder": TransformerEncoderLayer,
            "transformer_decoder": TransformerDecoderLayer,
            "latent_sampler": LatentSampler,
            "pooling": PoolingLayer,
            "regression_head": RegressionHead,
            "classification_head": ClassificationHead,
            "linear": LinearLayer,
        }

    def register_layer(self, layer_type: str, layer_class: type):
        """Register a new layer type."""
        if layer_type in self._registry:
            raise ValueError(f"Layer type '{layer_type}' is already registered")

        if not issubclass(layer_class, LayerInterface):
            raise TypeError("Layer class must implement LayerInterface")

        self._registry[layer_type] = layer_class

    def create_layer(self, config: LayerConfig) -> LayerInterface:
        """Create a layer from configuration."""
        if not config.layer_type:
            raise ValueError("Layer type cannot be empty or None")

        if config.layer_type not in self._registry:
            raise ValueError(f"Unknown layer type: {config.layer_type}. "
                           f"Available types: {sorted(self._registry.keys())}")

        layer_class = self._registry[config.layer_type]
        return layer_class(config)

    @property
    def registered_layers(self) -> Dict[str, type]:
        """Get dictionary of registered layer types."""
        return self._registry.copy()


# Global factory instance
_factory = LayerFactory()


def register_layer(layer_type: str):
    """Decorator for registering layer types."""
    def decorator(layer_class: type):
        _factory.register_layer(layer_type, layer_class)
        return layer_class
    return decorator


def create_layer(config: LayerConfig) -> LayerInterface:
    """Create a layer from configuration using the global factory."""
    return _factory.create_layer(config)


def get_registered_layers() -> List[str]:
    """Get list of registered layer types."""
    return list(_factory.registered_layers.keys())

