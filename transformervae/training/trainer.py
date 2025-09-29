"""
Training implementation for TransformerVAE models.

This module provides the main training loop and optimization logic
for VAE training with β-KL divergence loss.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List
import time
from pathlib import Path

from transformervae.config.basic_config import VAETrainingConfig
from transformervae.models.model import TransformerVAE
from .evaluator import TrainingEvaluator
from .callbacks import TrainingCallbacks
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

class VAETrainer:
    """VAE trainer with β-KL divergence loss."""

    def __init__(self, config: VAETrainingConfig, tokenizer=None):
        """
        Initialize VAE trainer.

        Args:
            config: Training configuration
            tokenizer: SMILES tokenizer for molecular accuracy computation (optional)
        """
        self.config = config
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.device = None
        self.tokenizer = tokenizer

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Beta scheduling for KL annealing
        self.current_beta = config.beta
        self.beta_schedule = getattr(config, 'beta_schedule', None)

        # Components
        self.evaluator = TrainingEvaluator()
        self.callbacks = TrainingCallbacks()

        # Metrics tracking
        self.train_metrics = []
        self.val_metrics = []

    @classmethod
    def from_config(cls, config: VAETrainingConfig, tokenizer=None) -> 'VAETrainer':
        """Create trainer from configuration."""
        return cls(config, tokenizer)

    def setup_model(self, model: TransformerVAE) -> None:
        """
        Setup model for training.

        Args:
            model: TransformerVAE model to train
        """
        self.model = model
        self.device = next(model.parameters()).device

        # Setup optimizer
        self._setup_optimizer()

        # Setup learning rate scheduler
        self._setup_scheduler()

    def setup_data(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> None:
        """
        Setup data loaders.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
        """
        self.train_loader = train_loader
        self.val_loader = val_loader

    def _setup_optimizer(self) -> None:
        """Setup optimizer based on configuration."""
        if self.config.optimizer_type == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.config.optimizer_type}")

    def _setup_scheduler(self) -> None:
        """Setup learning rate scheduler."""
        scheduler_config = self.config.scheduler_config

        if scheduler_config.get("type") == "reduce_on_plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=scheduler_config.get("patience", 10),
                factor=scheduler_config.get("factor", 0.5),
                min_lr=scheduler_config.get("min_lr", 1e-6)
            )
        elif scheduler_config.get("type") == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get("step_size", 30),
                gamma=scheduler_config.get("gamma", 0.1)
            )
        elif scheduler_config.get("type") == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=scheduler_config.get("min_lr", 1e-6)
            )

    def train(self) -> Dict[str, List[float]]:
        """
        Run complete training loop.

        Returns:
            Dictionary containing training and validation metrics
        """
        if self.model is None:
            raise ValueError("Model not setup. Call setup_model() first.")
        if self.train_loader is None:
            raise ValueError("Data not setup. Call setup_data() first.")

        print(f"Starting training for {self.config.epochs} epochs")
        print(f"Training on device: {self.device}")

        for epoch in range(self.config.epochs):
            self.current_epoch = epoch

            # Training phase
            train_metrics = self._train_epoch()
            self.train_metrics.append(train_metrics)

            # Validation phase
            if self.val_loader is not None and epoch % self.config.validation_freq == 0:
                val_metrics = self._validate_epoch()
                self.val_metrics.append(val_metrics)

                # Learning rate scheduling (for ReduceLROnPlateau)
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['total_loss'])

                # Save best model
                if val_metrics['total_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['total_loss']
                    self._save_checkpoint('best_model.pt')

            # Other scheduler types
            if self.scheduler is not None and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()

            # Update beta for KL annealing
            self._update_beta(epoch)

            # Checkpoint saving
            if epoch % self.config.checkpoint_freq == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch}.pt')

            # Logging
            self._log_epoch_metrics(epoch, train_metrics,
                                  val_metrics if self.val_loader is not None else None)

        print("Training completed!")

        return {
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics
        }

    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        epoch_metrics = {
            'total_loss': 0.0,
            'recon_loss': 0.0,
            'kl_loss': 0.0,
            'num_batches': 0
        }

        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass with proper VAE structure
            # Encoder processes the input sequence
            # Decoder generates from latent variables with teacher forcing
            outputs = self.model(batch['sequences'], target_sequence=batch['sequences'], training=True)

            # Compute loss
            loss_dict = self._compute_loss(outputs, batch)

            # Backward pass
            self.optimizer.zero_grad()
            loss_dict['total_loss'].backward()

            # Gradient clipping
            if self.config.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm
                )

            self.optimizer.step()
            self.global_step += 1

            # Update metrics
            epoch_metrics['total_loss'] += loss_dict['total_loss'].item()
            epoch_metrics['recon_loss'] += loss_dict['recon_loss'].item()
            epoch_metrics['kl_loss'] += loss_dict['kl_loss'].item()
            epoch_metrics['num_batches'] += 1

        # Average metrics
        for key in ['total_loss', 'recon_loss', 'kl_loss']:
            epoch_metrics[key] /= epoch_metrics['num_batches']

        return epoch_metrics

    def _validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()

        epoch_metrics = {
            'total_loss': 0.0,
            'recon_loss': 0.0,
            'kl_loss': 0.0,
            'recon_accuracy': 0.0,
            'num_batches': 0
        }

        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # Forward pass with teacher forcing during validation for consistent loss computation
                outputs = self.model(batch['sequences'], target_sequence=batch['sequences'], training=False)

                # Compute loss
                loss_dict = self._compute_loss(outputs, batch)

                # Compute reconstruction accuracy
                recon_accuracy = self._compute_reconstruction_accuracy(outputs, batch)

                # Update metrics
                epoch_metrics['total_loss'] += loss_dict['total_loss'].item()
                epoch_metrics['recon_loss'] += loss_dict['recon_loss'].item()
                epoch_metrics['kl_loss'] += loss_dict['kl_loss'].item()
                epoch_metrics['recon_accuracy'] += recon_accuracy
                epoch_metrics['num_batches'] += 1

        # Average metrics
        for key in ['total_loss', 'recon_loss', 'kl_loss', 'recon_accuracy']:
            epoch_metrics[key] /= epoch_metrics['num_batches']

        return epoch_metrics

    def _compute_loss(self, outputs: Dict[str, torch.Tensor],
                     batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute VAE loss components.

        Args:
            outputs: Model outputs
            batch: Input batch

        Returns:
            Dictionary containing loss components
        """
        # Reconstruction loss
        if 'reconstruction' in outputs:
            reconstruction = outputs['reconstruction']
            targets = batch['sequences']

            # Debug prints
            print(f"DEBUG: reconstruction.shape = {reconstruction.shape}")
            print(f"DEBUG: targets.shape = {targets.shape}")

            # Handle different output shapes
            if reconstruction.dim() == 3:  # (batch, seq, vocab) - sequence generation
                # Use cross-entropy for categorical distribution over vocabulary
                recon_loss = nn.functional.cross_entropy(
                    reconstruction.view(-1, reconstruction.size(-1)),
                    targets.view(-1),
                    ignore_index=0,  # Assume 0 is pad token
                    reduction='mean'
                )
            elif reconstruction.dim() == 2 and targets.dim() == 2:  # (batch, seq) - direct sequence matching
                # For sequence-to-sequence with same dimensions
                if reconstruction.shape == targets.shape:
                    recon_loss = nn.functional.mse_loss(reconstruction, targets.float())
                else:
                    # If shapes don't match, it's likely vocab prediction vs token targets
                    # This shouldn't happen with our fixed architecture, but safety check
                    raise ValueError(f"Reconstruction shape {reconstruction.shape} doesn't match target shape {targets.shape}")
            else:  # Other cases - try MSE
                recon_loss = nn.functional.mse_loss(reconstruction, targets.float())
        else:
            recon_loss = torch.tensor(0.0, device=self.device)

        # KL divergence loss
        if 'mu' in outputs and 'logvar' in outputs:
            mu = outputs['mu']
            logvar = outputs['logvar']
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
        else:
            kl_loss = torch.tensor(0.0, device=self.device)

        # Total loss with dynamic β weighting
        total_loss = recon_loss + self.current_beta * kl_loss

        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }

    def _save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, filename)
        print(f"Checkpoint saved: {filename}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Checkpoint loaded from {checkpoint_path}")

    def _update_beta(self, epoch: int) -> None:
        """Update beta value for KL annealing."""
        if self.beta_schedule is None:
            return

        schedule_type = self.beta_schedule.get('type', 'linear')
        start_beta = self.beta_schedule.get('start_beta', 0.01)
        end_beta = self.beta_schedule.get('end_beta', 0.5)
        warmup_epochs = self.beta_schedule.get('warmup_epochs', 20)

        if epoch < warmup_epochs:
            if schedule_type == 'linear':
                # Linear annealing
                progress = epoch / warmup_epochs
                self.current_beta = start_beta + (end_beta - start_beta) * progress
            elif schedule_type == 'cosine':
                # Cosine annealing
                import math
                progress = epoch / warmup_epochs
                self.current_beta = start_beta + (end_beta - start_beta) * (1 - math.cos(progress * math.pi)) / 2
            elif schedule_type == 'cyclical':
                # Cyclical annealing
                import math
                cycle_length = warmup_epochs // 4  # 4 cycles in warmup period
                cycle_progress = (epoch % cycle_length) / cycle_length
                self.current_beta = start_beta + (end_beta - start_beta) * cycle_progress
        else:
            self.current_beta = end_beta

    def _log_epoch_metrics(self, epoch: int, train_metrics: Dict[str, float],
                          val_metrics: Optional[Dict[str, float]] = None) -> None:
        """Log metrics for current epoch."""
        log_str = f"Epoch {epoch:3d}: "
        log_str += f"Train Loss: {train_metrics['total_loss']:.4f} "
        log_str += f"(Recon: {train_metrics['recon_loss']:.4f}, "
        log_str += f"KL: {train_metrics['kl_loss']:.4f}, "
        log_str += f"β: {self.current_beta:.4f})"

        if val_metrics is not None:
            log_str += f" | Val Loss: {val_metrics['total_loss']:.4f} "
            log_str += f"(Recon: {val_metrics['recon_loss']:.4f}, "
            log_str += f"KL: {val_metrics['kl_loss']:.4f}, "
            log_str += f"Mol Acc: {val_metrics['recon_accuracy']:.3f})"

        print(log_str)

    def evaluate_model(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on test set.

        Args:
            test_loader: Test data loader

        Returns:
            Test metrics
        """
        self.model.eval()

        test_metrics = {
            'total_loss': 0.0,
            'recon_loss': 0.0,
            'kl_loss': 0.0,
            'num_batches': 0
        }

        with torch.no_grad():
            for batch in test_loader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # Forward pass with teacher forcing during evaluation for consistent loss computation
                outputs = self.model(batch['sequences'], target_sequence=batch['sequences'], training=False)

                # Compute loss
                loss_dict = self._compute_loss(outputs, batch)

                # Update metrics
                test_metrics['total_loss'] += loss_dict['total_loss'].item()
                test_metrics['recon_loss'] += loss_dict['recon_loss'].item()
                test_metrics['kl_loss'] += loss_dict['kl_loss'].item()
                test_metrics['num_batches'] += 1

        # Average metrics
        for key in ['total_loss', 'recon_loss', 'kl_loss']:
            test_metrics[key] /= test_metrics['num_batches']

        return test_metrics

    def _compute_reconstruction_accuracy(self, outputs: Dict[str, torch.Tensor],
                                       batch: Dict[str, torch.Tensor]) -> float:
        """
        Compute molecular reconstruction accuracy for validation.

        This compares whether the reconstructed molecules are chemically
        equivalent to the original molecules using RDKit.

        Args:
            outputs: Model outputs
            batch: Input batch containing original SMILES

        Returns:
            Molecular reconstruction accuracy as float
        """
        if 'reconstruction' not in outputs or self.tokenizer is None:
            return 0.0

        reconstruction = outputs['reconstruction']

        if reconstruction.dim() != 3:  # Should be (batch, seq, vocab)
            return 0.0

        # Get predicted tokens (argmax over vocabulary)
        predicted_tokens = torch.argmax(reconstruction, dim=-1)  # (batch, seq)

        # Decode predicted tokens to SMILES strings
        try:
            predicted_smiles = self.tokenizer.decode_batch(predicted_tokens, skip_special_tokens=True)
        except Exception:
            return 0.0

        # Get original SMILES from batch
        if 'smiles' not in batch:
            return 0.0

        original_smiles = batch['smiles']

        # Compare molecules
        correct_molecules = 0
        total_molecules = len(original_smiles)

        for orig_smi, pred_smi in zip(original_smiles, predicted_smiles):
            if self._molecules_equivalent(orig_smi, pred_smi, Chem):
                correct_molecules += 1

        accuracy = correct_molecules / total_molecules if total_molecules > 0 else 0.0
        return accuracy

    def _molecules_equivalent(self, smiles1: str, smiles2: str, Chem) -> bool:
        """
        Check if two SMILES represent the same molecule.

        Args:
            smiles1: First SMILES string
            smiles2: Second SMILES string
            Chem: RDKit Chem module

        Returns:
            True if molecules are equivalent, False otherwise
        """
        try:
            # Parse SMILES to molecule objects
            mol1 = Chem.MolFromSmiles(smiles1)
            mol2 = Chem.MolFromSmiles(smiles2)

            # Check if both SMILES are valid
            if mol1 is None or mol2 is None:
                return False

            # Compare canonical SMILES (most reliable method)
            canonical1 = Chem.MolToSmiles(mol1, canonical=True)
            canonical2 = Chem.MolToSmiles(mol2, canonical=True)

            return canonical1 == canonical2

        except Exception:
            return False

    def _compute_token_accuracy(self, outputs: Dict[str, torch.Tensor],
                               batch: Dict[str, torch.Tensor]) -> float:
        """
        Fallback token-level accuracy computation.
        """
        reconstruction = outputs['reconstruction']
        targets = batch['sequences']

        if reconstruction.dim() == 3:  # (batch, seq, vocab)
            predicted_tokens = torch.argmax(reconstruction, dim=-1)
            mask = (targets != 0).float()
            correct_predictions = (predicted_tokens == targets).float() * mask

            if mask.sum() > 0:
                accuracy = correct_predictions.sum() / mask.sum()
            else:
                accuracy = 0.0

            return accuracy.item()

        return 0.0