#!/usr/bin/env python3
"""
Main training script for TransformerVAE.

This script provides a command-line interface for training TransformerVAE models
with various configurations and datasets.
"""

import argparse
import os
import sys
import torch
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from transformervae.config.basic_config import (
    load_model_config, load_training_config,
    DetailedModelArchitecture, VAETrainingConfig
)
from transformervae.models.model import TransformerVAE
from transformervae.training.trainer import VAETrainer
from transformervae.training.callbacks import TrainingCallbacks
from transformervae.data.tokenizer import SMILESTokenizer
from transformervae.data.dataset import MolecularDataset, collate_molecular_batch
from transformervae.utils.reproducibility import set_random_seeds
from transformervae.utils.metrics import compute_molecular_metrics

from torch.utils.data import DataLoader


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train TransformerVAE for molecular generation"
    )

    # Configuration files
    parser.add_argument(
        "--model_config",
        type=str,
        default="transformervae/config/model_configs/base_transformer.yaml",
        help="Path to model configuration file"
    )
    parser.add_argument(
        "--training_config",
        type=str,
        default="transformervae/config/training_configs/moses_config.yaml",
        help="Path to training configuration file"
    )

    # Data and output
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Override data path from config"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory for models and logs"
    )

    # Training modes
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Resume training from checkpoint"
    )
    parser.add_argument(
        "--evaluate_only",
        action="store_true",
        help="Only evaluate, don't train"
    )
    parser.add_argument(
        "--generate_samples",
        type=int,
        default=0,
        help="Generate N samples after training"
    )

    # Device and logging
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (cuda, cpu, or auto)"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed override"
    )

    # Debug options
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (smaller dataset, fewer epochs)"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiling"
    )

    return parser.parse_args()


def setup_device(device_arg: str) -> torch.device:
    """Setup compute device."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            print("Using CPU")
    else:
        device = torch.device(device_arg)
        print(f"Using device: {device}")

    return device


def load_configurations(args: argparse.Namespace) -> tuple[DetailedModelArchitecture, VAETrainingConfig]:
    """Load model and training configurations."""
    print("Loading configurations...")

    # Load model configuration
    if not os.path.exists(args.model_config):
        raise FileNotFoundError(f"Model config not found: {args.model_config}")

    model_config = load_model_config(args.model_config)
    print(f"Loaded model config from: {args.model_config}")

    # Load training configuration
    if not os.path.exists(args.training_config):
        raise FileNotFoundError(f"Training config not found: {args.training_config}")

    training_config = load_training_config(args.training_config)
    print(f"Loaded training config from: {args.training_config}")

    # Override configurations with command line arguments
    if args.data_path:
        if training_config.dataset_config:
            training_config.dataset_config['data_path'] = args.data_path
        else:
            print("Warning: No dataset config found, ignoring --data_path")

    if args.seed is not None:
        training_config.random_seed = args.seed

    # Debug mode adjustments
    if args.debug:
        print("Debug mode: Reducing epochs and batch size")
        training_config.epochs = min(training_config.epochs, 5)
        training_config.batch_size = min(training_config.batch_size, 16)
        training_config.validation_freq = 1
        training_config.checkpoint_freq = 2

    return model_config, training_config


def setup_data(training_config: VAETrainingConfig, debug: bool = False) -> tuple[SMILESTokenizer, DataLoader, DataLoader, DataLoader]:
    """Setup data loaders and tokenizer."""
    print("Setting up data...")

    if not training_config.dataset_config:
        raise ValueError("No dataset configuration found in training config")

    dataset_config = training_config.dataset_config

    # Initialize tokenizer
    tokenizer = SMILESTokenizer(
        vocab_size=dataset_config.get('vocab_size', 1000),
        max_length=dataset_config.get('max_sequence_length', 100)
    )

    # Load dataset
    dataset = MolecularDataset.from_config(dataset_config, tokenizer)

    # Limit dataset size in debug mode
    if debug:
        print("Debug mode: Using small dataset subset (1000 samples)")
        dataset.smiles_data = dataset.smiles_data[:1000]

    # Build vocabulary from dataset
    print("Building vocabulary...")
    vocab_size = min(len(dataset.smiles_data), 10000)  # Use subset for vocab building
    tokenizer.build_vocab(dataset.smiles_data[:vocab_size])
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")

    # Split dataset
    train_split = dataset_config.get('train_split', 0.8)
    val_split = dataset_config.get('val_split', 0.1)
    test_split = dataset_config.get('test_split', 0.1)

    train_dataset, val_dataset, test_dataset = dataset.split_dataset(
        train_split, val_split, test_split
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        collate_fn=collate_molecular_batch,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        collate_fn=collate_molecular_batch,
        num_workers=2,
        pin_memory=True
    ) if len(val_dataset) > 0 else None

    test_loader = DataLoader(
        test_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        collate_fn=collate_molecular_batch,
        num_workers=2,
        pin_memory=True
    ) if len(test_dataset) > 0 else None

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset) if val_dataset else 0}")
    print(f"Test samples: {len(test_dataset) if test_dataset else 0}")

    return tokenizer, train_loader, val_loader, test_loader


def setup_model_and_trainer(model_config: DetailedModelArchitecture,
                           training_config: VAETrainingConfig,
                           device: torch.device,
                           tokenizer: SMILESTokenizer) -> tuple[TransformerVAE, VAETrainer]:
    """Setup model and trainer."""
    print("Setting up model...")

    # Update model config to match tokenizer vocab size
    actual_vocab_size = tokenizer.get_vocab_size()
    print(f"Updating model config to use vocab_size={actual_vocab_size}")

    # Update encoder embedding input_dim
    model_config.encoder[0].input_dim = actual_vocab_size

    # Update decoder final layer output_dim
    model_config.decoder[-1].output_dim = actual_vocab_size

    # Create model
    model = TransformerVAE.from_config(model_config)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model created with {total_params:,} total parameters")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create trainer with tokenizer for molecular accuracy computation
    trainer = VAETrainer.from_config(training_config, tokenizer=tokenizer)
    trainer.setup_model(model)

    return model, trainer


def train_model(trainer: VAETrainer, train_loader: DataLoader,
               val_loader: Optional[DataLoader], callbacks: TrainingCallbacks) -> dict:
    """Train the model."""
    print("Starting training...")

    # Setup data
    trainer.setup_data(train_loader, val_loader)

    # Training callbacks
    callbacks.on_training_start(trainer.config.__dict__)

    # Run training
    try:
        training_results = trainer.train()
        callbacks.on_training_end(training_results)
        return training_results
    except KeyboardInterrupt:
        print("Training interrupted by user")
        return {"interrupted": True}
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise


def evaluate_model(model: TransformerVAE, test_loader: DataLoader,
                  tokenizer: SMILESTokenizer, device: torch.device) -> dict:
    """Evaluate the trained model."""
    print("Evaluating model...")

    model.eval()
    test_metrics = {}

    if test_loader:
        # Compute test loss
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                outputs = model(batch['sequences'], target_sequence=None, training=False)

                # Proper loss computation matching trainer logic
                if 'reconstruction' in outputs and 'mu' in outputs and 'logvar' in outputs:
                    reconstruction = outputs['reconstruction']
                    targets = batch['sequences']

                    # Reconstruction loss - handle different output shapes
                    if reconstruction.dim() == 3:  # (batch, seq, vocab) - sequence generation
                        # Handle sequence length mismatch (autoregressive can produce shorter sequences)
                        recon_seq_len = reconstruction.size(1)
                        target_seq_len = targets.size(1)

                        if recon_seq_len != target_seq_len:
                            # Align sequences: truncate targets to match reconstruction length
                            min_len = min(recon_seq_len, target_seq_len)
                            reconstruction = reconstruction[:, :min_len, :]
                            targets = targets[:, :min_len]

                        # Use cross-entropy for categorical distribution over vocabulary
                        recon_loss = torch.nn.functional.cross_entropy(
                            reconstruction.reshape(-1, reconstruction.size(-1)),
                            targets.reshape(-1),
                            ignore_index=0,  # Assume 0 is pad token
                            reduction='mean'
                        )
                    elif reconstruction.dim() == 2 and targets.dim() == 2:  # (batch, seq) - direct sequence matching
                        if reconstruction.shape == targets.shape:
                            recon_loss = torch.nn.functional.mse_loss(reconstruction, targets.float())
                        else:
                            print(f"Warning: Shape mismatch - reconstruction {reconstruction.shape} vs targets {targets.shape}")
                            continue
                    else:
                        recon_loss = torch.nn.functional.mse_loss(reconstruction, targets.float())

                    # KL divergence loss
                    mu, logvar = outputs['mu'], outputs['logvar']
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)

                    # Total loss (using beta=1.0 for evaluation)
                    batch_loss = recon_loss + kl_loss

                    total_loss += batch_loss.item()
                    num_batches += 1

        if num_batches > 0:
            test_metrics['test_loss'] = total_loss / num_batches

    # Generate samples and compute molecular metrics
    print("Generating samples for evaluation...")

    try:
        with torch.no_grad():
            # Generate samples
            generated_samples = model.sample(num_samples=1000, device=str(device))

            # Decode to SMILES (if samples are token sequences)
            if generated_samples.dim() == 2:  # Token sequences
                generated_smiles = tokenizer.decode_batch(generated_samples)
            else:
                # For continuous outputs, we'd need a different approach
                generated_smiles = [f"sample_{i}" for i in range(len(generated_samples))]

            # Get reference molecules
            reference_smiles = []
            if test_loader:
                for batch in test_loader:
                    reference_smiles.extend(batch['smiles'])
                    if len(reference_smiles) >= 1000:  # Limit for efficiency
                        break

            # Compute molecular metrics
            if reference_smiles:
                molecular_metrics = compute_molecular_metrics(generated_smiles, reference_smiles)
                test_metrics.update(molecular_metrics)

                print("Molecular Generation Metrics:")
                for key, value in molecular_metrics.items():
                    print(f"  {key}: {value:.4f}")

    except Exception as e:
        print(f"Warning: Failed to generate samples for evaluation: {e}")

    return test_metrics


def generate_samples(model: TransformerVAE, tokenizer: SMILESTokenizer,
                    num_samples: int, device: torch.device, output_dir: str) -> None:
    """Generate molecular samples."""
    print(f"Generating {num_samples} samples...")

    model.eval()

    with torch.no_grad():
        samples = model.sample(num_samples=num_samples, device=str(device))

        # Decode samples
        if samples.dim() == 2:  # Token sequences
            smiles_samples = tokenizer.decode_batch(samples)
        else:
            smiles_samples = [f"continuous_sample_{i}" for i in range(len(samples))]

    # Save samples
    samples_file = os.path.join(output_dir, "generated_samples.txt")
    with open(samples_file, 'w') as f:
        for smiles in smiles_samples:
            f.write(f"{smiles}\n")

    print(f"Samples saved to: {samples_file}")


def main():
    """Main training function."""
    # Parse arguments
    args = parse_arguments()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    # Setup device
    device = setup_device(args.device)

    try:
        # Load configurations
        model_config, training_config = load_configurations(args)

        # Set random seeds
        if training_config.random_seed is not None:
            set_random_seeds(training_config.random_seed)
            print(f"Random seed set to: {training_config.random_seed}")

        # Setup callbacks
        callbacks = TrainingCallbacks(log_dir=args.output_dir)

        # Setup W&B if requested
        if args.wandb_project:
            callbacks.setup_wandb(args.wandb_project, training_config.__dict__)

        # Setup data
        tokenizer, train_loader, val_loader, test_loader = setup_data(training_config, debug=args.debug)

        # Setup model and trainer
        model, trainer = setup_model_and_trainer(model_config, training_config, device, tokenizer)

        # Resume from checkpoint if requested
        if args.resume_from:
            if os.path.exists(args.resume_from):
                trainer.load_checkpoint(args.resume_from)
                print(f"Resumed from checkpoint: {args.resume_from}")
            else:
                print(f"Warning: Checkpoint not found: {args.resume_from}")

        if not args.evaluate_only:
            # Train model
            training_results = train_model(trainer, train_loader, val_loader, callbacks)

            # Save final model
            final_model_path = os.path.join(args.output_dir, "final_model.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': model_config,
                'training_config': training_config,
                'tokenizer': tokenizer,
            }, final_model_path)
            print(f"Final model saved to: {final_model_path}")

        # Evaluate model
        if test_loader:
            test_metrics = evaluate_model(model, test_loader, tokenizer, device)
            print(f"Test metrics: {test_metrics}")

        # Generate samples if requested
        if args.generate_samples > 0:
            generate_samples(model, tokenizer, args.generate_samples, device, args.output_dir)

        # Create summary report
        summary_report = callbacks.create_summary_report()
        print("\n" + summary_report)

        print("Training completed successfully!")

    except Exception as e:
        print(f"Error during training: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()