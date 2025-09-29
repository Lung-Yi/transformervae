#!/usr/bin/env python3
"""
Test training with a very small dataset to check performance.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from transformervae.config.basic_config import load_model_config, load_training_config
from transformervae.models.model import TransformerVAE
from transformervae.training.trainer import VAETrainer
from transformervae.data.tokenizer import SMILESTokenizer
from torch.utils.data import DataLoader, TensorDataset

def test_small_training():
    """Test training with minimal data."""
    print("Testing small training loop...")

    # Load configurations
    model_config = load_model_config("transformervae/config/model_configs/base_transformer.yaml")
    training_config = load_training_config("transformervae/config/training_configs/moses_config.yaml")

    # Override for very small test
    training_config.batch_size = 2
    training_config.epochs = 1

    # Create tokenizer
    tokenizer = SMILESTokenizer(vocab_size=26, max_length=20)
    # Simple vocabulary for testing
    tokenizer.char_to_idx = {f'char_{i}': i for i in range(26)}
    tokenizer.idx_to_char = {i: f'char_{i}' for i in range(26)}

    # Create model
    model = TransformerVAE.from_config(model_config)

    # Create trainer
    trainer = VAETrainer.from_config(training_config, tokenizer=tokenizer)
    trainer.setup_model(model)

    # Create minimal dummy data
    sequences = torch.randint(1, 26, (10, 20))  # 10 samples, length 20
    smiles = [f"dummy_smiles_{i}" for i in range(10)]

    # Create dataset
    dataset = TensorDataset(sequences)

    # Custom collate function
    def collate_fn(batch):
        sequences = torch.stack([item[0] for item in batch])
        batch_smiles = smiles[:len(sequences)]  # Get corresponding SMILES
        return {
            'sequences': sequences,
            'smiles': batch_smiles
        }

    # Create data loader
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    trainer.setup_data(train_loader, None)

    print("Starting small training test...")
    try:
        # Train for just one batch to test
        trainer.model.train()

        for batch_idx, batch in enumerate(train_loader):
            print(f"Processing batch {batch_idx + 1}...")

            # Move batch to device
            batch = {k: v.to(trainer.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass with teacher forcing
            target_sequence = batch['sequences']
            outputs = trainer.model(batch['sequences'], target_sequence=target_sequence, training=True)

            # Compute loss
            loss_dict = trainer._compute_loss(outputs, batch)

            # Backward pass
            trainer.optimizer.zero_grad()
            loss_dict['total_loss'].backward()
            trainer.optimizer.step()

            print(f"  Total loss: {loss_dict['total_loss'].item():.4f}")
            print(f"  Recon loss: {loss_dict['recon_loss'].item():.4f}")
            print(f"  KL loss: {loss_dict['kl_loss'].item():.4f}")

            # Only test one batch
            break

        print("Small training test completed successfully!")
        return True

    except Exception as e:
        print(f"Error in training: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_small_training()
    if success:
        print("Training test passed!")
    else:
        print("Training test failed!")
        sys.exit(1)