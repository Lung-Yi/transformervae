#!/usr/bin/env python3
"""
Quick test script for teacher forcing implementation.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from transformervae.config.basic_config import load_model_config, load_training_config
from transformervae.models.model import TransformerVAE
from transformervae.data.tokenizer import SMILESTokenizer

def test_teacher_forcing():
    """Test teacher forcing forward pass."""
    print("Testing teacher forcing implementation...")

    # Load configurations
    model_config = load_model_config("transformervae/config/model_configs/base_transformer.yaml")

    # Create model
    model = TransformerVAE.from_config(model_config)
    model.eval()

    # Create dummy data
    batch_size = 4
    seq_len = 20

    # Create dummy input sequences (token indices)
    input_sequences = torch.randint(1, 26, (batch_size, seq_len))  # vocab size 26

    print(f"Input shape: {input_sequences.shape}")

    # Test teacher forcing mode (training)
    print("\nTesting teacher forcing mode...")
    try:
        with torch.no_grad():
            outputs_tf = model(input_sequences, target_sequence=input_sequences, training=True)
            print(f"Teacher forcing output shapes:")
            for key, value in outputs_tf.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}")
    except Exception as e:
        print(f"Error in teacher forcing: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test autoregressive mode (inference)
    print("\nTesting autoregressive mode...")
    try:
        with torch.no_grad():
            outputs_ar = model(input_sequences, target_sequence=None, training=False)
            print(f"Autoregressive output shapes:")
            for key, value in outputs_ar.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}")
    except Exception as e:
        print(f"Error in autoregressive: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\nTeacher forcing test completed successfully!")
    return True

if __name__ == "__main__":
    success = test_teacher_forcing()
    if success:
        print("All tests passed!")
    else:
        print("Tests failed!")
        sys.exit(1)