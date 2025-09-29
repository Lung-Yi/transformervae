# TransformerVAE Example Usage Guide

This guide provides comprehensive examples of how to use the modified TransformerVAE project for molecular generation and analysis.

## Overview

The project has been enhanced with a modern configuration system using YAML files and dataclasses, making it easier to train and configure TransformerVAE models. The main training script `main_train_VAE.py` provides a complete command-line interface with extensive configuration options.

## Prerequisites

### Environment Setup

```bash
# Create conda environment
conda create -n transformervae python=3.13
conda activate transformervae

# Install dependencies
pip install -r requirements.txt

# Install PyTorch (adjust for your CUDA version)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129
```

### Required Dependencies
- Python 3.13+
- PyTorch 2.0+
- RDKit 2025.3.6
- PyYAML 6.0+
- NumPy, Pandas, tqdm, matplotlib, seaborn
- wandb (for experiment tracking)

## Project Structure

```
TransformerVAE/
├── main_train_VAE.py           # Main training script (NEW)
├── train.py                    # Legacy training script
├── generate.py                 # Molecule generation
├── decode.py                   # Decode from latent space
├── featurize.py               # Extract molecular features
├── transformervae/            # Core package
│   ├── config/               # Configuration system (NEW)
│   │   ├── basic_config.py   # Configuration dataclasses
│   │   ├── model_configs/    # Model architecture configs
│   │   └── training_configs/ # Training parameter configs
│   ├── models/               # Model implementations
│   ├── data/                 # Data handling
│   ├── training/             # Training utilities
│   └── utils/                # Utilities
└── data/                     # Dataset directory
```

## Configuration System

The new configuration system uses YAML files and Python dataclasses for type-safe, validated configurations.

### Model Configuration

Model architectures are defined in `transformervae/config/model_configs/`:

**Example: base_transformer.yaml**
```yaml
encoder:
  - layer_type: "transformer_encoder"
    input_dim: 100
    output_dim: 256
    dropout: 0.1
    activation: "relu"
    layer_params:
      num_heads: 8
      dim_feedforward: 512
      num_layers: 4

  - layer_type: "pooling"
    input_dim: 256
    output_dim: 256
    dropout: 0.0
    activation: "linear"
    layer_params:
      pooling_type: "mean"

sampler:
  - layer_type: "latent_sampler"
    input_dim: 256
    output_dim: 64
    dropout: 0.0
    activation: "linear"

decoder:
  - layer_type: "transformer_decoder"
    input_dim: 64
    output_dim: 256
    dropout: 0.1
    activation: "relu"
    layer_params:
      num_heads: 8
      dim_feedforward: 512
      num_layers: 4

  - layer_type: "linear"
    input_dim: 256
    output_dim: 100
    dropout: 0.0
    activation: "linear"
```

### Training Configuration

Training parameters are defined in `transformervae/config/training_configs/`:

**Example: moses_config.yaml**
```yaml
learning_rate: 0.001
batch_size: 64
epochs: 100
beta: 1.0
optimizer_type: "adam"
weight_decay: 0.0001
gradient_clip_norm: 1.0
validation_freq: 5
checkpoint_freq: 10
random_seed: 42

scheduler_config:
  type: "reduce_on_plateau"
  patience: 10
  factor: 0.5
  min_lr: 0.00001

dataset_config:
  dataset_type: "moses"
  data_path: "data/moses"
  max_sequence_length: 100
  vocab_size: 1000
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  preprocessing_config:
    augment_smiles: true
    canonical: false
    max_atoms: 50
    add_explicit_hydrogens: false
```

## Usage Examples

### 1. Basic Training

Train a model with default configurations:

```bash
python main_train_VAE.py
```

This will use:
- Model config: `transformervae/config/model_configs/base_transformer.yaml`
- Training config: `transformervae/config/training_configs/moses_config.yaml`
- Output directory: `results/`

### 2. Custom Configuration Training

Train with custom configurations:

```bash
python main_train_VAE.py \
    --model_config transformervae/config/model_configs/large_transformer.yaml \
    --training_config transformervae/config/training_configs/zinc_config.yaml \
    --output_dir my_experiment \
    --data_path data/ZINC_250k.smi

# or training on MOSES dataset
python main_train_VAE.py \
    --model_config transformervae/config/model_configs/base_transformer.yaml \
    --training_config transformervae/config/training_configs/moses_config_fixed.yaml \
    --output_dir moses_experiment \
    --data_path data/moses
```

### 3. Training with Weights & Biases

Enable experiment tracking:

```bash
python main_train_VAE.py \
    --wandb_project "transformervae-experiments" \
    --output_dir wandb_experiment
```

### 4. Debug Mode Training

Quick training for testing (reduced epochs and batch size):

```bash
python main_train_VAE.py \
    --debug \
    --output_dir debug_run
```

### 5. Resume Training from Checkpoint

```bash
python main_train_VAE.py \
    --resume_from results/checkpoints/epoch_50.pt \
    --output_dir resumed_training
```

### 6. Evaluation Only

Evaluate a trained model without training:

```bash
python main_train_VAE.py \
    --evaluate_only \
    --resume_from results/final_model.pt \
    --output_dir evaluation
```

### 7. Training with Sample Generation

Train and generate 1000 samples after training:

```bash
python main_train_VAE.py \
    --generate_samples 1000 \
    --output_dir training_with_generation
```

### 8. Custom Device and Seed

```bash
python main_train_VAE.py \
    --device cuda:1 \
    --seed 123 \
    --output_dir custom_device_seed
```

## Data Preparation

### Prepare SMILES Dataset

If using your own SMILES data:

1. **Format your data**: Create a text file with one SMILES string per line
   ```
   CCO
   CC(=O)O
   c1ccccc1
   ```

2. **Place in data directory**:
   ```bash
   mkdir -p data/my_dataset
   cp my_smiles.smi data/my_dataset/
   ```

3. **Update dataset configuration** or use command line override:
   ```bash
   python main_train_VAE.py \
       --data_path data/my_dataset/my_smiles.smi
   ```

### Using Provided Datasets

The project supports MOSES and ZINC datasets:

```bash
# Download ZINC-250k (example)
wget -O data/ZINC_250k.smi "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/data/zinc_250k.smi"

# Train with ZINC data
python main_train_VAE.py \
    --training_config transformervae/config/training_configs/zinc_config.yaml \
    --data_path data/ZINC_250k.smi
```

## Advanced Usage

### Creating Custom Configurations

1. **Custom Model Architecture**:
   ```bash
   cp transformervae/config/model_configs/base_transformer.yaml my_model.yaml
   # Edit my_model.yaml with your architecture
   python main_train_VAE.py --model_config my_model.yaml
   ```

2. **Custom Training Parameters**:
   ```bash
   cp transformervae/config/training_configs/moses_config.yaml my_training.yaml
   # Edit my_training.yaml with your parameters
   python main_train_VAE.py --training_config my_training.yaml
   ```

### Hyperparameter Sweeps

Example script for hyperparameter sweeping:

```bash
#!/bin/bash
for lr in 0.001 0.0005 0.0001; do
  for beta in 0.5 1.0 2.0; do
    # Create custom config
    sed "s/learning_rate: 0.001/learning_rate: $lr/" moses_config.yaml > temp_config.yaml
    sed -i "s/beta: 1.0/beta: $beta/" temp_config.yaml

    # Train
    python main_train_VAE.py \
        --training_config temp_config.yaml \
        --output_dir "sweep_lr${lr}_beta${beta}" \
        --wandb_project "hyperparameter-sweep"
  done
done
```

### Profiling and Performance

Enable profiling for performance analysis:

```bash
python main_train_VAE.py \
    --profile \
    --debug \
    --output_dir profiling_run
```

## Post-Training Analysis

### Generate Molecules

After training, generate new molecules:

```bash
python generate.py \
    --weight results/final_model.pt \
    --n 30000 \
    --name my_generation
```

### Extract Molecular Features

Get latent representations:

```bash
python featurize.py \
    --data processed_data_name \
    --weight results/final_model.pt \
    --name my_features
```

### Decode from Latent Space

Decode molecules from custom latent vectors:

```bash
python decode.py \
    --latent my_latent_vectors.csv \
    --weight results/final_model.pt \
    --name my_decoding
```

## Output Files

After training, the output directory contains:

```
results/
├── final_model.pt              # Final trained model
├── checkpoints/               # Training checkpoints
├── logs/                      # Training logs
├── config.yaml               # Used configuration
├── val_score.csv             # Validation metrics
├── generated_samples.txt     # Generated molecules (if requested)
└── training_summary.txt      # Training summary
```

## Common Issues and Solutions

### 1. CUDA Out of Memory
- Reduce batch size in training config
- Use `--debug` mode for smaller models
- Check GPU memory with `nvidia-smi`

### 2. Configuration Validation Errors
- Check YAML syntax
- Ensure dimension compatibility between layers
- Verify data types (int vs float)

### 3. Data Loading Issues
- Verify file paths exist
- Check SMILES format (one per line)
- Ensure proper file permissions

### 4. Slow Training
- Enable CUDA if available
- Increase batch size if memory allows
- Use multiple workers for data loading

## Best Practices

1. **Start with debug mode** for new configurations
2. **Use version control** for your configs
3. **Monitor with wandb** for longer experiments
4. **Save checkpoints frequently** for long training runs
5. **Validate your data** before training
6. **Test generation quality** with small samples first

## Configuration Reference

### Supported Layer Types
- `transformer_encoder`
- `transformer_decoder`
- `latent_sampler`
- `pooling`
- `linear`
- `regression_head`
- `classification_head`

### Supported Activations
- `relu`, `gelu`, `tanh`, `sigmoid`, `linear`, `leaky_relu`

### Supported Optimizers
- `adam`, `sgd`, `adamw`

### Supported Datasets
- `moses`, `zinc15`, `chembl`

This guide should provide a comprehensive overview of how to use the modified TransformerVAE project effectively for molecular generation tasks.