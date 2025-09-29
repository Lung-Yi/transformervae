# TransformerVAE Configuration System Quickstart

This guide demonstrates the complete workflow for using the modular TransformerVAE configuration system, from setup through training and evaluation.

## Prerequisites

```bash
# Install dependencies
pip install torch>=2.0 rdkit pyyaml dataclasses-json pytest

# Clone and setup repository
git clone <repository-url>
cd TransformerVAE
pip install -e .
```

## Step 1: Configure Your Model

### Create Model Architecture Configuration

Create a YAML file defining your model architecture:

```yaml
# config/model_configs/my_transformer.yaml
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
      pooling_type: "attention"

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

latent_regression_head:
  - layer_type: "regression_head"
    input_dim: 64
    output_dim: 5
    dropout: 0.1
    activation: "relu"
    layer_params:
      hidden_dims: [128, 64]
```

### Create Training Configuration

```yaml
# config/training_configs/my_training.yaml
learning_rate: 0.001
batch_size: 32
epochs: 100
beta: 1.0
optimizer_type: "adam"
weight_decay: 0.0001
gradient_clip_norm: 1.0
validation_freq: 5
checkpoint_freq: 10

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

random_seed: 42
```

## Step 2: Load and Validate Configuration

```python
from transformervae.config.basic_config import (
    load_model_config,
    load_training_config,
    validate_config
)

# Load configurations
model_config = load_model_config("config/model_configs/my_transformer.yaml")
training_config = load_training_config("config/training_configs/my_training.yaml")

# Validate configuration compatibility
validate_config(model_config, training_config)
print("Configuration validation passed!")
```

## Step 3: Create and Initialize Model

```python
from transformervae.models.model import TransformerVAE
from transformervae.models.utils import count_parameters, model_summary

# Create model from configuration
model = TransformerVAE.from_config(model_config)

# Display model information
print(f"Total parameters: {count_parameters(model):,}")
print(model_summary(model))

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
```

## Step 4: Setup Data Loading

```python
from transformervae.data.dataset import MolecularDataset
from transformervae.data.tokenizer import SMILESTokenizer
from torch.utils.data import DataLoader

# Initialize tokenizer and dataset
tokenizer = SMILESTokenizer(vocab_size=training_config.dataset_config["vocab_size"])
dataset = MolecularDataset.from_config(training_config.dataset_config, tokenizer)

# Create data loaders
train_loader = DataLoader(
    dataset.train_dataset,
    batch_size=training_config.batch_size,
    shuffle=True
)

val_loader = DataLoader(
    dataset.val_dataset,
    batch_size=training_config.batch_size,
    shuffle=False
)
```

## Step 5: Setup Training

```python
from transformervae.training.trainer import Trainer
from transformervae.training.evaluator import Evaluator
from transformervae.utils.reproducibility import set_random_seeds

# Set random seeds for reproducibility
set_random_seeds(training_config.random_seed)

# Create trainer and evaluator
trainer = Trainer.from_config(training_config)
evaluator = Evaluator()

# Setup model for training
trainer.setup_model(model)
trainer.setup_data(train_loader, val_loader)
```

## Step 6: Train the Model

```python
# Start training
trainer.train()

# Training will automatically:
# - Save checkpoints every checkpoint_freq epochs
# - Validate every validation_freq epochs
# - Log metrics to specified logging system
# - Apply learning rate scheduling
```

## Step 7: Evaluate Model Performance

```python
from transformervae.utils.metrics import compute_molecular_metrics

# Generate molecules
model.eval()
with torch.no_grad():
    generated_samples = model.sample(num_samples=1000, device=device)

# Convert to SMILES
generated_smiles = tokenizer.decode_batch(generated_samples)

# Load reference molecules for comparison
reference_smiles = dataset.get_reference_molecules()

# Compute molecular generation metrics
metrics = compute_molecular_metrics(generated_smiles, reference_smiles)

print("Generation Metrics:")
print(f"Validity: {metrics['validity']:.3f}")
print(f"Uniqueness: {metrics['uniqueness']:.3f}")
print(f"Novelty: {metrics['novelty']:.3f}")
print(f"FCD Score: {metrics['fcd_score']:.3f}")
```

## Step 8: Property Prediction (Optional)

If your model includes regression/classification heads:

```python
# Evaluate property prediction
if hasattr(model, 'latent_regression_head'):
    # Test on molecules with known properties
    test_molecules = dataset.get_property_test_set()
    property_predictions = []

    for batch in test_molecules:
        with torch.no_grad():
            output = model(batch['smiles'])
            properties = output['property_regression']
            property_predictions.append(properties)

    # Compute property prediction metrics
    from sklearn.metrics import mean_squared_error, r2_score

    predictions = torch.cat(property_predictions).cpu().numpy()
    targets = test_molecules['properties'].numpy()

    mse = mean_squared_error(targets, predictions)
    r2 = r2_score(targets, predictions)

    print(f"Property Prediction MSE: {mse:.4f}")
    print(f"Property Prediction R²: {r2:.4f}")
```

## Step 9: Save and Load Trained Model

```python
# Save complete model state
checkpoint = {
    'model_state_dict': model.state_dict(),
    'model_config': model_config,
    'training_config': training_config,
    'tokenizer': tokenizer,
    'metrics': metrics
}

torch.save(checkpoint, 'trained_model.pt')

# Load model later
checkpoint = torch.load('trained_model.pt')
loaded_model = TransformerVAE.from_config(checkpoint['model_config'])
loaded_model.load_state_dict(checkpoint['model_state_dict'])
```

## Command Line Interface

For convenience, use the main training script:

```bash
# Train with default configurations
python main_train_VAE.py \
    --model_config config/model_configs/my_transformer.yaml \
    --training_config config/training_configs/my_training.yaml \
    --output_dir results/my_experiment

# Resume from checkpoint
python main_train_VAE.py \
    --model_config config/model_configs/my_transformer.yaml \
    --training_config config/training_configs/my_training.yaml \
    --resume_from results/my_experiment/checkpoint_epoch_50.pt

# Evaluate only (no training)
python main_train_VAE.py \
    --model_config config/model_configs/my_transformer.yaml \
    --checkpoint results/my_experiment/final_model.pt \
    --evaluate_only
```

## Configuration Examples

### Small Model for Testing
```yaml
# config/model_configs/small_test.yaml
encoder:
  - layer_type: "transformer_encoder"
    input_dim: 50
    output_dim: 128
    dropout: 0.1
    activation: "relu"
    layer_params:
      num_heads: 4
      dim_feedforward: 256
      num_layers: 2

sampler:
  - layer_type: "latent_sampler"
    input_dim: 128
    output_dim: 32
    dropout: 0.0
    activation: "linear"

decoder:
  - layer_type: "transformer_decoder"
    input_dim: 32
    output_dim: 50
    dropout: 0.1
    activation: "relu"
    layer_params:
      num_heads: 4
      dim_feedforward: 256
      num_layers: 2
```

### Large Model for Production
```yaml
# config/model_configs/large_production.yaml
encoder:
  - layer_type: "transformer_encoder"
    input_dim: 200
    output_dim: 512
    dropout: 0.1
    activation: "gelu"
    layer_params:
      num_heads: 16
      dim_feedforward: 2048
      num_layers: 8

sampler:
  - layer_type: "latent_sampler"
    input_dim: 512
    output_dim: 128
    dropout: 0.0
    activation: "linear"

decoder:
  - layer_type: "transformer_decoder"
    input_dim: 128
    output_dim: 200
    dropout: 0.1
    activation: "gelu"
    layer_params:
      num_heads: 16
      dim_feedforward: 2048
      num_layers: 8
```

## Troubleshooting

### Common Configuration Errors

1. **Dimension Mismatch**: Ensure encoder output matches sampler input
2. **Invalid Layer Types**: Check supported layer types in documentation
3. **Missing Files**: Verify all configuration file paths are correct
4. **Memory Issues**: Reduce batch size or model dimensions

### Performance Optimization

1. **Use Mixed Precision**: Add `use_amp: true` to training config
2. **Gradient Accumulation**: Increase effective batch size with `gradient_accumulation_steps`
3. **Learning Rate Scheduling**: Experiment with different scheduler types
4. **Data Loading**: Increase `num_workers` in DataLoader for faster I/O

### Validation Steps

Before training, verify:
- [ ] Configuration files load without errors
- [ ] Model creates successfully from config
- [ ] Sample data passes through model
- [ ] All metrics can be computed
- [ ] Checkpointing works correctly

This completes the quickstart guide for the TransformerVAE configuration system!