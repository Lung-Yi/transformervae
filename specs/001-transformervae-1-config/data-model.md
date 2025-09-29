# Data Model: TransformerVAE Configuration System

## Core Configuration Entities

### LayerConfig
Defines configuration for individual neural network layers.

**Fields:**
- `layer_type: str` - Type identifier for the layer (e.g., "transformer_encoder", "transformer_decoder", "linear", "pooling")
- `input_dim: int` - Input dimension size
- `output_dim: int` - Output dimension size
- `dropout: float` - Dropout probability (0.0-1.0)
- `activation: str` - Activation function name ("relu", "gelu", "tanh", etc.)
- `layer_params: Dict[str, Any]` - Additional layer-specific parameters

**Validation Rules:**
- `input_dim` and `output_dim` must be positive integers
- `dropout` must be between 0.0 and 1.0 inclusive
- `layer_type` must be from predefined set of supported layer types
- `activation` must be from supported activation functions

**Relationships:**
- Used by `DetailedModelArchitecture` in encoder, decoder, sampler, and head configurations

### DetailedModelArchitecture
Defines the complete structure of a TransformerVAE model.

**Fields:**
- `encoder: List[LayerConfig]` - Sequential encoder layer configurations
- `sampler: List[LayerConfig]` - VAE sampling layer configurations (typically μ and σ layers)
- `decoder: List[LayerConfig]` - Sequential decoder layer configurations
- `latent_regression_head: Optional[List[LayerConfig]]` - Optional molecular property regression head
- `latent_classification_head: Optional[List[LayerConfig]]` - Optional molecular classification head

**Validation Rules:**
- All layer lists must be non-empty (except optional heads)
- Encoder output dimension must match sampler input dimension
- Sampler output dimension must match decoder input dimension
- Layer dimensions must be compatible in sequence

**State Transitions:**
- Configuration → Validation → Model Instantiation → Training Ready

### VAETrainingConfig
Contains all training-related configuration parameters.

**Fields:**
- `learning_rate: float` - Optimizer learning rate
- `batch_size: int` - Training batch size
- `epochs: int` - Number of training epochs
- `beta: float` - KL divergence loss weight for β-VAE
- `scheduler_config: Dict[str, Any]` - Learning rate scheduler configuration
- `optimizer_type: str` - Optimizer type ("adam", "sgd", "adamw")
- `weight_decay: float` - L2 regularization weight
- `gradient_clip_norm: Optional[float]` - Gradient clipping norm
- `validation_freq: int` - Validation frequency in epochs
- `checkpoint_freq: int` - Model checkpoint frequency in epochs

**Validation Rules:**
- `learning_rate` must be positive
- `batch_size` must be positive integer
- `epochs` must be positive integer
- `beta` must be non-negative
- `weight_decay` must be non-negative
- Frequency parameters must be positive integers

### DatasetConfig
Configuration for dataset loading and preprocessing.

**Fields:**
- `dataset_type: str` - Dataset identifier ("moses", "zinc15")
- `data_path: str` - Path to dataset files
- `max_sequence_length: int` - Maximum SMILES sequence length
- `vocab_size: int` - Vocabulary size for tokenization
- `train_split: float` - Training set proportion (0.0-1.0)
- `val_split: float` - Validation set proportion (0.0-1.0)
- `test_split: float` - Test set proportion (0.0-1.0)
- `preprocessing_config: Dict[str, Any]` - Additional preprocessing parameters

**Validation Rules:**
- `dataset_type` must be from supported datasets
- `data_path` must exist and be accessible
- `max_sequence_length` must be positive
- Split proportions must sum to 1.0
- All split values must be between 0.0 and 1.0

## Training Runtime Entities

### ModelCheckpoint
Represents saved model state during training.

**Fields:**
- `epoch: int` - Training epoch number
- `model_state: Dict[str, Any]` - PyTorch model state dictionary
- `optimizer_state: Dict[str, Any]` - Optimizer state dictionary
- `config_snapshot: DetailedModelArchitecture` - Model configuration at checkpoint
- `training_config_snapshot: VAETrainingConfig` - Training configuration at checkpoint
- `metrics: Dict[str, float]` - Performance metrics at checkpoint time
- `timestamp: datetime` - Checkpoint creation timestamp
- `git_commit: Optional[str]` - Git commit hash for reproducibility

**Relationships:**
- References original configuration entities for reproducibility

### TrainingMetrics
Encapsulates evaluation metrics for molecular generation quality.

**Fields:**
- `reconstruction_loss: float` - VAE reconstruction loss component
- `kl_loss: float` - KL divergence loss component
- `total_loss: float` - Combined loss (reconstruction + β * KL)
- `validity: float` - Percentage of valid generated molecules
- `uniqueness: float` - Percentage of unique generated molecules
- `novelty: float` - Percentage of novel molecules (not in training set)
- `fcd_score: float` - Fréchet ChemNet Distance score
- `molecular_properties: Dict[str, float]` - Optional property prediction metrics

**Validation Rules:**
- Loss values must be non-negative
- Percentage metrics must be between 0.0 and 1.0
- FCD score must be non-negative

## Data Relationships

```
DetailedModelArchitecture
├── encoder: List[LayerConfig]
├── sampler: List[LayerConfig]
├── decoder: List[LayerConfig]
├── latent_regression_head: Optional[List[LayerConfig]]
└── latent_classification_head: Optional[List[LayerConfig]]

VAETrainingConfig
└── scheduler_config: Dict[str, Any]

DatasetConfig
└── preprocessing_config: Dict[str, Any]

ModelCheckpoint
├── config_snapshot: DetailedModelArchitecture
├── training_config_snapshot: VAETrainingConfig
└── metrics: TrainingMetrics

TrainingMetrics
└── molecular_properties: Dict[str, float]
```

## Configuration Validation Pipeline

1. **Schema Validation**: Verify all required fields present and types correct
2. **Domain Validation**: Check value ranges and constraints
3. **Consistency Validation**: Verify dimensional compatibility between layers
4. **Resource Validation**: Check dataset paths and system requirements
5. **Constitutional Validation**: Ensure compliance with architecture principles

## State Management

### Configuration States
- `DRAFT` - Configuration being edited
- `VALIDATED` - Passed all validation checks
- `ACTIVE` - Currently used for training
- `ARCHIVED` - Historical configuration for reference

### Model States
- `UNINITIALIZED` - Configuration loaded but model not created
- `INITIALIZED` - Model created from configuration
- `TRAINING` - Model currently training
- `TRAINED` - Training completed successfully
- `CHECKPOINTED` - Model saved with checkpoint

## Error Handling

### Configuration Errors
- `InvalidLayerTypeError` - Unsupported layer type specified
- `DimensionMismatchError` - Incompatible layer dimensions
- `ValidationError` - Generic validation failure
- `ConfigurationNotFoundError` - Configuration file not accessible

### Runtime Errors
- `ModelInitializationError` - Failed to create model from configuration
- `TrainingError` - Training process failure
- `CheckpointError` - Failed to save/load model checkpoint
- `DatasetError` - Dataset loading or processing failure