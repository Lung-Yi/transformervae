"""
Training callbacks for experiment tracking and monitoring.

This module provides callback functionality for logging, checkpointing,
and experiment tracking during training.
"""

import os
import time
import json
from typing import Dict, Any, Optional, List
from pathlib import Path


class TrainingCallbacks:
    """Training callbacks for monitoring and logging."""

    def __init__(self, log_dir: Optional[str] = None):
        """
        Initialize training callbacks.

        Args:
            log_dir: Directory for saving logs and outputs
        """
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Tracking variables
        self.start_time = None
        self.epoch_start_time = None
        self.training_log = []

        # Experiment tracking
        self.experiment_id = self._generate_experiment_id()

        # Initialize logging
        self.log_file = self.log_dir / f"training_{self.experiment_id}.log"

    def _generate_experiment_id(self) -> str:
        """Generate unique experiment ID."""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def on_training_start(self, config: Dict[str, Any]) -> None:
        """
        Called at the start of training.

        Args:
            config: Training configuration
        """
        self.start_time = time.time()

        # Log training start
        self._log_message("Training started")
        self._log_message(f"Experiment ID: {self.experiment_id}")

        # Save configuration
        config_file = self.log_dir / f"config_{self.experiment_id}.json"
        with open(config_file, 'w') as f:
            # Convert config to serializable format
            serializable_config = self._make_serializable(config)
            json.dump(serializable_config, f, indent=2)

        self._log_message(f"Configuration saved to {config_file}")

    def on_epoch_start(self, epoch: int, total_epochs: int) -> None:
        """
        Called at the start of each epoch.

        Args:
            epoch: Current epoch number
            total_epochs: Total number of epochs
        """
        self.epoch_start_time = time.time()
        self._log_message(f"Starting epoch {epoch + 1}/{total_epochs}")

    def on_epoch_end(self, epoch: int, train_metrics: Dict[str, float],
                     val_metrics: Optional[Dict[str, float]] = None) -> None:
        """
        Called at the end of each epoch.

        Args:
            epoch: Current epoch number
            train_metrics: Training metrics for the epoch
            val_metrics: Validation metrics for the epoch
        """
        epoch_time = time.time() - self.epoch_start_time

        # Create epoch log entry
        epoch_log = {
            'epoch': epoch,
            'epoch_time': epoch_time,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }

        self.training_log.append(epoch_log)

        # Log epoch summary
        log_msg = f"Epoch {epoch + 1} completed in {epoch_time:.2f}s - "
        log_msg += f"Train Loss: {train_metrics.get('total_loss', 0):.4f}"

        if val_metrics:
            log_msg += f", Val Loss: {val_metrics.get('total_loss', 0):.4f}"

        self._log_message(log_msg)

    def on_training_end(self, final_metrics: Dict[str, Any]) -> None:
        """
        Called at the end of training.

        Args:
            final_metrics: Final training metrics
        """
        total_time = time.time() - self.start_time

        self._log_message(f"Training completed in {total_time:.2f}s")
        self._log_message(f"Final metrics: {final_metrics}")

        # Save training log
        log_file = self.log_dir / f"training_log_{self.experiment_id}.json"
        with open(log_file, 'w') as f:
            json.dump(self.training_log, f, indent=2)

        self._log_message(f"Training log saved to {log_file}")

    def on_checkpoint_save(self, checkpoint_path: str, metrics: Dict[str, float]) -> None:
        """
        Called when a checkpoint is saved.

        Args:
            checkpoint_path: Path to saved checkpoint
            metrics: Current metrics
        """
        self._log_message(f"Checkpoint saved: {checkpoint_path}")
        self._log_message(f"Checkpoint metrics: {metrics}")

    def on_best_model_save(self, model_path: str, metrics: Dict[str, float]) -> None:
        """
        Called when the best model is saved.

        Args:
            model_path: Path to saved model
            metrics: Best metrics achieved
        """
        self._log_message(f"Best model saved: {model_path}")
        self._log_message(f"Best metrics: {metrics}")

    def on_validation_end(self, epoch: int, val_metrics: Dict[str, float]) -> None:
        """
        Called after validation.

        Args:
            epoch: Current epoch
            val_metrics: Validation metrics
        """
        self._log_message(f"Validation completed for epoch {epoch + 1}")

        # Log detailed validation metrics
        for key, value in val_metrics.items():
            self._log_message(f"  {key}: {value:.4f}")

    def log_molecular_metrics(self, epoch: int, molecular_metrics: Dict[str, float]) -> None:
        """
        Log molecular generation metrics.

        Args:
            epoch: Current epoch
            molecular_metrics: Molecular generation metrics
        """
        self._log_message(f"Molecular generation metrics for epoch {epoch + 1}:")

        for key, value in molecular_metrics.items():
            self._log_message(f"  {key}: {value:.4f}")

    def log_custom_metric(self, name: str, value: Any, epoch: Optional[int] = None) -> None:
        """
        Log custom metric.

        Args:
            name: Metric name
            value: Metric value
            epoch: Optional epoch number
        """
        if epoch is not None:
            self._log_message(f"Epoch {epoch + 1} - {name}: {value}")
        else:
            self._log_message(f"{name}: {value}")

    def _log_message(self, message: str) -> None:
        """
        Log message to file and console.

        Args:
            message: Message to log
        """
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"

        # Print to console
        print(log_entry)

        # Write to log file
        with open(self.log_file, 'a') as f:
            f.write(log_entry + '\n')

    def _make_serializable(self, obj: Any) -> Any:
        """
        Convert object to JSON-serializable format.

        Args:
            obj: Object to convert

        Returns:
            JSON-serializable object
        """
        if hasattr(obj, '__dict__'):
            # Handle dataclass or object with attributes
            result = {}
            for key, value in obj.__dict__.items():
                if not key.startswith('_'):  # Skip private attributes
                    result[key] = self._make_serializable(value)
            return result
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            # For other types, convert to string representation
            return str(obj)

    def setup_wandb(self, project_name: str, config: Dict[str, Any],
                   entity: Optional[str] = None) -> None:
        """
        Setup Weights & Biases logging.

        Args:
            project_name: W&B project name
            config: Configuration to log
            entity: W&B entity (optional)
        """
        try:
            import wandb

            wandb.init(
                project=project_name,
                entity=entity,
                config=self._make_serializable(config),
                name=f"experiment_{self.experiment_id}"
            )

            self._log_message("Weights & Biases logging initialized")
            self.wandb_enabled = True

        except ImportError:
            self._log_message("Warning: wandb not available, skipping W&B logging")
            self.wandb_enabled = False
        except Exception as e:
            self._log_message(f"Warning: Failed to initialize W&B: {e}")
            self.wandb_enabled = False

    def log_to_wandb(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log metrics to Weights & Biases.

        Args:
            metrics: Metrics to log
            step: Optional step number
        """
        if hasattr(self, 'wandb_enabled') and self.wandb_enabled:
            try:
                import wandb
                wandb.log(metrics, step=step)
            except Exception as e:
                self._log_message(f"Warning: Failed to log to W&B: {e}")

    def create_summary_report(self) -> str:
        """
        Create a summary report of the training session.

        Returns:
            Summary report as string
        """
        if not self.training_log:
            return "No training data available."

        # Calculate summary statistics
        final_epoch = self.training_log[-1]
        total_epochs = len(self.training_log)
        total_time = sum(entry['epoch_time'] for entry in self.training_log)

        # Best metrics
        best_train_loss = min(entry['train_metrics'].get('total_loss', float('inf'))
                             for entry in self.training_log)

        best_val_loss = float('inf')
        if any(entry['val_metrics'] for entry in self.training_log):
            best_val_loss = min(entry['val_metrics'].get('total_loss', float('inf'))
                               for entry in self.training_log
                               if entry['val_metrics'])

        # Create report
        report = f"""
Training Summary Report
=======================
Experiment ID: {self.experiment_id}
Total Epochs: {total_epochs}
Total Training Time: {total_time:.2f} seconds
Average Epoch Time: {total_time/total_epochs:.2f} seconds

Final Metrics:
  Train Loss: {final_epoch['train_metrics'].get('total_loss', 'N/A'):.4f}
  Val Loss: {final_epoch['val_metrics'].get('total_loss', 'N/A') if final_epoch['val_metrics'] else 'N/A'}

Best Metrics:
  Best Train Loss: {best_train_loss:.4f}
  Best Val Loss: {best_val_loss:.4f if best_val_loss != float('inf') else 'N/A'}

Log Directory: {self.log_dir}
"""

        # Save report
        report_file = self.log_dir / f"summary_{self.experiment_id}.txt"
        with open(report_file, 'w') as f:
            f.write(report)

        self._log_message(f"Summary report saved to {report_file}")

        return report