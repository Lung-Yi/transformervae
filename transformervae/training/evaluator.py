"""
Evaluation metrics and functionality for TransformerVAE.

This module provides evaluation of molecular generation quality
including validity, uniqueness, novelty, and FCD scores.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict


class TrainingEvaluator:
    """Evaluator for training metrics and molecular generation quality."""

    def __init__(self):
        """Initialize evaluator."""
        self.reset_metrics()

    def reset_metrics(self) -> None:
        """Reset accumulated metrics."""
        self.metrics = defaultdict(list)

    def evaluate_step(self, reconstruction_loss: float, kl_loss: float,
                     beta: float) -> Dict[str, float]:
        """
        Evaluate single training step.

        Args:
            reconstruction_loss: Reconstruction loss value
            kl_loss: KL divergence loss value
            beta: Beta weighting parameter

        Returns:
            Dictionary containing computed metrics
        """
        total_loss = reconstruction_loss + beta * kl_loss

        step_metrics = {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'kl_loss': kl_loss,
            'beta': beta
        }

        # Store metrics
        for key, value in step_metrics.items():
            self.metrics[key].append(value)

        return step_metrics

    def evaluate_batch(self, model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Evaluate model on a batch.

        Args:
            model: Model to evaluate
            batch: Input batch

        Returns:
            Batch evaluation metrics
        """
        model.eval()

        with torch.no_grad():
            outputs = model(batch['sequences'])

            # Compute losses
            if 'reconstruction' in outputs and 'mu' in outputs and 'logvar' in outputs:
                # Reconstruction loss
                reconstruction = outputs['reconstruction']
                targets = batch['sequences']

                if reconstruction.dim() == 3:
                    recon_loss = torch.nn.functional.cross_entropy(
                        reconstruction.view(-1, reconstruction.size(-1)),
                        targets.view(-1),
                        ignore_index=0,
                        reduction='mean'
                    )
                else:
                    recon_loss = torch.nn.functional.mse_loss(reconstruction, targets.float())

                # KL loss
                mu = outputs['mu']
                logvar = outputs['logvar']
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)

                return {
                    'reconstruction_loss': recon_loss.item(),
                    'kl_loss': kl_loss.item()
                }

        return {'reconstruction_loss': 0.0, 'kl_loss': 0.0}

    def get_average_metrics(self) -> Dict[str, float]:
        """Get average of accumulated metrics."""
        avg_metrics = {}
        for key, values in self.metrics.items():
            if values:
                avg_metrics[key] = np.mean(values)
            else:
                avg_metrics[key] = 0.0

        return avg_metrics

    def evaluate_molecular_generation(self, generated_smiles: List[str],
                                    reference_smiles: List[str],
                                    return_details: bool = False) -> Dict[str, float]:
        """
        Evaluate molecular generation quality.

        Args:
            generated_smiles: List of generated SMILES strings
            reference_smiles: List of reference SMILES strings
            return_details: Whether to return detailed metrics

        Returns:
            Dictionary containing molecular generation metrics
        """
        metrics = {}

        # Validity - fraction of valid molecules
        valid_smiles = []
        validity_details = []

        for smiles in generated_smiles:
            is_valid = self._is_valid_smiles(smiles)
            validity_details.append(is_valid)
            if is_valid:
                valid_smiles.append(smiles)

        validity = len(valid_smiles) / len(generated_smiles) if generated_smiles else 0.0
        metrics['validity'] = validity

        # Uniqueness - fraction of unique molecules among valid ones
        if valid_smiles:
            unique_smiles = list(set(valid_smiles))
            uniqueness = len(unique_smiles) / len(valid_smiles)
            metrics['uniqueness'] = uniqueness
        else:
            unique_smiles = []
            metrics['uniqueness'] = 0.0

        # Novelty - fraction of molecules not in reference set
        if unique_smiles and reference_smiles:
            reference_set = set(reference_smiles)
            novel_smiles = [smiles for smiles in unique_smiles if smiles not in reference_set]
            novelty = len(novel_smiles) / len(unique_smiles)
            metrics['novelty'] = novelty
        else:
            metrics['novelty'] = 0.0 if reference_smiles else 1.0

        # Compute FCD score if possible (simplified placeholder)
        try:
            fcd_score = self._compute_fcd_score(valid_smiles, reference_smiles)
            metrics['fcd_score'] = fcd_score
        except Exception:
            metrics['fcd_score'] = float('inf')

        if return_details:
            metrics['valid_smiles'] = valid_smiles
            metrics['unique_smiles'] = unique_smiles
            metrics['validity_details'] = validity_details

        return metrics

    def _is_valid_smiles(self, smiles: str) -> bool:
        """
        Check if SMILES string is valid.

        Args:
            smiles: SMILES string to validate

        Returns:
            True if valid, False otherwise
        """
        if not smiles or len(smiles) == 0:
            return False

        # Basic validity checks
        if any(char in smiles for char in ['<', '>', '|']):
            return False

        try:
            # Try to use RDKit if available
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except ImportError:
            # Fallback to basic checks if RDKit not available
            return self._basic_smiles_validation(smiles)

    def _basic_smiles_validation(self, smiles: str) -> bool:
        """
        Basic SMILES validation without RDKit.

        Args:
            smiles: SMILES string

        Returns:
            True if passes basic checks
        """
        # Check balanced parentheses
        if smiles.count('(') != smiles.count(')'):
            return False

        # Check balanced brackets
        if smiles.count('[') != smiles.count(']'):
            return False

        # Check for valid characters (simplified)
        valid_chars = set('BCNOSPFIKWUVYbcnops()[]=#+-/\\%123456789')
        if not all(c in valid_chars for c in smiles):
            return False

        return True

    def _compute_fcd_score(self, generated_smiles: List[str],
                          reference_smiles: List[str]) -> float:
        """
        Compute Fréchet ChemNet Distance (FCD) score.

        This is a simplified placeholder implementation.
        In practice, you would use the actual FCD implementation.

        Args:
            generated_smiles: Generated molecules
            reference_smiles: Reference molecules

        Returns:
            FCD score (lower is better)
        """
        if not generated_smiles or not reference_smiles:
            return float('inf')

        # Simplified FCD calculation using basic molecular descriptors
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors

            def get_descriptors(smiles_list):
                descriptors = []
                for smiles in smiles_list:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        desc = [
                            Descriptors.MolWt(mol),
                            Descriptors.MolLogP(mol),
                            Descriptors.NumHDonors(mol),
                            Descriptors.NumHAcceptors(mol),
                            Descriptors.TPSA(mol)
                        ]
                        descriptors.append(desc)
                return np.array(descriptors)

            gen_desc = get_descriptors(generated_smiles)
            ref_desc = get_descriptors(reference_smiles)

            if len(gen_desc) == 0 or len(ref_desc) == 0:
                return float('inf')

            # Compute means and covariances
            mu_gen = np.mean(gen_desc, axis=0)
            mu_ref = np.mean(ref_desc, axis=0)

            # Simplified FCD as Euclidean distance between means
            fcd = np.linalg.norm(mu_gen - mu_ref)

            return float(fcd)

        except ImportError:
            # Fallback: use string-based similarity
            gen_lengths = [len(s) for s in generated_smiles]
            ref_lengths = [len(s) for s in reference_smiles]

            gen_mean = np.mean(gen_lengths)
            ref_mean = np.mean(ref_lengths)

            return abs(gen_mean - ref_mean)

    def compute_property_metrics(self, predicted_properties: torch.Tensor,
                               true_properties: torch.Tensor) -> Dict[str, float]:
        """
        Compute molecular property prediction metrics.

        Args:
            predicted_properties: Predicted molecular properties
            true_properties: True molecular properties

        Returns:
            Property prediction metrics
        """
        if predicted_properties.numel() == 0 or true_properties.numel() == 0:
            return {'mse': float('inf'), 'mae': float('inf'), 'r2': 0.0}

        # Mean squared error
        mse = torch.mean((predicted_properties - true_properties) ** 2).item()

        # Mean absolute error
        mae = torch.mean(torch.abs(predicted_properties - true_properties)).item()

        # R-squared (coefficient of determination)
        ss_res = torch.sum((true_properties - predicted_properties) ** 2)
        ss_tot = torch.sum((true_properties - torch.mean(true_properties)) ** 2)

        if ss_tot > 0:
            r2 = 1 - (ss_res / ss_tot).item()
        else:
            r2 = 0.0

        return {
            'mse': mse,
            'mae': mae,
            'r2': r2
        }

    def log_metrics(self, metrics: Dict[str, Any], prefix: str = "") -> None:
        """
        Log metrics to console.

        Args:
            metrics: Dictionary of metrics to log
            prefix: Optional prefix for log messages
        """
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"{prefix}{key}: {value:.4f}")
            else:
                print(f"{prefix}{key}: {value}")

    def save_metrics(self, filepath: str) -> None:
        """
        Save accumulated metrics to file.

        Args:
            filepath: Path to save metrics
        """
        import json

        # Convert numpy arrays to lists for JSON serialization
        serializable_metrics = {}
        for key, values in self.metrics.items():
            if isinstance(values, list):
                serializable_metrics[key] = values
            else:
                serializable_metrics[key] = list(values)

        with open(filepath, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)

        print(f"Metrics saved to {filepath}")