"""
Dataset implementations for molecular data.

This module provides dataset classes for loading and preprocessing
molecular datasets like MOSES and ZINC-15.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from typing import List, Dict, Optional, Tuple, Any
from abc import ABC, abstractmethod
from .tokenizer import SMILESTokenizer


class MolecularDataset(Dataset, ABC):
    """Abstract base class for molecular datasets."""

    def __init__(self, tokenizer: SMILESTokenizer, config: Dict[str, Any]):
        """
        Initialize molecular dataset.

        Args:
            tokenizer: SMILES tokenizer instance
            config: Dataset configuration dictionary
        """
        self.tokenizer = tokenizer
        self.config = config
        self.smiles_data = []
        self.properties_data = None

    @abstractmethod
    def load_data(self) -> None:
        """Load data from source."""
        pass

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.smiles_data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get dataset item.

        Args:
            idx: Item index

        Returns:
            Dictionary containing encoded SMILES and optionally properties
        """
        smiles = self.smiles_data[idx]

        # Encode SMILES
        encoded = self.tokenizer.encode(smiles)

        item = {
            'smiles': smiles,
            'tokens': torch.tensor(encoded, dtype=torch.long),
            'length': torch.tensor(len(encoded), dtype=torch.long)
        }

        # Add properties if available
        if self.properties_data is not None:
            item['properties'] = torch.tensor(
                self.properties_data[idx], dtype=torch.float32
            )

        return item

    def split_dataset(self, train_split: float, val_split: float,
                     test_split: float) -> Tuple['MolecularDataset', 'MolecularDataset', 'MolecularDataset']:
        """
        Split dataset into train/val/test sets.

        Args:
            train_split: Training set proportion
            val_split: Validation set proportion
            test_split: Test set proportion

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        if abs(train_split + val_split + test_split - 1.0) > 1e-6:
            raise ValueError("Splits must sum to 1.0")

        dataset_size = len(self)
        train_size = int(train_split * dataset_size)
        val_size = int(val_split * dataset_size)
        test_size = dataset_size - train_size - val_size

        return random_split(self, [train_size, val_size, test_size])

    def get_reference_molecules(self) -> List[str]:
        """Get reference molecules for evaluation."""
        return self.smiles_data.copy()

    def apply_preprocessing(self, smiles_list: List[str]) -> List[str]:
        """
        Apply preprocessing to SMILES strings.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            Preprocessed SMILES strings
        """
        preprocessing_config = self.config.get('preprocessing_config', {})

        if preprocessing_config.get('canonical', False):
            # Convert to canonical SMILES (requires RDKit)
            try:
                from rdkit import Chem
                canonical_smiles = []
                for smiles in smiles_list:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        canonical_smiles.append(Chem.MolToSmiles(mol, canonical=True))
                return canonical_smiles
            except ImportError:
                print("Warning: RDKit not available for canonicalization")
                return smiles_list

        if preprocessing_config.get('augment_smiles', False):
            # SMILES augmentation by random enumeration
            try:
                from rdkit import Chem
                augmented_smiles = []
                for smiles in smiles_list:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        # Generate random SMILES
                        random_smiles = Chem.MolToSmiles(mol, doRandom=True)
                        augmented_smiles.append(random_smiles)
                    else:
                        augmented_smiles.append(smiles)
                return augmented_smiles
            except ImportError:
                print("Warning: RDKit not available for SMILES augmentation")
                return smiles_list

        return smiles_list

    @classmethod
    def from_config(cls, config: Dict[str, Any], tokenizer: SMILESTokenizer) -> 'MolecularDataset':
        """
        Create dataset from configuration.

        Args:
            config: Dataset configuration
            tokenizer: SMILES tokenizer

        Returns:
            Dataset instance
        """
        dataset_type = config['dataset_type']

        if dataset_type == 'moses':
            return MOSESDataset(tokenizer, config)
        elif dataset_type == 'zinc15':
            return ZINC15Dataset(tokenizer, config)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")


class MOSESDataset(MolecularDataset):
    """MOSES molecular dataset implementation."""

    def __init__(self, tokenizer: SMILESTokenizer, config: Dict[str, Any]):
        super().__init__(tokenizer, config)
        self.load_data()

    def load_data(self) -> None:
        """Load MOSES dataset."""
        data_path = self.config['data_path']

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"MOSES data path not found: {data_path}")

        # Try to load from common MOSES file formats
        if os.path.isfile(data_path):
            # Single file
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
                if 'SMILES' in df.columns:
                    self.smiles_data = df['SMILES'].tolist()
                elif 'smiles' in df.columns:
                    self.smiles_data = df['smiles'].tolist()
                else:
                    # Assume first column is SMILES
                    self.smiles_data = df.iloc[:, 0].tolist()
            else:
                # Assume text file with one SMILES per line
                with open(data_path, 'r') as f:
                    self.smiles_data = [line.strip() for line in f if line.strip()]
        else:
            # Directory with multiple files
            train_file = os.path.join(data_path, 'train.txt')
            if os.path.exists(train_file):
                with open(train_file, 'r') as f:
                    self.smiles_data = [line.strip() for line in f if line.strip()]
            else:
                raise FileNotFoundError(f"No valid MOSES data files found in {data_path}")

        # Apply preprocessing
        self.smiles_data = self.apply_preprocessing(self.smiles_data)

        # Filter by max sequence length if specified
        max_length = self.config.get('max_sequence_length')
        if max_length:
            filtered_smiles = []
            for smiles in self.smiles_data:
                tokens = self.tokenizer._tokenize_smiles(smiles)
                if len(tokens) <= max_length - 2:  # Account for SOS/EOS tokens
                    filtered_smiles.append(smiles)
            self.smiles_data = filtered_smiles

        print(f"Loaded {len(self.smiles_data)} MOSES molecules")


class ZINC15Dataset(MolecularDataset):
    """ZINC-15 molecular dataset implementation."""

    def __init__(self, tokenizer: SMILESTokenizer, config: Dict[str, Any]):
        super().__init__(tokenizer, config)
        self.load_data()

    def load_data(self) -> None:
        """Load ZINC-15 dataset."""
        data_path = self.config['data_path']

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"ZINC-15 data path not found: {data_path}")

        # ZINC-15 typically comes as SDF or SMILES files
        if os.path.isfile(data_path):
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
                if 'SMILES' in df.columns:
                    self.smiles_data = df['SMILES'].tolist()
                elif 'smiles' in df.columns:
                    self.smiles_data = df['smiles'].tolist()
                else:
                    self.smiles_data = df.iloc[:, 0].tolist()

                # Check for molecular properties
                property_columns = [col for col in df.columns
                                  if col.lower() not in ['smiles', 'id', 'name']]
                if property_columns:
                    self.properties_data = df[property_columns].values.astype(float)

            elif data_path.endswith('.smi') or data_path.endswith('.txt'):
                with open(data_path, 'r') as f:
                    lines = [line.strip() for line in f if line.strip()]

                # Handle tab-separated SMILES and properties
                if '\t' in lines[0]:
                    smiles_list = []
                    properties_list = []
                    for line in lines:
                        parts = line.split('\t')
                        smiles_list.append(parts[0])
                        if len(parts) > 1:
                            try:
                                props = [float(p) for p in parts[1:]]
                                properties_list.append(props)
                            except ValueError:
                                pass

                    self.smiles_data = smiles_list
                    if properties_list and len(properties_list) == len(smiles_list):
                        self.properties_data = properties_list
                else:
                    self.smiles_data = lines
        else:
            # Directory with ZINC files
            zinc_files = [f for f in os.listdir(data_path)
                         if f.endswith(('.smi', '.txt', '.csv'))]

            all_smiles = []
            for zinc_file in zinc_files[:5]:  # Limit to first 5 files for demo
                file_path = os.path.join(data_path, zinc_file)
                with open(file_path, 'r') as f:
                    file_smiles = [line.strip().split()[0] for line in f if line.strip()]
                    all_smiles.extend(file_smiles)

            self.smiles_data = all_smiles

        # Apply preprocessing
        self.smiles_data = self.apply_preprocessing(self.smiles_data)

        # Filter by max sequence length
        max_length = self.config.get('max_sequence_length')
        if max_length:
            filtered_data = []
            filtered_properties = [] if self.properties_data is not None else None

            for i, smiles in enumerate(self.smiles_data):
                tokens = self.tokenizer._tokenize_smiles(smiles)
                if len(tokens) <= max_length - 2:
                    filtered_data.append(smiles)
                    if self.properties_data is not None:
                        filtered_properties.append(self.properties_data[i])

            self.smiles_data = filtered_data
            if filtered_properties:
                self.properties_data = filtered_properties

        print(f"Loaded {len(self.smiles_data)} ZINC-15 molecules")
        if self.properties_data is not None:
            print(f"With {len(self.properties_data[0])} molecular properties per molecule")


def collate_molecular_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for molecular dataset batches.

    Args:
        batch: List of dataset items

    Returns:
        Batched data dictionary
    """
    # Get max sequence length in batch
    max_length = max(item['length'].item() for item in batch)

    # Pad sequences
    padded_tokens = []
    lengths = []
    smiles_list = []

    for item in batch:
        tokens = item['tokens']
        length = item['length'].item()

        # Pad sequence
        if length < max_length:
            padding = torch.zeros(max_length - length, dtype=torch.long)
            padded_tokens.append(torch.cat([tokens, padding]))
        else:
            padded_tokens.append(tokens)

        lengths.append(length)
        smiles_list.append(item['smiles'])

    batch_dict = {
        'sequences': torch.stack(padded_tokens),
        'lengths': torch.tensor(lengths, dtype=torch.long),
        'smiles': smiles_list
    }

    # Add properties if available
    if 'properties' in batch[0]:
        properties = torch.stack([item['properties'] for item in batch])
        batch_dict['properties'] = properties

    return batch_dict