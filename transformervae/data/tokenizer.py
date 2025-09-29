"""
SMILES tokenization functionality for molecular data.

This module provides tokenization of SMILES strings for use with
transformer models.
"""

import re
from typing import List, Dict, Optional, Union
import torch


class SMILESTokenizer:
    """Tokenizer for SMILES molecular representations."""

    def __init__(self, vocab_size: int = 1000, max_length: Optional[int] = None):
        """
        Initialize SMILES tokenizer.

        Args:
            vocab_size: Maximum vocabulary size
            max_length: Maximum sequence length (optional)
        """
        self.vocab_size = vocab_size
        self.max_length = max_length

        # Special tokens
        self.pad_token = "<PAD>"
        self.sos_token = "<SOS>"
        self.eos_token = "<EOS>"
        self.unk_token = "<UNK>"

        # Token to ID mapping
        self.token_to_id = {
            self.pad_token: 0,
            self.sos_token: 1,
            self.eos_token: 2,
            self.unk_token: 3,
        }

        # ID to token mapping
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

        # SMILES atom and bond patterns
        self.smiles_pattern = self._create_smiles_pattern()

        # Vocabulary built flag
        self._vocab_built = False

    def _create_smiles_pattern(self) -> re.Pattern:
        """Create regex pattern for SMILES tokenization."""
        # SMILES tokenization pattern
        pattern = r"""
            Cl|Br|                          # Two-character elements
            [BCNOSPFIKWUVYbcnops]|         # Single-character elements
            \[[^\]]+\]|                     # Bracketed atoms
            [()=+\-#@/\\%]|                # Bonds and structural elements
            [0-9]                          # Numbers
        """
        return re.compile(pattern, re.VERBOSE)

    def build_vocab(self, smiles_list: List[str]) -> None:
        """
        Build vocabulary from SMILES strings.

        Args:
            smiles_list: List of SMILES strings to build vocabulary from
        """
        # Tokenize all SMILES
        token_counts = {}
        for smiles in smiles_list:
            tokens = self._tokenize_smiles(smiles)
            for token in tokens:
                token_counts[token] = token_counts.get(token, 0) + 1

        # Sort tokens by frequency
        sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)

        # Add tokens to vocabulary (keeping special tokens)
        current_id = len(self.token_to_id)
        for token, count in sorted_tokens:
            if token not in self.token_to_id and current_id < self.vocab_size:
                self.token_to_id[token] = current_id
                self.id_to_token[current_id] = token
                current_id += 1

        self._vocab_built = True

    def _tokenize_smiles(self, smiles: str) -> List[str]:
        """Tokenize a single SMILES string."""
        return self.smiles_pattern.findall(smiles)

    def encode(self, smiles: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode SMILES string to token IDs.

        Args:
            smiles: SMILES string to encode
            add_special_tokens: Whether to add SOS/EOS tokens

        Returns:
            List of token IDs
        """
        if not self._vocab_built:
            raise ValueError("Vocabulary not built. Call build_vocab() first.")

        tokens = self._tokenize_smiles(smiles)

        # Convert tokens to IDs
        token_ids = []
        if add_special_tokens:
            token_ids.append(self.token_to_id[self.sos_token])

        for token in tokens:
            if token in self.token_to_id:
                token_ids.append(self.token_to_id[token])
            else:
                token_ids.append(self.token_to_id[self.unk_token])

        if add_special_tokens:
            token_ids.append(self.token_to_id[self.eos_token])

        # Truncate if necessary
        if self.max_length is not None and len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length - 1] + [self.token_to_id[self.eos_token]]

        return token_ids

    def encode_batch(self, smiles_list: List[str], add_special_tokens: bool = True,
                     padding: bool = True) -> torch.Tensor:
        """
        Encode batch of SMILES strings.

        Args:
            smiles_list: List of SMILES strings
            add_special_tokens: Whether to add SOS/EOS tokens
            padding: Whether to pad sequences to same length

        Returns:
            Tensor of token IDs with shape (batch_size, seq_len)
        """
        encoded_sequences = [self.encode(smiles, add_special_tokens) for smiles in smiles_list]

        if padding:
            # Pad sequences to same length
            max_len = max(len(seq) for seq in encoded_sequences)
            if self.max_length is not None:
                max_len = min(max_len, self.max_length)

            padded_sequences = []
            for seq in encoded_sequences:
                if len(seq) > max_len:
                    seq = seq[:max_len]
                elif len(seq) < max_len:
                    seq = seq + [self.token_to_id[self.pad_token]] * (max_len - len(seq))
                padded_sequences.append(seq)

            return torch.tensor(padded_sequences, dtype=torch.long)
        else:
            return [torch.tensor(seq, dtype=torch.long) for seq in encoded_sequences]

    def decode(self, token_ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to SMILES string.

        Args:
            token_ids: List or tensor of token IDs
            skip_special_tokens: Whether to skip special tokens in output

        Returns:
            Decoded SMILES string
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if skip_special_tokens and token in [self.pad_token, self.sos_token, self.eos_token]:
                    continue
                tokens.append(token)

        return "".join(tokens)

    def decode_batch(self, token_ids_batch: torch.Tensor, skip_special_tokens: bool = True) -> List[str]:
        """
        Decode batch of token ID sequences.

        Args:
            token_ids_batch: Tensor of shape (batch_size, seq_len)
            skip_special_tokens: Whether to skip special tokens

        Returns:
            List of decoded SMILES strings
        """
        return [self.decode(seq, skip_special_tokens) for seq in token_ids_batch]

    def get_vocab_size(self) -> int:
        """Get actual vocabulary size."""
        return len(self.token_to_id)

    def get_special_token_ids(self) -> Dict[str, int]:
        """Get special token IDs."""
        return {
            "pad_token_id": self.token_to_id[self.pad_token],
            "sos_token_id": self.token_to_id[self.sos_token],
            "eos_token_id": self.token_to_id[self.eos_token],
            "unk_token_id": self.token_to_id[self.unk_token],
        }

    def save_vocab(self, path: str) -> None:
        """Save vocabulary to file."""
        import json
        vocab_data = {
            "token_to_id": self.token_to_id,
            "vocab_size": self.vocab_size,
            "max_length": self.max_length,
        }
        with open(path, 'w') as f:
            json.dump(vocab_data, f, indent=2)

    def load_vocab(self, path: str) -> None:
        """Load vocabulary from file."""
        import json
        with open(path, 'r') as f:
            vocab_data = json.load(f)

        self.token_to_id = vocab_data["token_to_id"]
        self.id_to_token = {int(k): v for v, k in self.token_to_id.items()}
        self.vocab_size = vocab_data["vocab_size"]
        self.max_length = vocab_data.get("max_length")
        self._vocab_built = True