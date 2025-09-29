"""
Molecular generation metrics computation.

This module provides functions for computing standard molecular generation
metrics including validity, uniqueness, novelty, and FCD scores.
"""

import numpy as np
from typing import List, Dict, Optional, Set
from collections import Counter


def compute_molecular_metrics(generated_smiles: List[str],
                            reference_smiles: List[str],
                            return_details: bool = False) -> Dict[str, float]:
    """
    Compute comprehensive molecular generation metrics.

    Args:
        generated_smiles: List of generated SMILES strings
        reference_smiles: List of reference SMILES strings
        return_details: Whether to return detailed breakdown

    Returns:
        Dictionary containing molecular generation metrics
    """
    if not generated_smiles:
        return {
            'validity': 0.0,
            'uniqueness': 0.0,
            'novelty': 0.0,
            'fcd_score': float('inf'),
            'diversity': 0.0
        }

    metrics = {}

    # Validity - fraction of chemically valid molecules
    valid_smiles = []
    for smiles in generated_smiles:
        if is_valid_smiles(smiles):
            valid_smiles.append(smiles)

    validity = len(valid_smiles) / len(generated_smiles)
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
        metrics['novelty'] = 1.0 if not reference_smiles else 0.0

    # Diversity - internal diversity of generated molecules
    diversity = compute_diversity(unique_smiles)
    metrics['diversity'] = diversity

    # FCD Score - Fréchet ChemNet Distance
    try:
        fcd_score = compute_fcd_score(valid_smiles, reference_smiles)
        metrics['fcd_score'] = fcd_score
    except Exception:
        metrics['fcd_score'] = float('inf')

    # Additional metrics
    if return_details:
        metrics.update({
            'num_generated': len(generated_smiles),
            'num_valid': len(valid_smiles),
            'num_unique': len(unique_smiles),
            'num_novel': len(novel_smiles) if 'novel_smiles' in locals() else 0,
            'valid_smiles': valid_smiles,
            'unique_smiles': unique_smiles
        })

    return metrics


def is_valid_smiles(smiles: str) -> bool:
    """
    Check if SMILES string is chemically valid.

    Args:
        smiles: SMILES string to validate

    Returns:
        True if valid, False otherwise
    """
    if not smiles or len(smiles.strip()) == 0:
        return False

    # Remove special tokens if present
    smiles = smiles.replace('<PAD>', '').replace('<SOS>', '').replace('<EOS>', '').replace('<UNK>', '')
    smiles = smiles.strip()

    if not smiles:
        return False

    try:
        # Try to use RDKit if available
        from rdkit import Chem
        from rdkit.Chem import SanitizeMol

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False

        # Additional sanitization check
        try:
            SanitizeMol(mol)
            return True
        except:
            return False

    except ImportError:
        # Fallback to basic SMILES validation
        return basic_smiles_validation(smiles)


def basic_smiles_validation(smiles: str) -> bool:
    """
    Basic SMILES validation without RDKit.

    Args:
        smiles: SMILES string

    Returns:
        True if passes basic checks
    """
    if not smiles:
        return False

    # Check balanced parentheses
    if smiles.count('(') != smiles.count(')'):
        return False

    # Check balanced brackets
    if smiles.count('[') != smiles.count(']'):
        return False

    # Check for invalid characters
    valid_chars = set('BCNOSPFIKWUVYbcnops()[]=#+-/\\%123456789.@HhRrLlTtAaDdGgEeZzMmXx')
    if not all(c in valid_chars for c in smiles):
        return False

    # Check for reasonable length
    if len(smiles) > 200:
        return False

    return True


def compute_diversity(smiles_list: List[str]) -> float:
    """
    Compute internal diversity of molecule set.

    Args:
        smiles_list: List of SMILES strings

    Returns:
        Average pairwise Tanimoto distance
    """
    if len(smiles_list) < 2:
        return 0.0

    try:
        from rdkit import Chem
        from rdkit.Chem import DataStructs, rdMolDescriptors

        # Convert to fingerprints
        fingerprints = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = rdMolDescriptors.GetMorganFingerprint(mol, 2)
                fingerprints.append(fp)

        if len(fingerprints) < 2:
            return 0.0

        # Compute pairwise similarities
        similarities = []
        for i in range(len(fingerprints)):
            for j in range(i + 1, len(fingerprints)):
                sim = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
                similarities.append(sim)

        # Diversity is 1 - average similarity
        avg_similarity = np.mean(similarities)
        return 1.0 - avg_similarity

    except ImportError:
        # Fallback: string-based diversity
        return string_based_diversity(smiles_list)


def string_based_diversity(smiles_list: List[str]) -> float:
    """
    Compute diversity based on string edit distances.

    Args:
        smiles_list: List of SMILES strings

    Returns:
        Normalized average edit distance
    """
    if len(smiles_list) < 2:
        return 0.0

    def edit_distance(s1: str, s2: str) -> int:
        """Compute Levenshtein distance."""
        if len(s1) < len(s2):
            return edit_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    distances = []
    for i in range(len(smiles_list)):
        for j in range(i + 1, len(smiles_list)):
            dist = edit_distance(smiles_list[i], smiles_list[j])
            max_len = max(len(smiles_list[i]), len(smiles_list[j]))
            normalized_dist = dist / max_len if max_len > 0 else 0
            distances.append(normalized_dist)

    return np.mean(distances) if distances else 0.0


def compute_fcd_score(generated_smiles: List[str], reference_smiles: List[str]) -> float:
    """
    Compute Fréchet ChemNet Distance (FCD) score.

    Args:
        generated_smiles: Generated molecules
        reference_smiles: Reference molecules

    Returns:
        FCD score (lower is better)
    """
    if not generated_smiles or not reference_smiles:
        return float('inf')

    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors

        def compute_descriptors(smiles_list: List[str]) -> np.ndarray:
            """Compute molecular descriptors."""
            descriptors = []

            for smiles in smiles_list:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    desc = [
                        Descriptors.MolWt(mol),
                        Descriptors.MolLogP(mol),
                        Descriptors.NumHDonors(mol),
                        Descriptors.NumHAcceptors(mol),
                        Descriptors.TPSA(mol),
                        Descriptors.NumRotatableBonds(mol),
                        Descriptors.NumAromaticRings(mol),
                        Descriptors.NumSaturatedRings(mol)
                    ]
                    descriptors.append(desc)

            return np.array(descriptors)

        # Compute descriptors
        gen_descriptors = compute_descriptors(generated_smiles)
        ref_descriptors = compute_descriptors(reference_smiles)

        if len(gen_descriptors) == 0 or len(ref_descriptors) == 0:
            return float('inf')

        # Compute means
        mu_gen = np.mean(gen_descriptors, axis=0)
        mu_ref = np.mean(ref_descriptors, axis=0)

        # Compute covariances
        cov_gen = np.cov(gen_descriptors.T)
        cov_ref = np.cov(ref_descriptors.T)

        # Ensure covariance matrices are 2D
        if cov_gen.ndim == 0:
            cov_gen = np.array([[cov_gen]])
        elif cov_gen.ndim == 1:
            cov_gen = np.diag(cov_gen)

        if cov_ref.ndim == 0:
            cov_ref = np.array([[cov_ref]])
        elif cov_ref.ndim == 1:
            cov_ref = np.diag(cov_ref)

        # Simplified FCD calculation (Euclidean distance between means)
        # Full FCD would require matrix square root computation
        fcd = np.linalg.norm(mu_gen - mu_ref)

        # Add trace of covariance difference as penalty
        try:
            cov_penalty = np.trace(cov_gen - cov_ref) ** 2
            fcd += 0.1 * cov_penalty
        except:
            pass

        return float(fcd)

    except ImportError:
        # Fallback: simplified distance based on string statistics
        return simplified_fcd_score(generated_smiles, reference_smiles)


def simplified_fcd_score(generated_smiles: List[str], reference_smiles: List[str]) -> float:
    """
    Simplified FCD score based on string statistics.

    Args:
        generated_smiles: Generated molecules
        reference_smiles: Reference molecules

    Returns:
        Simplified FCD score
    """
    def get_string_stats(smiles_list: List[str]) -> Dict[str, float]:
        """Get basic string statistics."""
        if not smiles_list:
            return {'mean_length': 0, 'char_diversity': 0, 'ring_count': 0}

        lengths = [len(s) for s in smiles_list]
        all_chars = ''.join(smiles_list)
        char_counts = Counter(all_chars)

        ring_counts = [s.count('c') + s.count('C') + s.count('n') + s.count('N') for s in smiles_list]

        return {
            'mean_length': np.mean(lengths),
            'char_diversity': len(char_counts),
            'ring_count': np.mean(ring_counts)
        }

    gen_stats = get_string_stats(generated_smiles)
    ref_stats = get_string_stats(reference_smiles)

    # Compute differences
    length_diff = abs(gen_stats['mean_length'] - ref_stats['mean_length'])
    diversity_diff = abs(gen_stats['char_diversity'] - ref_stats['char_diversity'])
    ring_diff = abs(gen_stats['ring_count'] - ref_stats['ring_count'])

    # Combine into single score
    fcd = length_diff + diversity_diff + ring_diff

    return fcd


def compute_coverage(generated_smiles: List[str], reference_smiles: List[str]) -> float:
    """
    Compute coverage: fraction of reference molecules that are similar to generated ones.

    Args:
        generated_smiles: Generated molecules
        reference_smiles: Reference molecules

    Returns:
        Coverage score
    """
    if not generated_smiles or not reference_smiles:
        return 0.0

    try:
        from rdkit import Chem
        from rdkit.Chem import DataStructs, rdMolDescriptors

        # Convert to fingerprints
        gen_fps = []
        for smiles in generated_smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = rdMolDescriptors.GetMorganFingerprint(mol, 2)
                gen_fps.append(fp)

        ref_fps = []
        for smiles in reference_smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = rdMolDescriptors.GetMorganFingerprint(mol, 2)
                ref_fps.append(fp)

        if not gen_fps or not ref_fps:
            return 0.0

        # For each reference molecule, check if any generated molecule is similar
        covered = 0
        threshold = 0.7  # Similarity threshold

        for ref_fp in ref_fps:
            max_sim = 0.0
            for gen_fp in gen_fps:
                sim = DataStructs.TanimotoSimilarity(ref_fp, gen_fp)
                max_sim = max(max_sim, sim)

            if max_sim >= threshold:
                covered += 1

        return covered / len(ref_fps)

    except ImportError:
        # Fallback: exact string matching
        gen_set = set(generated_smiles)
        ref_set = set(reference_smiles)
        return len(gen_set.intersection(ref_set)) / len(ref_set)


def molecular_metrics_summary(metrics: Dict[str, float]) -> str:
    """
    Create a formatted summary of molecular metrics.

    Args:
        metrics: Dictionary of computed metrics

    Returns:
        Formatted summary string
    """
    summary = "Molecular Generation Metrics:\n"
    summary += "=" * 30 + "\n"

    metric_descriptions = {
        'validity': 'Validity (valid SMILES)',
        'uniqueness': 'Uniqueness (among valid)',
        'novelty': 'Novelty (not in reference)',
        'diversity': 'Diversity (internal)',
        'fcd_score': 'FCD Score (lower is better)',
        'coverage': 'Coverage (reference similarity)'
    }

    for key, value in metrics.items():
        if key in metric_descriptions:
            if key == 'fcd_score':
                summary += f"{metric_descriptions[key]}: {value:.2f}\n"
            else:
                summary += f"{metric_descriptions[key]}: {value:.3f}\n"

    return summary