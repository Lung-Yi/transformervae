"""
Visualization utilities for TransformerVAE training and results.

This module provides functions for plotting training curves, molecular metrics,
and other visualizations to help with model analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import seaborn as sns


def plot_training_curves(train_metrics: List[Dict[str, float]],
                        val_metrics: Optional[List[Dict[str, float]]] = None,
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot training and validation curves.

    Args:
        train_metrics: List of training metrics per epoch
        val_metrics: Optional list of validation metrics per epoch
        save_path: Optional path to save the plot

    Returns:
        Matplotlib figure
    """
    # Set style
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Training Progress', fontsize=16)

    # Extract epochs
    epochs = list(range(1, len(train_metrics) + 1))

    # Plot total loss
    ax = axes[0, 0]
    train_losses = [m.get('total_loss', 0) for m in train_metrics]
    ax.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)

    if val_metrics:
        val_epochs = list(range(1, len(val_metrics) + 1))
        val_losses = [m.get('total_loss', 0) for m in val_metrics]
        ax.plot(val_epochs, val_losses, 'r-', label='Val Loss', linewidth=2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot reconstruction loss
    ax = axes[0, 1]
    recon_losses = [m.get('recon_loss', 0) for m in train_metrics]
    ax.plot(epochs, recon_losses, 'g-', label='Train Recon Loss', linewidth=2)

    if val_metrics:
        val_recon_losses = [m.get('recon_loss', 0) for m in val_metrics]
        ax.plot(val_epochs, val_recon_losses, 'orange', label='Val Recon Loss', linewidth=2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Reconstruction Loss')
    ax.set_title('Reconstruction Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot KL loss
    ax = axes[1, 0]
    kl_losses = [m.get('kl_loss', 0) for m in train_metrics]
    ax.plot(epochs, kl_losses, 'purple', label='Train KL Loss', linewidth=2)

    if val_metrics:
        val_kl_losses = [m.get('kl_loss', 0) for m in val_metrics]
        ax.plot(val_epochs, val_kl_losses, 'brown', label='Val KL Loss', linewidth=2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('KL Divergence Loss')
    ax.set_title('KL Divergence Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot learning rate if available
    ax = axes[1, 1]
    if 'learning_rate' in train_metrics[0]:
        lrs = [m.get('learning_rate', 0) for m in train_metrics]
        ax.plot(epochs, lrs, 'c-', label='Learning Rate', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    else:
        # If no learning rate data, plot loss ratio
        if recon_losses and kl_losses:
            ratios = [r / (k + 1e-8) for r, k in zip(recon_losses, kl_losses)]
            ax.plot(epochs, ratios, 'm-', label='Recon/KL Ratio', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Reconstruction/KL Ratio')
            ax.set_title('Loss Component Ratio')
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")

    return fig


def plot_molecular_metrics(metrics_history: List[Dict[str, float]],
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot molecular generation metrics over time.

    Args:
        metrics_history: List of molecular metrics per evaluation
        save_path: Optional path to save the plot

    Returns:
        Matplotlib figure
    """
    if not metrics_history:
        print("No metrics to plot")
        return None

    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Molecular Generation Metrics', fontsize=16)

    epochs = list(range(1, len(metrics_history) + 1))

    # Validity
    ax = axes[0, 0]
    validity = [m.get('validity', 0) for m in metrics_history]
    ax.plot(epochs, validity, 'g-o', linewidth=2, markersize=4)
    ax.set_xlabel('Evaluation')
    ax.set_ylabel('Validity')
    ax.set_title('Molecular Validity')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Uniqueness
    ax = axes[0, 1]
    uniqueness = [m.get('uniqueness', 0) for m in metrics_history]
    ax.plot(epochs, uniqueness, 'b-o', linewidth=2, markersize=4)
    ax.set_xlabel('Evaluation')
    ax.set_ylabel('Uniqueness')
    ax.set_title('Molecular Uniqueness')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Novelty
    ax = axes[1, 0]
    novelty = [m.get('novelty', 0) for m in metrics_history]
    ax.plot(epochs, novelty, 'r-o', linewidth=2, markersize=4)
    ax.set_xlabel('Evaluation')
    ax.set_ylabel('Novelty')
    ax.set_title('Molecular Novelty')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # FCD Score
    ax = axes[1, 1]
    fcd_scores = [m.get('fcd_score', float('inf')) for m in metrics_history]
    # Filter out infinite values for plotting
    finite_fcd = [(i, score) for i, score in enumerate(fcd_scores) if np.isfinite(score)]

    if finite_fcd:
        finite_epochs, finite_scores = zip(*finite_fcd)
        ax.plot([e + 1 for e in finite_epochs], finite_scores, 'purple', marker='o', linewidth=2, markersize=4)

    ax.set_xlabel('Evaluation')
    ax.set_ylabel('FCD Score')
    ax.set_title('FCD Score (Lower is Better)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Molecular metrics plot saved to {save_path}")

    return fig


def plot_latent_space(latent_representations: np.ndarray,
                     labels: Optional[List[str]] = None,
                     method: str = 'tsne',
                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot 2D visualization of latent space.

    Args:
        latent_representations: Array of latent vectors (n_samples, latent_dim)
        labels: Optional labels for coloring points
        method: Dimensionality reduction method ('tsne', 'pca', 'umap')
        save_path: Optional path to save the plot

    Returns:
        Matplotlib figure
    """
    if latent_representations.shape[1] == 2:
        # Already 2D
        embedded = latent_representations
    else:
        # Reduce dimensionality
        if method == 'tsne':
            from sklearn.manifold import TSNE
            embedder = TSNE(n_components=2, random_state=42)
        elif method == 'pca':
            from sklearn.decomposition import PCA
            embedder = PCA(n_components=2)
        elif method == 'umap':
            try:
                import umap
                embedder = umap.UMAP(n_components=2, random_state=42)
            except ImportError:
                print("UMAP not available, falling back to t-SNE")
                from sklearn.manifold import TSNE
                embedder = TSNE(n_components=2, random_state=42)
        else:
            raise ValueError(f"Unknown method: {method}")

        embedded = embedder.fit_transform(latent_representations)

    # Create plot
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')

    fig, ax = plt.subplots(figsize=(10, 8))

    if labels is not None:
        # Color by labels
        unique_labels = list(set(labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            ax.scatter(embedded[mask, 0], embedded[mask, 1],
                      c=[colors[i]], label=label, alpha=0.7, s=20)

        ax.legend()
    else:
        # Single color
        ax.scatter(embedded[:, 0], embedded[:, 1], alpha=0.7, s=20)

    ax.set_xlabel(f'{method.upper()} Component 1')
    ax.set_ylabel(f'{method.upper()} Component 2')
    ax.set_title(f'Latent Space Visualization ({method.upper()})')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Latent space plot saved to {save_path}")

    return fig


def plot_molecular_properties(properties: np.ndarray,
                             property_names: List[str],
                             generated_properties: Optional[np.ndarray] = None,
                             save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot distribution of molecular properties.

    Args:
        properties: Array of molecular properties (n_molecules, n_properties)
        property_names: Names of the properties
        generated_properties: Optional generated properties for comparison
        save_path: Optional path to save the plot

    Returns:
        Matplotlib figure
    """
    n_properties = len(property_names)
    n_cols = min(3, n_properties)
    n_rows = (n_properties + n_cols - 1) // n_cols

    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    for i, prop_name in enumerate(property_names):
        ax = axes[i] if i < len(axes) else plt.subplot(n_rows, n_cols, i + 1)

        # Plot reference distribution
        ax.hist(properties[:, i], bins=30, alpha=0.7, label='Reference', density=True)

        # Plot generated distribution if provided
        if generated_properties is not None:
            ax.hist(generated_properties[:, i], bins=30, alpha=0.7,
                   label='Generated', density=True)

        ax.set_xlabel(prop_name)
        ax.set_ylabel('Density')
        ax.set_title(f'{prop_name} Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_properties, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Molecular properties plot saved to {save_path}")

    return fig


def plot_loss_components(loss_history: List[Dict[str, float]],
                        beta_values: Optional[List[float]] = None,
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot detailed analysis of loss components.

    Args:
        loss_history: List of loss dictionaries
        beta_values: Optional beta values used during training
        save_path: Optional path to save the plot

    Returns:
        Matplotlib figure
    """
    if not loss_history:
        print("No loss history to plot")
        return None

    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Loss Component Analysis', fontsize=16)

    epochs = list(range(1, len(loss_history) + 1))

    # Raw loss components
    ax = axes[0, 0]
    recon_losses = [h.get('recon_loss', 0) for h in loss_history]
    kl_losses = [h.get('kl_loss', 0) for h in loss_history]

    ax.plot(epochs, recon_losses, 'g-', label='Reconstruction Loss', linewidth=2)
    ax.plot(epochs, kl_losses, 'r-', label='KL Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss Value')
    ax.set_title('Raw Loss Components')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Weighted KL loss
    ax = axes[0, 1]
    if beta_values:
        weighted_kl = [beta * kl for beta, kl in zip(beta_values, kl_losses)]
        ax.plot(epochs, weighted_kl, 'purple', label='β * KL Loss', linewidth=2)
        ax.plot(epochs, recon_losses, 'g-', label='Reconstruction Loss', linewidth=2)
    else:
        # Assume constant beta from first entry
        beta = loss_history[0].get('beta', 1.0)
        weighted_kl = [beta * kl for kl in kl_losses]
        ax.plot(epochs, weighted_kl, 'purple', label=f'β({beta}) * KL Loss', linewidth=2)
        ax.plot(epochs, recon_losses, 'g-', label='Reconstruction Loss', linewidth=2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss Value')
    ax.set_title('Weighted Loss Components')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Loss ratios
    ax = axes[1, 0]
    ratios = [r / (k + 1e-8) for r, k in zip(recon_losses, kl_losses)]
    ax.plot(epochs, ratios, 'm-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Ratio')
    ax.set_title('Reconstruction/KL Loss Ratio')
    ax.grid(True, alpha=0.3)

    # Cumulative losses
    ax = axes[1, 1]
    cum_recon = np.cumsum(recon_losses)
    cum_kl = np.cumsum(kl_losses)

    ax.plot(epochs, cum_recon, 'g-', label='Cumulative Reconstruction', linewidth=2)
    ax.plot(epochs, cum_kl, 'r-', label='Cumulative KL', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cumulative Loss')
    ax.set_title('Cumulative Loss Components')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Loss components plot saved to {save_path}")

    return fig


def create_training_dashboard(train_metrics: List[Dict[str, float]],
                            val_metrics: Optional[List[Dict[str, float]]] = None,
                            molecular_metrics: Optional[List[Dict[str, float]]] = None,
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a comprehensive training dashboard.

    Args:
        train_metrics: Training metrics per epoch
        val_metrics: Validation metrics per epoch
        molecular_metrics: Molecular generation metrics
        save_path: Optional path to save the plot

    Returns:
        Matplotlib figure
    """
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Training loss
    ax1 = fig.add_subplot(gs[0, 0])
    epochs = list(range(1, len(train_metrics) + 1))
    train_losses = [m.get('total_loss', 0) for m in train_metrics]
    ax1.plot(epochs, train_losses, 'b-', linewidth=2, label='Train')

    if val_metrics:
        val_epochs = list(range(1, len(val_metrics) + 1))
        val_losses = [m.get('total_loss', 0) for m in val_metrics]
        ax1.plot(val_epochs, val_losses, 'r-', linewidth=2, label='Val')

    ax1.set_title('Total Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss components
    ax2 = fig.add_subplot(gs[0, 1])
    recon_losses = [m.get('recon_loss', 0) for m in train_metrics]
    kl_losses = [m.get('kl_loss', 0) for m in train_metrics]
    ax2.plot(epochs, recon_losses, 'g-', linewidth=2, label='Reconstruction')
    ax2.plot(epochs, kl_losses, 'purple', linewidth=2, label='KL Divergence')
    ax2.set_title('Loss Components')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    # Molecular metrics (if available)
    if molecular_metrics:
        # Validity
        ax3 = fig.add_subplot(gs[0, 2])
        validity = [m.get('validity', 0) for m in molecular_metrics]
        eval_epochs = list(range(1, len(molecular_metrics) + 1))
        ax3.plot(eval_epochs, validity, 'g-o', linewidth=2)
        ax3.set_title('Molecular Validity')
        ax3.set_xlabel('Evaluation')
        ax3.set_ylabel('Validity')
        ax3.set_ylim(0, 1.05)
        ax3.grid(True, alpha=0.3)

        # Uniqueness and Novelty
        ax4 = fig.add_subplot(gs[1, 0])
        uniqueness = [m.get('uniqueness', 0) for m in molecular_metrics]
        novelty = [m.get('novelty', 0) for m in molecular_metrics]
        ax4.plot(eval_epochs, uniqueness, 'b-o', linewidth=2, label='Uniqueness')
        ax4.plot(eval_epochs, novelty, 'r-o', linewidth=2, label='Novelty')
        ax4.set_title('Uniqueness & Novelty')
        ax4.set_xlabel('Evaluation')
        ax4.set_ylabel('Score')
        ax4.set_ylim(0, 1.05)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # FCD Score
        ax5 = fig.add_subplot(gs[1, 1])
        fcd_scores = [m.get('fcd_score', float('inf')) for m in molecular_metrics]
        finite_fcd = [(i, score) for i, score in enumerate(fcd_scores) if np.isfinite(score)]
        if finite_fcd:
            finite_epochs, finite_scores = zip(*finite_fcd)
            ax5.plot([e + 1 for e in finite_epochs], finite_scores, 'purple', marker='o', linewidth=2)
        ax5.set_title('FCD Score')
        ax5.set_xlabel('Evaluation')
        ax5.set_ylabel('FCD Score')
        ax5.grid(True, alpha=0.3)

    # Loss ratio
    ax6 = fig.add_subplot(gs[1, 2])
    if recon_losses and kl_losses:
        ratios = [r / (k + 1e-8) for r, k in zip(recon_losses, kl_losses)]
        ax6.plot(epochs, ratios, 'm-', linewidth=2)
    ax6.set_title('Recon/KL Ratio')
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Ratio')
    ax6.grid(True, alpha=0.3)

    # Training progress
    ax7 = fig.add_subplot(gs[2, :])
    ax7.text(0.5, 0.5, f'Training Progress Summary\n\n'
                      f'Total Epochs: {len(train_metrics)}\n'
                      f'Final Train Loss: {train_losses[-1]:.4f}\n'
                      f'Final Val Loss: {val_losses[-1]:.4f if val_metrics else "N/A"}\n'
                      f'Best Val Loss: {min(val_losses) if val_metrics else "N/A"}\n'
                      f'Final Validity: {validity[-1]:.3f if molecular_metrics else "N/A"}',
             ha='center', va='center', fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    ax7.set_xlim(0, 1)
    ax7.set_ylim(0, 1)
    ax7.axis('off')

    plt.suptitle('TransformerVAE Training Dashboard', fontsize=16)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training dashboard saved to {save_path}")

    return fig


def save_all_plots(output_dir: str, **plot_data) -> None:
    """
    Save all available plots to output directory.

    Args:
        output_dir: Directory to save plots
        **plot_data: Keyword arguments containing plot data
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    if 'train_metrics' in plot_data:
        plot_training_curves(
            plot_data['train_metrics'],
            plot_data.get('val_metrics'),
            save_path=os.path.join(output_dir, 'training_curves.png')
        )

    if 'molecular_metrics' in plot_data:
        plot_molecular_metrics(
            plot_data['molecular_metrics'],
            save_path=os.path.join(output_dir, 'molecular_metrics.png')
        )

    if 'loss_history' in plot_data:
        plot_loss_components(
            plot_data['loss_history'],
            plot_data.get('beta_values'),
            save_path=os.path.join(output_dir, 'loss_components.png')
        )

    # Training dashboard
    if 'train_metrics' in plot_data:
        create_training_dashboard(
            plot_data['train_metrics'],
            plot_data.get('val_metrics'),
            plot_data.get('molecular_metrics'),
            save_path=os.path.join(output_dir, 'training_dashboard.png')
        )

    print(f"All plots saved to {output_dir}")