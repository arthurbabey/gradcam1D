import numpy as np
import matplotlib.pyplot as plt

from src.regions import find_top_k_important_windows  # noqa: F401


def plot_top_windows(sequence, combined_attribution, top_windows):
    """
    Plot the attribution heatmaps for each selected window.

    Parameters:
    -----------
    sequence : str
        Full DNA sequence.
    combined_attribution : 1D np.array
        Grad-CAM attributions, length = len(sequence).
    top_windows : list of tuples
        (start, end, mean_val) for each window.
    """
    n = len(top_windows)
    fig, axs = plt.subplots(n, 1, figsize=(12, 3*n))
    if n == 1:
        axs = [axs]  # ensure iterable

    for ax, (start, end, mean_val) in zip(axs, top_windows):
        region_seq = sequence[start:end]
        region_attr = combined_attribution[start:end]
        
        heatmap = region_attr[np.newaxis, :]  # shape (1, window_size)
        im = ax.imshow(heatmap, aspect='auto', cmap='coolwarm')
        
        ax.set_xticks(np.arange(len(region_seq)))
        ax.set_xticklabels(list(region_seq), fontsize=6, rotation=90)
        ax.set_yticks([])
        ax.set_title(f"Window {start}-{end}, Mean Attribution={mean_val:.4f}")
        
        # colorbar
        fig.colorbar(im, ax=ax, fraction=0.015, pad=0.1, label='Grad-CAM importance')

    plt.tight_layout()
    plt.show()
    return fig, axs
