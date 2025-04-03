import numpy as np
import matplotlib.pyplot as plt

def find_top_k_important_windows(sequence, 
                                 combined_attribution, 
                                 k=5, 
                                 window_size=500):
    """
    Identify top k non-overlapping windows of fixed size based on mean Grad-CAM attribution.
    
    Parameters:
    -----------
    sequence : str
        The full DNA sequence (length N).
    combined_attribution : 1D np.array
        Grad-CAM attributions of length N.
    k : int
        Number of windows to retrieve.
    window_size : int
        Size of each window (bp).
    
    Returns:
    --------
    top_windows : list of tuples
        Each tuple contains (start_index, end_index, mean_attribution).
    """
    N = len(sequence)
    
    # Safety check
    if len(combined_attribution) != N:
        raise ValueError("Sequence and attribution must have the same length.")
    if window_size > N:
        raise ValueError("window_size cannot exceed the length of the sequence.")

    # 1. Compute mean attribution for each window using convolution
    # sums[i] will be sum of attributions in the window [i, i+window_size-1]
    # means will be sums / window_size
    sums = np.convolve(combined_attribution, np.ones(window_size, dtype=np.float32), mode='valid')
    means = sums / window_size  # shape: (N - window_size + 1,)

    # 2. Find indices that give the highest mean attribution
    # Sort window start indices by descending mean
    sorted_indices = np.argsort(means)[::-1]

    # 3. Pick top k non-overlapping windows
    selected_indices = []
    top_windows = []
    for idx in sorted_indices:
        # window range is [idx, idx + window_size)
        # check overlap with already selected windows
        overlap = False
        for (start_sel, end_sel, _) in top_windows:
            if not (idx + window_size <= start_sel or idx >= end_sel):
                # They overlap
                overlap = True
                break

        if not overlap:
            window_mean = means[idx]
            top_windows.append((idx, idx + window_size, window_mean))

        if len(top_windows) == k:
            break

    # Sort the final windows by their start index
    top_windows.sort(key=lambda x: x[0])
    return top_windows


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
