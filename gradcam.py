import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd


class GradCAM:
    """Grad-CAM for dual-branch 1D CNN model (phage-bacteria interaction)."""
    def __init__(self, model, target_layer_bacteria=None, target_layer_phage=None):
        self.model = model.eval()  # put model in evaluation mode
        # Identify target layers (last conv layers in each branch)
        if target_layer_bacteria is None:
            target_layer_bacteria = model.bacteria_branch.conv3
        if target_layer_phage is None:
            target_layer_phage = model.phage_branch.conv2
        self.target_layer_bacteria = target_layer_bacteria
        self.target_layer_phage = target_layer_phage

        # Placeholders for features and gradients
        self.bact_features = None
        self.phage_features = None

        # Forward hooks to capture feature maps
        target_layer_bacteria.register_forward_hook(self._save_bact_feature)
        target_layer_phage.register_forward_hook(self._save_phage_feature)
        # We will use .retain_grad() on feature maps to get gradients during backward

    def _save_bact_feature(self, module, input, output):
        """Forward hook: store bacterial conv feature map and enable gradient retention."""
        self.bact_features = output  # feature map from conv3
        output.retain_grad()        # keep gradient for this tensor

    def _save_phage_feature(self, module, input, output):
        """Forward hook: store phage conv feature map and enable gradient retention."""
        self.phage_features = output  # feature map from conv2
        output.retain_grad()

    def generate(self, bact_input, phage_input):
        """
        Compute Grad-CAM heatmaps for the given inputs.
        Inputs should be tensors (1 x 4 x length) for each branch.
        Returns:
            cam_bact (numpy 1D array of length ~279) – importance map for bacterium sequence.
            cam_phage (numpy 1D array of length ~199) – importance map for phage sequence.
        """
        # Ensure model is in eval and gradients are zeroed
        self.model.eval()
        self.model.zero_grad()

        # Forward pass through the model
        output = self.model(bact_input, phage_input)
        # We assume a single input (batch size 1). If batch>1, pick the first instance or specify index.
        target_score = output.squeeze()  # scalar prediction for interaction probability
        # Backward pass to compute gradients of target_score w.r.t. feature maps
        target_score.backward()

        # Get the gradients of the conv feature maps
        grad_bact = self.bact_features.grad  # shape: (1, channels, L_bact_feature)
        grad_phage = self.phage_features.grad  # shape: (1, channels, L_phage_feature)

        # Compute channel-wise weights: global average pooling of gradients over the length dimension
        # This yields a weight for each channel (filter) of the conv layer
        weights_bact = grad_bact.mean(dim=2, keepdim=True)  # shape: (1, channels, 1)
        weights_phage = grad_phage.mean(dim=2, keepdim=True)

        # Weight the feature maps by these importance weights and sum over channels
        cam_bact = (weights_bact * self.bact_features).sum(dim=1)  # shape: (1, L_bact_feature)
        cam_phage = (weights_phage * self.phage_features).sum(dim=1)  # shape: (1, L_phage_feature)

        # Apply ReLU to the weighted maps to keep only positive contributions
        cam_bact = F.relu(cam_bact)
        cam_phage = F.relu(cam_phage)

        # Remove batch dimension and normalize the heatmaps to [0, 1]
        cam_bact = cam_bact.detach().cpu().numpy()[0]  # shape (L_bact_feature,)
        cam_phage = cam_phage.detach().cpu().numpy()[0]
        if cam_bact.max() != 0:
            cam_bact = cam_bact / cam_bact.max()
        if cam_phage.max() != 0:
            cam_phage = cam_phage / cam_phage.max()

        return cam_bact, cam_phage


def plot_sequence_gradcam(sequence, importance_scores, start=None, end=None, window=100, cmap='coolwarm'):
    """
    Plot a segment of the sequence with Grad-CAM importance scores overlaid as a heatmap.
    - sequence: DNA sequence string (e.g., "ACGT...") 
    - importance_scores: 1D numpy array of Grad-CAM scores (normalized 0 to 1) corresponding to sequence positions.
    - start, end: optional indices to specify the sequence region to plot. If None, will focus on top scoring region.
    - window: if start/end not provided, the number of bases around the top score to display.
    - cmap: colormap for the heatmap (default 'coolwarm').
    """
    seq_len = len(sequence)
    # Determine region to visualize
    if start is None or end is None:
        # Find the position of maximum importance and center the window around it
        max_idx = int(np.argmax(importance_scores))
        half_win = window // 2
        start = max(0, max_idx - half_win)
        end = min(seq_len, max_idx + half_win)
    # Extract the region of interest
    seq_region = sequence[start:end]
    scores_region = importance_scores[start:end]

    # Create heatmap image (1 x region_length) using importance scores
    fig, ax = plt.subplots(figsize=(max(10, 0.2 * len(seq_region)), 2))
    # Reshape scores to (1, L) for imshow
    heatmap = scores_region[np.newaxis, :]  # shape (1, L_region)
    im = ax.imshow(heatmap, aspect='auto', cmap=cmap, vmin=0, vmax=1)
    # Set nucleotide labels on the x-axis
    ax.set_xticks(np.arange(len(seq_region)))
    ax.set_xticklabels(list(seq_region))
    ax.set_yticks([])  # hide y-axis
    ax.set_xlabel(f"Sequence positions {start} to {end-1}")
    # Rotate labels if the region is long for better visibility
    if len(seq_region) > 20:
        plt.setp(ax.get_xticklabels(), rotation=90, fontsize=8)
    # Add a color bar to show the importance scale
    plt.colorbar(im, ax=ax, fraction=0.015, pad=0.1, label='Grad-CAM importance')
    plt.title("Grad-CAM highlight on sequence segment")
    plt.tight_layout()
    plt.show()
    return fig, ax
