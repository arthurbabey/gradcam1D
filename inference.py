import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import os

# Constants for fixed input lengths
BACTERIUM_THRESHOLD = 7000000  # padded length for bacterial sequences
PHAGE_THRESHOLD = 200000       # padded length for phage sequences

# One-hot encoding map for IUPAC nucleotides (same mapping as in training)
onehot_map = {
    "A": np.array([1, 0, 0, 0], dtype=np.uint8),
    "C": np.array([0, 1, 0, 0], dtype=np.uint8),
    "G": np.array([0, 0, 1, 0], dtype=np.uint8),
    "T": np.array([0, 0, 0, 1], dtype=np.uint8),
    "R": np.array([1, 0, 1, 0], dtype=np.uint8),
    "Y": np.array([0, 1, 0, 1], dtype=np.uint8),
    "K": np.array([0, 0, 1, 1], dtype=np.uint8),
    "M": np.array([1, 1, 0, 0], dtype=np.uint8),
    "S": np.array([0, 1, 1, 0], dtype=np.uint8),
    "W": np.array([1, 0, 0, 1], dtype=np.uint8),
    "B": np.array([0, 1, 1, 1], dtype=np.uint8),
    "D": np.array([1, 0, 1, 1], dtype=np.uint8),
    "H": np.array([1, 1, 0, 1], dtype=np.uint8),
    "V": np.array([1, 1, 1, 0], dtype=np.uint8),
    "N": np.array([1, 1, 1, 1], dtype=np.uint8),
    "Z": np.array([0, 0, 0, 0], dtype=np.uint8)  # padding or gap
}

inv_onehot_map = {tuple(v.tolist()): k for k, v in onehot_map.items()}


# Precompute mapping from ASCII code to one-hot vector for fast encoding
ascii_to_onehot = np.zeros((256, 4), dtype=np.uint8)
for base, vec in onehot_map.items():
    ascii_to_onehot[ord(base)] = vec

def encode_sequence_onehot(seq: str, target_length: int) -> np.ndarray:
    """
    One-hot encode a DNA sequence and pad or truncate it to the target length.
    Returns a numpy array of shape (4, target_length) with dtype uint8.
    """
    seq_bytes = seq.upper().encode('ascii', 'ignore')
    seq_codes = np.frombuffer(seq_bytes, dtype=np.uint8)
    onehot_seq = ascii_to_onehot[seq_codes]  # shape (len(seq), 4)
    if onehot_seq.shape[0] >= target_length:
        onehot_padded = onehot_seq[:target_length, :]
    else:
        onehot_padded = np.zeros((target_length, 4), dtype=np.uint8)
        onehot_padded[:onehot_seq.shape[0], :] = onehot_seq
    # Transpose to (4, L) for PyTorch (channels-first format)
    return onehot_padded.T

def tensor_to_sequence(onehot_tensor: np.ndarray):
    """
    Convert a one-hot encoded (4,L) or (N,4,L) array back to list of strings
    based on `inv_map`. If it's just (4,L), returns a list of length 1.
    """
    arr = onehot_tensor

    # Handle shape (4, L) => expand to (1,4,L)
    if arr.ndim == 2 and arr.shape[0] == 4:
        arr = np.expand_dims(arr, axis=0)

    # By now we expect shape (N,4,L).
    N, C, L = arr.shape
    if C != 4:
        raise ValueError(f"Expected second dimension=4, got {C}")

    decoded_seqs = []
    for i in range(N):
        # subarr = shape (4,L) => subarr.T => (L,4)
        subarr = arr[i].T
        chars = []
        for row in subarr:
            row_tuple = tuple(row.tolist())
            chars.append(inv_onehot_map.get(row_tuple, "N"))  # default 'N' if not found
        decoded_seqs.append("".join(chars))
    return decoded_seqs[0]

def precompute_and_cache_sequences(phages_df: pd.DataFrame, bacteria_df: pd.DataFrame, cache_dir: str):
    """
    Precompute one-hot encoded & padded sequences for all phages and bacteria, 
    and save them to .npy files. Returns dictionaries mapping IDs to file paths.
    """
    os.makedirs(cache_dir, exist_ok=True)
    phage_cache = os.path.join(cache_dir, "phage")
    bacteria_cache = os.path.join(cache_dir, "bacteria")
    os.makedirs(phage_cache, exist_ok=True)
    os.makedirs(bacteria_cache, exist_ok=True)
    phage_id_to_path = {}
    bacterium_id_to_path = {}
    # Precompute phage sequences
    for _, row in phages_df.iterrows():
        pid = row['phage_id']
        seq = row['phage_sequence']
        out_path = os.path.join(phage_cache, f"{pid}.npy")
        phage_id_to_path[pid] = out_path
        if not os.path.exists(out_path):
            onehot_array = encode_sequence_onehot(seq, PHAGE_THRESHOLD)
            np.save(out_path, onehot_array)
    # Precompute bacteria sequences
    for _, row in bacteria_df.iterrows():
        bid = row['bacterium_id']
        seq = row['bacterium_sequence']
        out_path = os.path.join(bacteria_cache, f"{bid}.npy")
        bacterium_id_to_path[bid] = out_path
        if not os.path.exists(out_path):
            onehot_array = encode_sequence_onehot(seq, BACTERIUM_THRESHOLD)
            np.save(out_path, onehot_array)
    return phage_id_to_path, bacterium_id_to_path

class BacteriaBranch(nn.Module):
    """CNN branch for bacterial DNA sequence (channels-first input)."""
    def __init__(self):
        super(BacteriaBranch, self).__init__()
        self.conv1 = nn.Conv1d(4, 64, kernel_size=30, stride=10, bias=True)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=15, stride=5)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=25, stride=10, bias=True)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=10, stride=5)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=10, stride=5, bias=True)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
    def forward(self, x):
        # x shape: (batch, 4, BACTERIUM_THRESHOLD)
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.pool2(x)
        x = self.relu3(self.conv3(x))
        x = self.pool3(x)
        print(f"BacterialBranch shape after conv3 {x.shape}")
        # Permute to (batch, length, channels) to mimic Keras channels-last flatten, then flatten
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(x.size(0), -1)
        print(f"BacterialBranch shape after reshaping {x.shape}")        
        return x

class PhageBranch(nn.Module):
    """CNN branch for phage DNA sequence (channels-first input)."""
    def __init__(self):
        super(PhageBranch, self).__init__()
        self.conv1 = nn.Conv1d(4, 64, kernel_size=30, stride=10, bias=True)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=15, stride=5)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=25, stride=10, bias=True)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
    def forward(self, x):
        # x shape: (batch, 4, PHAGE_THRESHOLD)
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.pool2(x)
        print(f"PhageBranch shape after conv2 {x.shape}")
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(x.size(0), -1)
        print(f"PhageBranch shape after reshaping {x.shape}")        
        return x

class PerphectInteractionModel(nn.Module):
    """Dual-input CNN for phage-bacterium interaction prediction."""
    def __init__(self):
        super(PerphectInteractionModel, self).__init__()
        self.bacteria_branch = BacteriaBranch()
        self.phage_branch = PhageBranch()
        # Flattened feature lengths: 8928 (bacteria) + 6368 (phage) = 15296
        self.fc1 = nn.Linear(15296, 100, bias=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(100, 1, bias=True)
    def forward(self, bacteria_input, phage_input):
        # Pass each input through its branch
        bact_features = self.bacteria_branch(bacteria_input)
        phage_features = self.phage_branch(phage_input)
        # Concatenate features (batch, 15296)
        combined = torch.cat([bact_features, phage_features], dim=1)
        # Fully-connected layers and sigmoid output
        x = self.relu(self.fc1(combined))
        x = self.dropout(x)
        out = torch.sigmoid(self.fc2(x))
        return out

def load_keras_weights(pytorch_model: nn.Module, keras_h5_path: str):
    """
    Load weights from a Keras .h5 model file into the PyTorch model.
    Handles necessary transpositions for Conv1D and Dense layers.
    """
    with h5py.File(keras_h5_path, 'r') as f:
        model_weights = f['model_weights']
        state_dict = {}
        # Convolutional layers (transpose kernels from (kernel, in, out) to (out, in, kernel))
        def copy_conv(layer_name, pytorch_w, pytorch_b):
            kernel = model_weights[layer_name][layer_name]['kernel:0'][()]
            bias = model_weights[layer_name][layer_name]['bias:0'][()]
            state_dict[pytorch_w] = torch.tensor(kernel).permute(2, 1, 0)
            state_dict[pytorch_b] = torch.tensor(bias)
        copy_conv('bacterial_conv_1', 'bacteria_branch.conv1.weight', 'bacteria_branch.conv1.bias')
        copy_conv('bacterial_conv_2', 'bacteria_branch.conv2.weight', 'bacteria_branch.conv2.bias')
        copy_conv('bacterial_conv_3', 'bacteria_branch.conv3.weight', 'bacteria_branch.conv3.bias')
        copy_conv('phage_conv_1', 'phage_branch.conv1.weight', 'phage_branch.conv1.bias')
        copy_conv('phage_conv_2', 'phage_branch.conv2.weight', 'phage_branch.conv2.bias')
        # Dense layers (transpose weight matrices from (in, out) to (out, in))
        dense_kernel = model_weights['dense']['dense']['kernel:0'][()]
        dense_bias   = model_weights['dense']['dense']['bias:0'][()]
        dense1_kernel = model_weights['dense_1']['dense_1']['kernel:0'][()]
        dense1_bias   = model_weights['dense_1']['dense_1']['bias:0'][()]
        state_dict['fc1.weight'] = torch.tensor(dense_kernel).T
        state_dict['fc1.bias']   = torch.tensor(dense_bias)
        state_dict['fc2.weight'] = torch.tensor(dense1_kernel).T
        state_dict['fc2.bias']   = torch.tensor(dense1_bias)
    pytorch_model.load_state_dict(state_dict)
    return pytorch_model


# Dataset that loads precomputed tensors
class InteractionDataset(torch.utils.data.Dataset):
    def __init__(self, couples_df, phage_id_to_path, bacterium_id_to_path):
        self.couples = couples_df.reset_index(drop=True)
        self.phage_path = phage_id_to_path
        self.bact_path = bacterium_id_to_path
    def __len__(self):
        return len(self.couples)
    def __getitem__(self, idx):
        row = self.couples.iloc[idx]
        phage_id = row['phage_id']
        bacterium_id = row['bacterium_id']
        label = torch.tensor([row['interaction_type']], dtype=torch.float32)
        # Load precomputed one-hot arrays from disk and convert to float32 tensor
        phage_arr = np.load(self.phage_path[phage_id])
        bact_arr = np.load(self.bact_path[bacterium_id])
        phage_tensor = torch.from_numpy(phage_arr.astype(np.float32))
        bacterium_tensor = torch.from_numpy(bact_arr.astype(np.float32))
        return bacterium_tensor, phage_tensor, label
