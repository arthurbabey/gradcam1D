import torch
import torch.nn as nn
import h5py
import numpy as np
import pandas as pd
import os 

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


def combine_attributions(attr_tensor, method='sum', input_length=None):

    if attr_tensor.dim() != 3:
        raise ValueError("Attribution tensor must have 3 dimensions (batch, channels, length)")
    
    batch, channels, length = attr_tensor.shape
    
    # Case 1: GuidedGradCam attribution with 4 channels (one per nucleotide)
    if channels == 4:
        attr = attr_tensor.squeeze(0)  # now shape is [4, L]
        if method == 'sum':
            combined = attr.sum(dim=0)
        elif method == 'max':
            combined, _ = attr.max(dim=0)
        else:
            raise ValueError("Unsupported method. Choose 'sum' or 'max'.")
        # Normalize to range [0, 1]
        combined = (combined - combined.min()) / (combined.max() - combined.min())
        return combined.cpu().numpy()
    
    # Case 2: LayerGradCam attribution with 1 channel (coarse feature map)
    elif channels == 1:
        if input_length is None:
            raise ValueError("For a single-channel attribution (LayerGradCam), you must provide input_length.")
        # Convert input_length to int and wrap it in a tuple
        target_size = (int(input_length),)
        # Upsample from current length (N) to input_length using nearest-neighbor interpolation
        upsampled = F.interpolate(attr_tensor, size=target_size, mode="nearest")
        combined = upsampled.squeeze(0).squeeze(0)
        # Normalize to range [0, 1]
        combined = (combined - combined.min()) / (combined.max() - combined.min())
        return combined.cpu().numpy()
    
    else:
        raise ValueError("Unexpected channel dimension. Expected 1 or 4 channels.")
    
    
# Define a function to write this to a FASTA file
def write_fasta(seq, filename, seq_id="my_sequence"):
    with open(filename, "w") as f:
        f.write(f">{seq_id}\n")  # fasta header
        f.write(seq + "\n")       # actual sequence