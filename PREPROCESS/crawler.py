import os
import torch
import numpy as np
from tqdm import tqdm

# =============================
# Configuration Parameters
# =============================

CONFIG = {
    'hash_encoding': {
        'num_levels': 16,             # Number of levels in the hash encoding
        'level_dim': 2,               # Dimension of feature vectors at each level
        'input_dim': 3,               # Usually 3D input coordinates (x, y, z)
        'log2_hashmap_size': 19,      # Hash table size: 2^19 entries
        'base_resolution': 16         # Resolution of the first (lowest) level
    }
}

# =============================
# Utility Functions
# =============================

def load_torch_weights(file_path):
    """
    Load model weights from a checkpoint .pth file.

    Args:
        file_path (str): Path to the checkpoint file.

    Returns:
        dict or None: The model weight dictionary if successful, else None.
    """
    try:
        weights = torch.load(file_path, map_location='cpu')
        return weights['model']  # Assumes weights are under 'model' key
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def extract_hash_encoding_structure(model_weights, config):
    """
    Extract per-level embeddings from the hash encoder based on NeRF model weights.

    Args:
        model_weights (dict): State dict containing model parameters.
        config (dict): Configuration for the hash encoding.

    Returns:
        dict: A dictionary of level-wise embeddings (e.g., 'level_0', 'level_1', ...).
    """
    embeddings = model_weights['_orig_mod.grid_encoder.embeddings']  # Shape: [total_params, level_dim]

    # Unpack configuration values
    num_levels = config['num_levels']
    level_dim = config['level_dim']
    input_dim = config['input_dim']
    log2_hashmap_size = config['log2_hashmap_size']
    base_resolution = config['base_resolution']

    max_params = 2 ** log2_hashmap_size
    # Scaling factor for resolution per level
    per_level_scale = np.exp2(np.log2(2048 / base_resolution) / (num_levels - 1))

    hash_structure = {}
    offset = 0  # Index to slice embeddings for each level

    for level in range(num_levels):
        # Calculate the resolution and number of parameters for this level
        resolution = int(np.ceil(base_resolution * (per_level_scale ** level)))
        params_in_level = min(max_params, resolution ** input_dim)
        params_in_level = int(np.ceil(params_in_level / 8) * 8)  # Ensure divisibility by 8

        # Slice the embedding weights for this level
        level_weights = embeddings[offset:offset + params_in_level]

        # Save the level embedding
        hash_structure[f'level_{level}'] = level_weights

        # Update slicing offset for the next level
        offset += params_in_level

    return hash_structure

def preprocess_to_tokens_and_positions(base_dict):
    """
    Flatten the hash structure into token and position tensors.

    Args:
        base_dict (dict): Dictionary with level-wise weights (e.g., 'level_0': Tensor[N, 2]).

    Returns:
        (Tensor, Tensor): Tuple of:
            - tokens: Tensor of shape [Total_N, 2]
            - positions: Tensor of shape [Total_N, 3] where each row is [global_index, layer_index, index_in_layer]
    """
    tokens = []
    positions = []
    global_index = 0

    # Sort keys to ensure consistent layer ordering
    for key in sorted(base_dict.keys(), key=lambda x: int(x.split("_")[1])):
        layer_index = int(key.split("_")[1])
        layer_tensor = base_dict[key]
        num_tokens = layer_tensor.shape[0]

        for pos_in_layer in range(num_tokens):
            token = layer_tensor[pos_in_layer]  # Shape: [2]
            tokens.append(token)
            positions.append(torch.tensor([global_index, layer_index, pos_in_layer]))
            global_index += 1

    return torch.stack(tokens), torch.stack(positions)

def save_processed_model(output_dir, model_id, tokens, positions):
    """
    Save preprocessed token and position tensors to disk.

    Args:
        output_dir (str): Destination directory for saving files.
        model_id (str): Identifier used to name the output files.
        tokens (Tensor): Tensor of shape [N, 2]
        positions (Tensor): Tensor of shape [N, 3]
    """
    os.makedirs(output_dir, exist_ok=True)

    torch.save(tokens, os.path.join(output_dir, f"{model_id}_tokens.pt"))
    torch.save(positions, os.path.join(output_dir, f"{model_id}_positions.pt"))

# =============================
# Crawler: Batch Preprocessing
# =============================

def run_crawler(model_root_dir, output_dir):
    """
    Scan a directory tree for .pth model files, extract their hash encoding weights,
    and save them as token/position pairs for downstream SANE processing.

    Args:
        model_root_dir (str): Root directory containing model .pth files.
        output_dir (str): Directory to save the processed token and position tensors.
    """
    model_paths = []

    # Step 1: Discover all .pth files in the directory tree
    for root, dirs, files in os.walk(model_root_dir):
        for file in files:
            if file.endswith('.pth'):
                full_path = os.path.join(root, file)
                model_paths.append(full_path)

    print(f"🔍 Found {len(model_paths)} models. Starting preprocessing...")

    # Step 2: Process each model file
    for path in tqdm(model_paths):
        # Generate a model identifier (file-path-safe)
        model_id = os.path.relpath(path, model_root_dir).replace(os.sep, '__').replace('.pth', '')

        # Load and process weights
        weights = load_torch_weights(path)
        if weights is None:
            continue

        # Extract hierarchical hash structure (16 levels)
        base_dict = extract_hash_encoding_structure(weights, CONFIG['hash_encoding'])

        # Convert to flat token + position format
        tokens, positions = preprocess_to_tokens_and_positions(base_dict)

        # Save the result
        save_processed_model(output_dir, model_id, tokens, positions)

    print(f"✅ Processing complete. Results saved to: {output_dir}")

run_crawler(
    model_root_dir='shared_data/',
    output_dir='crawled_models/'
)
