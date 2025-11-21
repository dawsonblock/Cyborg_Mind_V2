"""
Observation adapter for MineRL environments.

Converts MineRL observation dictionaries to the format expected by BrainCyborgMind.
"""

import numpy as np
import cv2
from typing import Dict, Any, Tuple


def obs_to_brain(
    obs: Dict[str, Any],
    image_size: int = 128
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert MineRL observation dictionary to brain inputs.
    
    Args:
        obs: MineRL observation dictionary with at least 'pov' key
        image_size: Target image size for pixels (default: 128)
        
    Returns:
        Tuple of (pixels, scalars, goal):
        - pixels: [3, H, W] float32 in [0, 1]
        - scalars: [20] float32 game state features
        - goal: [4] float32 goal embedding
    """
    # Extract and process POV (point-of-view) image
    pov = obs["pov"]  # Shape: [H, W, 3], dtype: uint8
    
    # Resize to target size
    pov_resized = cv2.resize(
        pov,
        (image_size, image_size),
        interpolation=cv2.INTER_AREA
    )
    
    # Normalize to [0, 1] and transpose to [C, H, W]
    pixels = pov_resized.astype(np.float32) / 255.0
    pixels = np.transpose(pixels, (2, 0, 1))  # [3, H, W]
    
    # Extract scalar features from observation
    # TODO: Populate with actual game state features:
    # - Health, hunger, oxygen
    # - Inventory counts (logs, planks, tools, etc.)
    # - Position info (y-level, biome)
    # - Time of day
    # - Distance to objectives
    # For now, return zeros as placeholder
    scalars = np.zeros(20, dtype=np.float32)
    
    # Example of how to populate scalars when available:
    # if 'inventory' in obs:
    #     scalars[0] = obs['inventory'].get('log', 0) / 64.0  # Normalized log count
    #     scalars[1] = obs['inventory'].get('planks', 0) / 64.0
    #     scalars[2] = obs['inventory'].get('stick', 0) / 64.0
    # if 'health' in obs:
    #     scalars[3] = obs['health'] / 20.0  # Normalized health
    # if 'food' in obs:
    #     scalars[4] = obs['food'] / 20.0  # Normalized food level
    
    # Goal embedding (task-specific)
    # TODO: Encode current task/objective
    # For TreeChop: [is_chopping_task, has_axe, logs_needed, time_remaining]
    # For now, return zeros
    goal = np.zeros(4, dtype=np.float32)
    
    return pixels, scalars, goal


def obs_to_teacher_inputs(
    obs: Dict[str, Any],
    image_size: int = 128
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert MineRL observation to RealTeacher inputs (simplified version).
    
    Args:
        obs: MineRL observation dictionary
        image_size: Target image size (default: 128)
        
    Returns:
        Tuple of (pixels, scalars):
        - pixels: [3, H, W] float32 in [0, 1]
        - scalars: [20] float32 state features
    """
    pixels, scalars, _ = obs_to_brain(obs, image_size)
    return pixels, scalars


def extract_inventory_features(obs: Dict[str, Any]) -> np.ndarray:
    """
    Extract inventory-related features from observation.
    
    Args:
        obs: MineRL observation dictionary
        
    Returns:
        Feature vector with inventory counts (normalized)
    """
    features = []
    
    if 'inventory' in obs:
        inv = obs['inventory']
        # Common items for TreeChop task
        items = ['log', 'planks', 'stick', 'crafting_table', 'wooden_axe',
                 'stone_axe', 'iron_axe', 'dirt', 'cobblestone', 'coal']
        for item in items:
            count = inv.get(item, 0)
            # Normalize by stack size (64 for most items)
            features.append(float(count) / 64.0)
    else:
        # If no inventory info, return zeros
        features = [0.0] * 10
    
    return np.array(features, dtype=np.float32)


def extract_status_features(obs: Dict[str, Any]) -> np.ndarray:
    """
    Extract player status features (health, hunger, etc.).
    
    Args:
        obs: MineRL observation dictionary
        
    Returns:
        Feature vector with status info (normalized)
    """
    features = []
    
    # Health (0-20)
    if 'life_stats' in obs and 'life' in obs['life_stats']:
        features.append(float(obs['life_stats']['life']) / 20.0)
    else:
        features.append(1.0)  # Assume full health if unknown
    
    # Food level (0-20)
    if 'life_stats' in obs and 'food' in obs['life_stats']:
        features.append(float(obs['life_stats']['food']) / 20.0)
    else:
        features.append(1.0)  # Assume full food if unknown
    
    # Oxygen (0-300)
    if 'life_stats' in obs and 'air' in obs['life_stats']:
        features.append(float(obs['life_stats']['air']) / 300.0)
    else:
        features.append(1.0)  # Assume full oxygen if unknown
    
    # XP level
    if 'life_stats' in obs and 'xp' in obs['life_stats']:
        features.append(float(obs['life_stats']['xp']) / 100.0)
    else:
        features.append(0.0)
    
    return np.array(features, dtype=np.float32)


def build_rich_scalars(obs: Dict[str, Any]) -> np.ndarray:
    """
    Build a comprehensive scalar feature vector from observation.
    
    This combines inventory, status, and other game state info into
    a single vector that can be used as input to the brain.
    
    Args:
        obs: MineRL observation dictionary
        
    Returns:
        20-dimensional feature vector (float32)
    """
    features = []
    
    # Inventory features (10 dims)
    inv_feats = extract_inventory_features(obs)
    features.extend(inv_feats)
    
    # Status features (4 dims)
    status_feats = extract_status_features(obs)
    features.extend(status_feats)
    
    # Position/environment (6 dims)
    # Y-level, is_on_ground, is_in_water, is_in_lava, time_of_day, etc.
    # Placeholder for now
    features.extend([0.0] * 6)
    
    # Ensure exactly 20 dimensions
    features = features[:20]
    while len(features) < 20:
        features.append(0.0)
    
    return np.array(features, dtype=np.float32)
