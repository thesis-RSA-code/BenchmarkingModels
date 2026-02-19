"""
Batch preparation functions for sparse (PyG) and dense (batched) models.

These functions convert partition data into appropriate formats for:
- PatchGNNTransformer: PyG-style flat format
- PatchDenseTransformer: Batched format with proper dimensions
"""

import torch
from pc_func.masked_stars import create_partition_mask_padded_vectorized


def prepare_sparse_batch(batched_partitions, num_cls_tokens, hidden_channels, device):
    """ 
    Prepare batch in PyG sparse format for PatchGNNTransformer.
    
    Args:
        batched_partitions: dict from generate_batched_partitions containing:
            - partition_counts: [total_active_partitions] - PMTs per partition
            - batch_ids: [total_active_partitions] - event index for each partition
            - num_active_per_batch: [B] - active partitions per event
        num_cls_tokens: int - number of CLS tokens per partition
        hidden_channels: int - feature dimension
        device: torch device
        
    Returns:
        dict with:
            - x: [total_nodes, hidden_channels] - flat node features (random)
            - partition_counts: [total_active_partitions] - PMTs per partition
    """
    partition_counts = batched_partitions['partition_counts']
    
    # Calculate total nodes (PMTs + CLS tokens across all partitions)
    nodes_per_partition = partition_counts + num_cls_tokens
    total_nodes = nodes_per_partition.sum().item()
    
    # Create flat node features (random for benchmarking)
    x = torch.randn(total_nodes, hidden_channels, device=device, requires_grad=False)
    
    return {
        'x': x,
        'partition_counts': partition_counts,
        'batch_ids': batched_partitions['batch_ids'],
        'num_active_per_batch': batched_partitions['num_active_per_batch'],
    }


def prepare_dense_batch(batched_partitions, num_partitions, max_partition_size, 
                       num_cls_tokens, hidden_channels, device):
    """
    Prepare batch in dense batched format for PatchDenseTransformer.
    
    Args:
        batched_partitions: dict from generate_batched_partitions containing:
            - partition_counts: [total_active_partitions] - PMTs per partition
            - batch_ids: [total_active_partitions] - event index for each partition
            - num_active_per_batch: [B] - active partitions per event
        num_partitions: int - maximum number of partitions per event (for padding)
        max_partition_size: int - maximum PMTs per partition (excluding CLS tokens)
        num_cls_tokens: int - number of CLS tokens per partition
        hidden_channels: int - feature dimension
        device: torch device
        
    Returns:
        dict with:
            - x_batched: [B, num_partitions, max_partition_size + num_cls_tokens, hidden_channels]
            - num_active_per_batch: [B] - active partitions per event
    """
    partition_counts = batched_partitions['partition_counts']
    batch_ids = batched_partitions['batch_ids']
    num_active_per_batch = batched_partitions['num_active_per_batch']
    
    batch_size = len(num_active_per_batch)
    total_active_partitions = len(partition_counts)
    cls_included_max_size = max_partition_size + num_cls_tokens
    
    # Initialize batched tensor with zeros (inactive partitions remain zero)
    x_batched = torch.zeros(
        batch_size, num_partitions, cls_included_max_size, hidden_channels,
        device=device, requires_grad=False
    )
    
    # Fill in active partitions with random data
    for idx in range(total_active_partitions):
        batch_id = batch_ids[idx].item()
        n_pmts = partition_counts[idx].item()
        n_nodes = n_pmts + num_cls_tokens
        
        # Find which partition slot this is within its batch
        # Count how many partitions from the same batch came before this one
        within_batch_idx = (batch_ids[:idx+1] == batch_id).sum().item() - 1
        
        # Fill with random features for active nodes
        x_batched[batch_id, within_batch_idx, :n_nodes, :] = torch.randn(
            n_nodes, hidden_channels, device=device
        )
    
    return {
        'x_batched': x_batched,
        'num_active_per_batch': num_active_per_batch,
    }


def create_partition_mask_batched(batched_partitions, num_partitions, max_partition_size, 
                                   num_cls_tokens, device):
    """
    Create batched attention masks for dense transformer.
    
    Converts flat partition masks to properly batched format with batch dimension.
    This function creates masks for an entire batch of events where each event
    can have a different number of active partitions.
    
    Star pattern: CLS tokens attend to all nodes in their partition.
    Inactive partitions (padding) have all-False masks.
    
    Args:
        batched_partitions: dict from generate_batched_partitions containing:
            - partition_counts: [total_active_partitions] - PMTs per partition (flat)
            - batch_ids: [total_active_partitions] - which event each partition belongs to
            - num_active_per_batch: [B] - number of active partitions per event
        num_partitions: int - maximum number of partitions per event (for padding)
        max_partition_size: int - maximum PMTs per partition (excluding CLS tokens)
        num_cls_tokens: int - number of CLS tokens per partition
        device: torch device
        
    Returns:
        masks_batched: [B, num_partitions, max_partition_size+num_cls_tokens, max_partition_size+num_cls_tokens]
            Batched attention masks. Inactive partitions have all-False masks.
            
    Example:
        If B=2, event 0 has 3 active partitions, event 1 has 2 active partitions,
        and num_partitions=5 (max), then:
        - masks_batched[0, 0:3] will contain real masks
        - masks_batched[0, 3:5] will be all-False (inactive)
        - masks_batched[1, 0:2] will contain real masks
        - masks_batched[1, 2:5] will be all-False (inactive)
    """
    partition_counts = batched_partitions['partition_counts']
    batch_ids = batched_partitions['batch_ids']
    num_active_per_batch = batched_partitions['num_active_per_batch']
    
    batch_size = len(num_active_per_batch)
    cls_included_max_size = max_partition_size + num_cls_tokens
    
    # First create flat masks for all active partitions
    masks_flat = create_partition_mask_padded_vectorized(
        partition_counts, num_cls_tokens, max_partition_size, device
    )  # [total_active_partitions, max_size, max_size]
    
    # Initialize batched masks (all False for inactive partitions)
    masks_batched = torch.zeros(
        batch_size, num_partitions, cls_included_max_size, cls_included_max_size,
        device=device, dtype=torch.bool
    )
    
    # Fill in masks for active partitions
    for idx in range(len(partition_counts)):
        batch_id = batch_ids[idx].item()
        # Find which partition slot this is within its event
        within_batch_idx = (batch_ids[:idx+1] == batch_id).sum().item() - 1
        masks_batched[batch_id, within_batch_idx] = masks_flat[idx]
    
    return masks_batched
