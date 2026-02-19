import torch

def create_partition_mask(partition_counts, num_cls_tokens, device):
    """
    Create attention mask for dense attention with STAR pattern (no explicit edges).
    
    Returns a boolean mask where True indicates valid attention positions.
    Star pattern: CLS tokens attend to all nodes in their partition, PMTs don't attend to others.
    
    Args:
        partition_counts: [num_active_partitions] - number of PMTs in each partition
        num_cls_tokens: int - number of CLS tokens per partition
        device: torch device
        
    Returns:
        mask: [total_nodes, total_nodes] - attention mask (True = attend, False = mask)
        total_nodes: int - total number of nodes
    """
    num_active_partitions = partition_counts.size(0)
    nodes_per_partition = partition_counts + num_cls_tokens
    total_nodes = nodes_per_partition.sum().item()
    
    # Initialize mask to False (no attention)
    mask = torch.zeros(total_nodes, total_nodes, dtype=torch.bool, device=device)
    
    # Set True for valid attention within each partition (star pattern)
    node_offset = 0
    for i in range(num_active_partitions):
        n_nodes = nodes_per_partition[i].item()
        # CLS tokens (first num_cls_tokens) attend to all nodes in partition
        mask[node_offset:node_offset+num_cls_tokens, node_offset:node_offset+n_nodes] = True
        node_offset += n_nodes
    
    return mask, total_nodes

def create_partition_mask_padded_vectorized(partition_counts, num_cls_tokens, max_partition_size, device):
    """
    Vectorized creation of per-partition masks with padding to max_partition_size.
    
    Creates separate masks for each partition (not one global mask).
    Star pattern: CLS tokens attend to all nodes in their partition.
    
    Args:
        partition_counts: [num_partitions] - number of PMTs in each partition
        num_cls_tokens: int - number of CLS tokens per partition
        max_partition_size: int - maximum partition size for padding (exluding CLS tokens)
        device: torch device
        
    Returns:
        masks: [num_partitions, max_partition_size, max_partition_size] - per-partition masks
    """
    num_partitions = partition_counts.size(0)
    
    # Total nodes per partition (CLS + PMTs)
    nodes_per_partition = partition_counts + num_cls_tokens  # [num_partitions]
    
    # Initialize all masks to False: [num_partitions, max_partition_size, max_partition_size]
    cls_included_max_partition_size = max_partition_size + num_cls_tokens
    masks = torch.zeros(num_partitions, cls_included_max_partition_size, cls_included_max_partition_size, dtype=torch.bool, device=device)
    
    # Create row and column indices for all positions
    # row_idx: [num_partitions, max_partition_size] - which row in each partition
    # col_idx: [num_partitions, max_partition_size] - which column in each partition
    row_idx = torch.arange(cls_included_max_partition_size, device=device).unsqueeze(0).expand(num_partitions, -1)
    col_idx = torch.arange(cls_included_max_partition_size, device=device).unsqueeze(0).expand(num_partitions, -1)
    
    # Conditions for valid attention (star pattern):
    # - Row must be a CLS token: row_idx < num_cls_tokens
    # - Column must be an active node: col_idx < nodes_per_partition
    is_cls_row = row_idx < num_cls_tokens  # [num_partitions, max_partition_size]
    is_active_col = col_idx < nodes_per_partition.unsqueeze(1)  # [num_partitions, max_partition_size]
    
    # Broadcast to [num_partitions, max_partition_size, max_partition_size]
    valid_attention = is_cls_row.unsqueeze(2) & is_active_col.unsqueeze(1)
    
    masks = valid_attention
    
    return masks

