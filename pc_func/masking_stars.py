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


def create_partition_mask_vectorized(partition_counts, num_cls_tokens, device):
    """
    Vectorized version of partition mask creation with STAR pattern.
    
    Star pattern: CLS tokens attend to all nodes in their partition, PMTs don't attend to others.
    
    Args:
        partition_counts: [num_active_partitions] - number of PMTs in each partition
        num_cls_tokens: int - number of CLS tokens per partition
        device: torch device
        
    Returns:
        mask: [total_nodes, total_nodes] - attention mask (star pattern)
        total_nodes: int - total number of nodes
    """
    num_active_partitions = partition_counts.size(0)
    nodes_per_partition = partition_counts + num_cls_tokens
    total_nodes = nodes_per_partition.sum().item()
    
    # Initialize mask to False
    mask = torch.zeros(total_nodes, total_nodes, dtype=torch.bool, device=device)
    
    # Cumulative offsets
    node_offsets = torch.cat([
        torch.zeros(1, dtype=torch.long, device=device),
        torch.cumsum(nodes_per_partition[:-1], dim=0)
    ])
    
    # For each partition, create partition ID and position within partition
    partition_ids = torch.arange(num_active_partitions, device=device).repeat_interleave(nodes_per_partition)
    
    # Position within partition (0 to n_nodes-1)
    local_positions = torch.cat([
        torch.arange(nodes_per_partition[i].item(), device=device)
        for i in range(num_active_partitions)
    ])
    
    # CLS nodes are those with local_positions < num_cls_tokens
    is_cls = local_positions < num_cls_tokens
    
    # Mask is True when:
    # - Source node is CLS AND both nodes in same partition
    mask_same_partition = partition_ids.unsqueeze(0) == partition_ids.unsqueeze(1)
    mask_src_is_cls = is_cls.unsqueeze(1)  # [total_nodes, 1]
    
    mask = mask_same_partition & mask_src_is_cls
    
    return mask, total_nodes
