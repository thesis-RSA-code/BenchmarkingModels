import torch


def create_star_edges_loop(partition_counts, num_cls_tokens, device):
    """
    Loop-based edge creation (baseline).
    
    Creates STAR graph edges within each partition by iterating over partitions.
    CLS tokens (first num_cls_tokens nodes) attend to all nodes in the partition.
    PMT nodes do not attend to each other.
    
    Pattern: star_node -> all_nodes (unidirectional)
    
    Args:
        partition_counts: [num_active_partitions] - number of PMTs in each partition
        num_cls_tokens: int - number of CLS tokens per partition
        device: torch device
        
    Returns:
        edge_index: [2, total_edges] - edge indices (src=CLS only, dst=all nodes)
    """
    edge_list = []
    node_offset = 0
    
    num_active_partitions = partition_counts.size(0)
    
    for i in range(num_active_partitions):
        num_pmts_in_partition = partition_counts[i].item()
        total_nodes_in_partition = num_cls_tokens + num_pmts_in_partition
        
        # Create star graph edges for this partition
        # CLS tokens (indices 0 to num_cls_tokens-1) attend to all nodes
        star_idxs = torch.arange(num_cls_tokens, device=device)
        all_idxs = torch.arange(total_nodes_in_partition, device=device)
        
        src = star_idxs.repeat_interleave(total_nodes_in_partition)  # CLS tokens as sources
        dst = all_idxs.repeat(num_cls_tokens)                        # All nodes as targets
        
        # Offset by global node index
        src = src + node_offset
        dst = dst + node_offset
        
        edge_list.append(torch.stack([src, dst]))
        node_offset += total_nodes_in_partition
    
    edge_index = torch.cat(edge_list, dim=1)
    return edge_index

def create_star_edges_vectorized(partition_counts, num_cls_tokens, device):
    """
    Improved vectorized star graph edge creation using repeat_interleave for batch operations.
    
    This version minimizes Python-level loops by using more tensor operations.
    Star pattern: CLS tokens attend to all nodes, PMTs don't attend to each other.
    
    Args:
        partition_counts: [num_active_partitions] - number of PMTs in each partition
        num_cls_tokens: int - number of CLS tokens per partition
        device: torch device
        
    Returns:
        edge_index: [2, total_edges] - edge indices (src=CLS only, dst=all nodes)
    """
    num_active_partitions = partition_counts.size(0)
    
    # Total nodes per partition
    nodes_per_partition = partition_counts + num_cls_tokens
    
    # Cumulative node offsets
    node_offsets = torch.cat([
        torch.zeros(1, dtype=torch.long, device=device),
        torch.cumsum(nodes_per_partition[:-1], dim=0)
    ])
    
    # Generate edges by repeating partition indices
    # For each partition p with n nodes, we need num_cls × n edges (star pattern)
    edges_per_partition = num_cls_tokens * nodes_per_partition
    
    # Create partition ID for each edge
    partition_ids = torch.arange(num_active_partitions, device=device).repeat_interleave(edges_per_partition)
    
    # For each edge, compute its local position within the partition's edge list
    edge_positions = torch.cat([
        torch.arange(edges_per_partition[i].item(), device=device)
        for i in range(num_active_partitions)
    ])
    
    # Convert edge position to (src, dst) within partition (star pattern)
    # src cycles through CLS tokens (0 to num_cls-1)
    # dst cycles through all nodes (0 to n_nodes-1)
    n_nodes_expanded = nodes_per_partition[partition_ids]
    local_src = edge_positions // n_nodes_expanded  # Which CLS token (0 to num_cls-1)
    local_dst = edge_positions % n_nodes_expanded    # Which target node (0 to n_nodes-1)
    
    # Add offsets to get global indices
    offsets_expanded = node_offsets[partition_ids]
    global_src = local_src + offsets_expanded
    global_dst = local_dst + offsets_expanded
    
    edge_index = torch.stack([global_src, global_dst], dim=0)
    return edge_index

