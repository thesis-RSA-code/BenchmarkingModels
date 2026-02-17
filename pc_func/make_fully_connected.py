import torch

def create_fully_connected_edges(batch_size, num_active_per_batch, device):
    """
    Create fully connected edge_index within each event (only active partitions).
    
    Args:
        batch_size: number of events
        num_partitions: total partitions per event (including non-active)
        num_active_per_batch: [batch_size] - number of active partitions per event
        device: torch device
        
    Returns:
        edge_index: [2, total_edges] - fully connected edges within active partitions per event
    """
    edges = []
    offset = 0
    for b in range(batch_size):
        num_active = num_active_per_batch[b].item()
        # Fully connected within active partitions of this event
        node_indices = torch.arange(offset, offset + num_active, device=device)
        r = node_indices.repeat_interleave(num_active)
        c = node_indices.repeat(num_active)
        edges.append(torch.stack([r, c]))
        offset += num_active
    
    return torch.cat(edges, dim=1)


def create_fully_connected_mask(batch_size, num_partitions, num_active_per_batch, device):
    """
    Create attention mask for dense attention with fully connected graph within active partitions.
    
    For each event, only the active partitions can attend to each other (fully connected).
    Non-active partitions are masked out.
    
    Args:
        batch_size: number of events
        num_partitions: total partitions per event (including non-active)
        num_active_per_batch: [batch_size] - number of active partitions per event
        device: torch device
        
    Returns:
        mask: [batch_size, num_partitions, num_partitions] - attention mask
              True = attend, False = mask out
    """
    mask = torch.zeros(batch_size, num_partitions, num_partitions, dtype=torch.bool, device=device)
    
    for b in range(batch_size):
        num_active = num_active_per_batch[b].item()
        mask[b, :num_active, :num_active] = True
    
    return mask