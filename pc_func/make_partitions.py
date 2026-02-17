"""
Dummy Partition Generator for Testing Star Graph Edge Creation

Generates synthetic partition data with variable sizes to simulate
real detector partitioning scenarios.
"""

import torch
import numpy as np


def generate_dummy_partitions(num_partitions=100, 
                              min_size=10, 
                              max_size=100,
                              distribution='uniform',
                              device='cpu',
                              seed=None):
    """
    Generate dummy partition sizes for testing.
    
    Args:
        num_partitions: number of partitions to generate
        min_size: minimum PMTs per partition
        max_size: maximum PMTs per partition
        distribution: 'uniform', 'normal', 'clustered', or 'realistic'
        device: torch device
        seed: random seed for reproducibility
        
    Returns:
        partition_counts: [num_partitions] tensor of PMT counts per partition
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    if distribution == 'uniform':
        # Uniform random distribution
        partition_counts = torch.randint(min_size, max_size + 1, 
                                        (num_partitions,), 
                                        device=device)
    
    elif distribution == 'normal':
        # Normal distribution centered at midpoint
        mean = (min_size + max_size) / 2
        std = (max_size - min_size) / 6  # ~99.7% within range
        
        counts = torch.randn(num_partitions, device=device) * std + mean
        partition_counts = counts.clamp(min_size, max_size).long()
    
    elif distribution == 'clustered':
        # Two clusters: small partitions and large partitions
        half = num_partitions // 2
        
        small_cluster = torch.randint(min_size, (min_size + max_size) // 2, 
                                     (half,), device=device)
        large_cluster = torch.randint((min_size + max_size) // 2, max_size + 1,
                                     (num_partitions - half,), device=device)
        
        partition_counts = torch.cat([small_cluster, large_cluster])
        partition_counts = partition_counts[torch.randperm(num_partitions)]
    
    elif distribution == 'realistic':
        # Simulates detector partitioning:
        # - Most partitions around 60-80 PMTs (well-balanced)
        # - Some edge cases (small: 10-30, large: 80-100)
        
        # 70% in main cluster (60-80)
        main_count = int(num_partitions * 0.7)
        main_cluster = torch.randint(60, 81, (main_count,), device=device)
        
        # 20% medium-small (30-60)
        small_count = int(num_partitions * 0.2)
        small_cluster = torch.randint(30, 61, (small_count,), device=device)
        
        # 10% very small or very large (10-30 or 80-100)
        edge_count = num_partitions - main_count - small_count
        edge_cluster = torch.cat([
            torch.randint(10, 31, (edge_count // 2,), device=device),
            torch.randint(80, 101, (edge_count - edge_count // 2,), device=device)
        ])
        
        partition_counts = torch.cat([main_cluster, small_cluster, edge_cluster])
        partition_counts = partition_counts[torch.randperm(num_partitions)]
    
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    
    return partition_counts


def generate_batched_partitions(batch_size=16,
                                num_partitions=100,
                                min_size=10,
                                max_size=100,
                                distribution='realistic',
                                device='cpu',
                                seed=None):
    """
    Generate dummy partitions for multiple events (batched).
    
    Simulates real training scenario where you have multiple events,
    each with their own set of partitions with hits.
    
    Args:
        batch_size: number of events
        num_partitions: total partitions per event
        min_size: minimum PMTs per partition
        max_size: maximum PMTs per partition
        distribution: partition size distribution
        device: torch device
        seed: random seed
        
    Returns:
        dict with:
            - partition_counts: [total_active_partitions] - PMTs per partition
            - batch_ids: [total_active_partitions] - which event each partition belongs to
            - num_active_per_batch: [batch_size] - number of active partitions per event
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    all_partition_counts = []
    all_batch_ids = []
    num_active_per_batch = []
    
    for b in range(batch_size):
        # Each event has a random subset of partitions with hits
        # (Not all partitions have hits in every event)
        num_active = torch.randint(int(num_partitions * 0.6), 
                                   int(num_partitions * 0.95), 
                                   (1,)).item()
        
        # Generate partition sizes for this event
        partitions = generate_dummy_partitions(
            num_partitions=num_active,
            min_size=min_size,
            max_size=max_size,
            distribution=distribution,
            device=device,
            seed=None  # Don't fix seed per batch
        )
        
        batch_ids = torch.full((num_active,), b, dtype=torch.long, device=device)
        
        all_partition_counts.append(partitions)
        all_batch_ids.append(batch_ids)
        num_active_per_batch.append(num_active)
    
    return {
        'partition_counts': torch.cat(all_partition_counts),
        'batch_ids': torch.cat(all_batch_ids),
        'num_active_per_batch': torch.tensor(num_active_per_batch, device=device),
    }


def print_partition_statistics(partition_counts):
    """
    Print statistics about partition size distribution.
    
    Args:
        partition_counts: [num_partitions] tensor
    """
    counts_np = partition_counts.cpu().numpy()
    
    print(f"\n{'='*60}")
    print(f"  Partition Statistics")
    print(f"{'='*60}")
    print(f"  Number of partitions: {len(counts_np)}")
    print(f"  Total PMTs: {counts_np.sum()}")
    print(f"  Min size: {counts_np.min()}")
    print(f"  Max size: {counts_np.max()}")
    print(f"  Mean size: {counts_np.mean():.1f}")
    print(f"  Median size: {np.median(counts_np.astype(float)):.1f}")
    print(f"  Std dev: {counts_np.std():.1f}")
    print(f"\n  Distribution:")
    print(f"    <30 PMTs: {(counts_np < 30).sum()} ({(counts_np < 30).sum()/len(counts_np)*100:.1f}%)")
    print(f"    30-60 PMTs: {((counts_np >= 30) & (counts_np < 60)).sum()} ({((counts_np >= 30) & (counts_np < 60)).sum()/len(counts_np)*100:.1f}%)")
    print(f"    60-80 PMTs: {((counts_np >= 60) & (counts_np < 80)).sum()} ({((counts_np >= 60) & (counts_np < 80)).sum()/len(counts_np)*100:.1f}%)")
    print(f"    ≥80 PMTs: {(counts_np >= 80).sum()} ({(counts_np >= 80).sum()/len(counts_np)*100:.1f}%)")
    print(f"{'='*60}\n")


# Example usage and testing
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Test different distributions
    for dist in ['uniform', 'normal', 'clustered', 'realistic']:
        print(f"\n{'#'*60}")
        print(f"  Testing distribution: {dist.upper()}")
        print(f"{'#'*60}")
        
        partitions = generate_dummy_partitions(
            num_partitions=100,
            min_size=10,
            max_size=100,
            distribution=dist,
            device=device,
            seed=42
        )
        
        print_partition_statistics(partitions)
    
    # Test batched generation
    print(f"\n{'#'*60}")
    print(f"  Testing BATCHED generation (realistic)")
    print(f"{'#'*60}")
    
    batched_data = generate_batched_partitions(
        batch_size=16,
        num_partitions=100,
        min_size=10,
        max_size=100,
        distribution='realistic',
        device=device,
        seed=42
    )
    
    print(f"\nBatch information:")
    print(f"  Total active partitions: {len(batched_data['partition_counts'])}")
    print(f"  Batch size: {batched_data['num_active_per_batch'].size(0)}")
    print(f"  Active partitions per event: min={batched_data['num_active_per_batch'].min().item()}, "
          f"max={batched_data['num_active_per_batch'].max().item()}, "
          f"mean={batched_data['num_active_per_batch'].float().mean().item():.1f}")
    
    print_partition_statistics(batched_data['partition_counts'])