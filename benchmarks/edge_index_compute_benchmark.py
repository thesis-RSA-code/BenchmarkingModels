"""
Benchmark: Edge Index Creation for Star Graphs within Partitions

Compares different methods for creating edge indices and masks:
- Loop-based edge creation (original)
- Vectorized edge creation (optimized)
- Dense mask creation (for padded approach)

Goal: Determine if sparse (edge_index) or dense (padding + mask) is faster
for creating the attention graph structure in PatchGNN's local transformer.
"""

import os
import time
import numpy as np
import torch
from tqdm import tqdm

from pc_func.sparse_stars import (
    create_star_edges_loop,
    create_star_edges_vectorized,
)
from pc_func.masked_stars import (
    create_partition_mask_padded_vectorized,
)
from pc_func.make_partitions import generate_batched_partitions
from utils.init_main import is_running_in_container


# ============================================================================
# Benchmarking
# ============================================================================

def benchmark_edge_creation(edge_fn, *args, num_iters=100, warmup=10):
    """
    Benchmark edge/mask creation time.
    
    Args:
        edge_fn: Function that creates edges or masks
        *args: Arguments to pass to edge_fn
        num_iters: Number of benchmark iterations
        warmup: Number of warmup iterations
    
    Returns:
        avg_time, peak_memory
    """
    device = args[1] if len(args) > 1 else torch.device('cuda')
    
    # Warmup
    for _ in tqdm(range(warmup), desc="Warmup", leave=False):
        _ = edge_fn(*args)
        torch.cuda.synchronize()
    
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    times = []
    
    for _ in tqdm(range(num_iters), desc="Benchmark", leave=False):
        torch.cuda.synchronize()
        start = time.time()
        _ = edge_fn(*args)
        torch.cuda.synchronize()
        times.append(time.time() - start)
    
    avg_time = np.mean(times)
    peak_mem = torch.cuda.max_memory_allocated()
    
    return avg_time, peak_mem


def save_benchmark_results(filepath, config, results):
    """Save benchmark results to NPZ file."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    save_data = {
        # Configuration
        'config/batch_size': config['batch_size'],
        'config/num_partitions': config['num_partitions'],
        'config/avg_partition_size': config['avg_partition_size'],
        'config/sparsity': config['sparsity'],
        'config/max_partition_size': config['max_partition_size'],
        'config/num_cls_tokens': config['num_cls_tokens'],
        'config/num_iters': config['num_iters'],
        'config/warmup': config['warmup'],
        'config/use_compile': config['use_compile'],
        'config/timestamp': timestamp,
        
        # Batch statistics
        'batch_stats/total_nodes': config['total_nodes'],
        'batch_stats/total_edges': config['total_edges'],
        'batch_stats/avg_nodes_per_partition': config['avg_nodes_per_partition'],
    }
    
    # Add results for each method
    for method_name, method_results in results.items():
        save_data[f'{method_name}/time'] = method_results['time']
        save_data[f'{method_name}/memory'] = method_results['memory']
    
    np.savez(filepath, **save_data)
    print(f"\nResults saved to: {filepath}")


# ============================================================================
# Main Benchmark
# ============================================================================

def main(
    batch_size=16,
    num_partitions=10,
    avg_partition_size=50,
    sparsity=0.7,  # 70% of nodes are active
    max_partition_size=150,
    num_cls_tokens=4,
    num_iters=100,
    warmup=10,
    use_compile=True,
    compiling_mode='max-autotune',
    save_folder='results/edge_creation',
    device='cuda',
):
    """
    Benchmark edge index creation methods.
    
    Args:
        batch_size: Number of events in batch
        num_partitions: Number of partitions per event
        avg_partition_size: Average number of PMTs per partition
        sparsity: Fraction of active nodes (0.1 = 10%, 0.7 = 70%)
        max_partition_size: Maximum partition size for padding
        num_cls_tokens: Number of CLS tokens per partition
        num_iters: Number of benchmark iterations
        warmup: Number of warmup iterations
        use_compile: Whether to use torch.compile
        compiling_mode: Compilation mode ('default', 'reduce-overhead', 'max-autotune')
        save_folder: Folder to save results
        device: Device to run on
    """
    device = torch.device(device)
    os.makedirs(save_folder, exist_ok=True)
    
    print("="*80)
    print("EDGE INDEX CREATION BENCHMARK")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Partitions per event: {num_partitions}")
    print(f"  Avg partition size: {avg_partition_size}")
    print(f"  Sparsity: {sparsity*100:.0f}%")
    print(f"  Max partition size (padding): {max_partition_size}")
    print(f"  CLS tokens: {num_cls_tokens}")
    print(f"  Iterations: {num_iters} (warmup: {warmup})")
    print(f"  Use compile: {use_compile} ({compiling_mode if use_compile else 'N/A'})")
    print(f"  Device: {device}")
    
    # ========================================================================
    # Generate Data
    # ========================================================================
    
    print("\n" + "="*80)
    print("Generating Data")
    print("="*80)
    
    # Calculate partition sizes with sparsity
    # Base size follows distribution, then multiply by sparsity
    min_size = max(1, int(avg_partition_size * 0.5))
    max_size = min(max_partition_size, int(avg_partition_size * 1.5))
    
    batched_partitions = generate_batched_partitions(
        batch_size=batch_size,
        num_partitions=num_partitions,
        min_size=min_size,
        max_size=max_size,
        device=device
    )
    
    # Apply sparsity: reduce each partition count
    partition_counts = (batched_partitions['partition_counts'] * sparsity).long()
    partition_counts = torch.clamp(partition_counts, min=1)  # At least 1 node
    
    total_nodes = partition_counts.sum().item()
    avg_nodes = partition_counts.float().mean().item()
    
    print(f"\nGenerated partition counts:")
    print(f"  Total partitions: {len(partition_counts)}")
    print(f"  Total nodes: {total_nodes}")
    print(f"  Avg nodes per partition: {avg_nodes:.1f}")
    print(f"  Min/Max: {partition_counts.min().item()}/{partition_counts.max().item()}")
    
    # ========================================================================
    # Prepare Functions
    # ========================================================================
    
    print("\n" + "="*80)
    print("Preparing Methods")
    print("="*80)
    
    # Create wrapper functions with consistent interface
    def create_edges_loop(partition_counts, device):
        return create_star_edges_loop(
            partition_counts, num_cls_tokens, device
        )
    
    def create_edges_vec(partition_counts, device):
        return create_star_edges_vectorized(
            partition_counts, num_cls_tokens, device
        )
    
    def create_mask_dense(partition_counts, device):
        return create_partition_mask_padded_vectorized(
            partition_counts, num_cls_tokens, max_partition_size, device
        )
    
    # Compile if requested
    if use_compile:
        print(f"Compiling functions with mode='{compiling_mode}'...")
        create_edges_loop_compiled = torch.compile(create_edges_loop, mode=compiling_mode)
        create_edges_vec_compiled = torch.compile(create_edges_vec, mode=compiling_mode)
        create_mask_dense_compiled = torch.compile(create_mask_dense, mode=compiling_mode)
        print("Compilation complete!")
        
        methods = {
            'loop_compiled': create_edges_loop_compiled,
            'vectorized_compiled': create_edges_vec_compiled,
            'dense_mask_compiled': create_mask_dense_compiled,
        }
    else:
        methods = {
            'loop': create_edges_loop,
            'vectorized': create_edges_vec,
            'dense_mask': create_mask_dense,
        }
    
    # Verify correctness (create one sample for stats)
    edge_index_sample = create_edges_loop(partition_counts, device)
    total_edges = edge_index_sample.shape[1]
    
    print(f"\nMethod comparison:")
    for name in methods.keys():
        print(f"  - {name}")
    
    # ========================================================================
    # Run Benchmarks
    # ========================================================================
    
    print("\n" + "="*80)
    print("Running Benchmarks")
    print("="*80)
    
    results = {}
    
    for i, (method_name, method_fn) in enumerate(methods.items(), 1):
        print(f"\n[{i}/{len(methods)}] {method_name}")
        avg_time, peak_mem = benchmark_edge_creation(
            method_fn, partition_counts, device,
            num_iters=num_iters, warmup=warmup
        )
        print(f"  Time:   {avg_time*1000:.3f} ms")
        print(f"  Memory: {peak_mem/1e6:.2f} MB")
        
        results[method_name] = {
            'time': avg_time,
            'memory': peak_mem,
        }
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Method':<30} {'Time (ms)':<15} {'Memory (MB)':<15}")
    print("-"*80)
    
    for method_name, method_results in results.items():
        print(f"{method_name:<30} {method_results['time']*1000:<15.3f} {method_results['memory']/1e6:<15.2f}")
    
    # Find baseline (loop or loop_compiled)
    baseline_name = 'loop_compiled' if use_compile else 'loop'
    baseline_time = results[baseline_name]['time']
    
    print("-"*80)
    print(f"\nSpeedup vs {baseline_name}:")
    for method_name, method_results in results.items():
        if method_name != baseline_name:
            speedup = baseline_time / method_results['time']
            print(f"  {method_name}: {speedup:.2f}x")
    
    # ========================================================================
    # Save Results
    # ========================================================================
    
    config = {
        'batch_size': batch_size,
        'num_partitions': num_partitions,
        'avg_partition_size': avg_partition_size,
        'sparsity': sparsity,
        'max_partition_size': max_partition_size,
        'num_cls_tokens': num_cls_tokens,
        'num_iters': num_iters,
        'warmup': warmup,
        'use_compile': use_compile,
        'total_nodes': total_nodes,
        'total_edges': total_edges,
        'avg_nodes_per_partition': avg_nodes,
    }
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"edge_creation_{timestamp}_bs{batch_size}_np{num_partitions}_avgsize{int(avg_partition_size)}_sp{int(sparsity*100)}.npz"
    save_benchmark_results(os.path.join(save_folder, filename), config, results)


if __name__ == '__main__':
    if is_running_in_container():
        print("Running inside container. Setting up cache directories...")
        user_name = os.getenv('USER', 'elebleve')
        cache_root = f"/tmp/{user_name}/torch_cache"
        os.makedirs(cache_root, exist_ok=True)
        os.environ["TRITON_CACHE_DIR"] = os.path.join(cache_root, "triton")
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(cache_root, "inductor")
        os.environ["XDG_CACHE_HOME"] = os.path.join(cache_root, "xdg")
        os.environ["TORCH_HOME"] = cache_root
        os.environ["HOME"] = cache_root
        os.environ["TMPDIR"] = cache_root
        print(f"Cache directories set to: {cache_root}")
    
    # Run benchmark with default parameters (you can adjust these)
    main(
        batch_size=50,
        num_partitions=500,
        avg_partition_size=70,
        sparsity=0.5,  # 70% active nodes
        max_partition_size=150,
        num_cls_tokens=3,
        num_iters=100,
        warmup=10,
        use_compile=False,
        compiling_mode='default',
        save_folder='outputs/edge_creation',
        device='cuda',
    )
