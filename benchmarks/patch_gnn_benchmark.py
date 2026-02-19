"""
Benchmark: Local PatchGNN Network (Step 2)

Compares two approaches for local attention within partitions:
- Model A: Sparse attention with vectorized star edge creation
- Model B: Dense attention with vectorized masking + PyTorch MHA

This isolates the local transformer step that creates super nodes from partitions.
"""

import os
import time
from tkinter import FALSE
import numpy as np
from tqdm import tqdm

from utils.init_main import is_running_in_container


# ============================================================================
# Benchmarking
# ============================================================================

def benchmark_model(model, input_data, edge_or_mask, num_iters=100, warmup=10):
    """
    Benchmark forward + backward pass (structure is pre-computed, not recreated each iteration).
    
    This benchmarks the model only:
    1. Forward pass with input data
    2. Backward pass
    
    Edge/mask creation is done ONCE before calling this function.
    
    Args:
        model: The model to benchmark
        input_data: Input features (x_flat for sparse, x_batched for dense)
        edge_or_mask: Pre-computed edges or masks (not recreated each iteration)
        num_iters: Number of benchmark iterations
        warmup: Number of warmup iterations
    
    Returns:
        avg_forward_time, avg_backward_time, total_time, peak_memory
    """
    
    # Warmup
    for _ in tqdm(range(warmup), desc="Warmup", leave=False):
        output = model(input_data, edge_or_mask)
        loss = output.sum()
        loss.backward()
        model.zero_grad()
    
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    forward_times = []
    backward_times = []
    
    for _ in tqdm(range(num_iters), desc="Benchmark", leave=False):

        # Forward
        torch.cuda.synchronize()
        fwd_start = time.time()
        output = model(input_data, edge_or_mask)
        loss = output.sum()
        torch.cuda.synchronize()
        forward_times.append(time.time() - fwd_start)
        
        # Backward
        torch.cuda.synchronize()
        bwd_start = time.time()
        loss.backward()
        torch.cuda.synchronize()
        backward_times.append(time.time() - bwd_start)
        
        model.zero_grad()
    
    avg_fwd = np.mean(forward_times)
    avg_bwd = np.mean(backward_times)
    total = avg_fwd + avg_bwd
    peak_mem = torch.cuda.max_memory_allocated()
    
    return avg_fwd, avg_bwd, total, peak_mem


def save_benchmark_results(filepath, config, results):
    """Save benchmark results to NPZ file with full configuration."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    save_data = {
        # Configuration
        'config/benchmark_type': config['benchmark_type'],
        'config/num_events': config['num_events'],
        'config/num_partitions': config['num_partitions'],
        'config/min_pmts_per_partition': config['min_pmts_per_partition'],
        'config/max_pmts_per_partition': config['max_pmts_per_partition'],
        'config/distribution': config['distribution'],
        'config/num_cls_tokens': config['num_cls_tokens'],
        'config/hidden_channels': config['hidden_channels'],
        'config/num_heads': config['num_heads'],
        'config/depth': config['depth'],
        'config/mlp_expansion_factor': config.get('mlp_expansion_factor', 2),
        'config/dropout': config.get('dropout', 0.0),
        'config/fused_qkv': config.get('fused_qkv', True),
        'config/db_precision': config.get('db_precision', False),
        'config/compiled': config.get('compiled', False),
        'config/compiling_mode': config.get('compiling_mode', 'default'),
        'config/device': config['device'],
        
        # Batch stats
        'batch/total_partitions': config.get('total_partitions', 0),
        'batch/total_nodes': config.get('total_nodes', 0),
        'batch/total_edges': config.get('total_edges', 0),
        'batch/avg_nodes_per_partition': config.get('avg_nodes_per_partition', 0),
        
        # Metadata
        'metadata/timestamp': timestamp,
        'metadata/seed': config.get('seed', 42),
    }
    
    # Add results for each model
    for model_name, model_results in results.items():
        prefix = f'results/{model_name}/'
        save_data[prefix + 'forward_time_ms'] = model_results['forward_time'] * 1000
        save_data[prefix + 'backward_time_ms'] = model_results['backward_time'] * 1000
        save_data[prefix + 'total_time_ms'] = model_results['total_time'] * 1000
        save_data[prefix + 'peak_memory_gb'] = model_results['peak_memory'] / 1e9
    
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    np.savez(filepath, **save_data)
    print(f"\n✓ Results saved to: {filepath}")



# ============================================================================
# Main Benchmark
# ============================================================================

def main():

    # Configuration
    num_events = 50
    num_partitions = 400
    min_pmts_per_partition = 10
    max_pmts_per_partition = 100
    distribution = 'realistic'
    num_cls_tokens = 3
    hidden_channels = 30
    num_heads = 5
    depth = 3
    mlp_expansion_factor = 2
    dropout = 0.0
    fused_qkv = True
    db_precision = False
    debug = False
    device = 'cuda'
    compiling = True
    compiling_mode = "default"
    save_folder = 'outputs'
    
    seed = 42
    set_seeds(seed)
    
    print("="*80)
    print("LOCAL PATCHGNN BENCHMARK: Sparse vs Dense Local Attention")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Batch size (events): {num_events}")
    print(f"  Num partitions: {num_partitions}")
    print(f"  Min partition size: {min_pmts_per_partition} PMTs")
    print(f"  Max partition size: {max_pmts_per_partition} PMTs")
    print(f"  CLS tokens per partition: {num_cls_tokens}")
    print(f"  Hidden channels: {hidden_channels}")
    print(f"  Num heads: {num_heads}")
    print(f"  Depth: {depth}")
    print(f"  Compiling: {compiling}")
    if compiling:
        print(f"  Compiling mode: {compiling_mode}")
    print(f"  Device: {device}")
    
    # Generate partition data
    print("\n" + "="*80)
    print("Generating Data")
    print("="*80)
    
    device_obj = torch.device(device)
    
    # Generate partition counts with realistic distribution
    if distribution == 'realistic':
        print(f"   Warning: using realistic distribution for partitions. "
              f"This mode does not use min_size and max_size parameters.")
    
    batched_partitions = generate_batched_partitions(
        batch_size=num_events,
        num_partitions=num_partitions,
        min_size=min_pmts_per_partition,
        max_size=max_pmts_per_partition,
        distribution=distribution,
        device=device_obj,
        seed=seed
    )
    
    # Use the generated partition counts directly
    partition_counts = batched_partitions['partition_counts']
    # Calculate statistics
    total_partitions = len(partition_counts)
    total_pmts = partition_counts.sum().item()
    total_nodes = total_pmts + (total_partitions * num_cls_tokens)
    avg_nodes_per_partition = (partition_counts.float().mean().item() + num_cls_tokens)
    
    print(f"\nData:")
    print(f"  Total partitions: {total_partitions}")
    print(f"  Total PMTs: {total_pmts}")
    print(f"  Total nodes (PMTs + CLS): {total_nodes}")
    print(f"  Avg nodes per partition: {avg_nodes_per_partition:.1f}")
    print(f"  Partition size range: {partition_counts.min().item()} - {partition_counts.max().item()} PMTs")
    print_partition_statistics(partition_counts)
    

    # Sparse batch (PyG format)
    sparse_batch = prepare_sparse_batch(
        batched_partitions=batched_partitions,
        num_cls_tokens=num_cls_tokens,
        hidden_channels=hidden_channels,
        device=device_obj
    )
    x_flat = sparse_batch['x']
    partition_counts = sparse_batch['partition_counts']
    
    # Dense batch (batched format)
    dense_batch = prepare_dense_batch(
        batched_partitions=batched_partitions,
        num_partitions=num_partitions,
        max_partition_size=max_pmts_per_partition,
        num_cls_tokens=num_cls_tokens,
        hidden_channels=hidden_channels,
        device=device_obj
    )
    x_batched = dense_batch['x_batched']
    num_active_per_batch = dense_batch['num_active_per_batch']
    
    print(f"  x_flat shape: {x_flat.shape}")
    print(f"  x_batched shape: {x_batched.shape}")
    
    # Pre-create edges and masks (done once, not in benchmark loop)
    print("\nPre-computing graph structures...")
    edge_index = create_star_edges_vectorized(partition_counts, num_cls_tokens, device_obj)
    print(f"  Total edges (star pattern): {edge_index.shape[1]}")
    print(f"  Edge index shape: {edge_index.shape}")
    
    masks = create_partition_mask_batched(
        batched_partitions=batched_partitions,
        num_partitions=num_partitions,
        max_partition_size=max_pmts_per_partition,
        num_cls_tokens=num_cls_tokens,
        device=device_obj
    )
    print(f"  Masks shape: {masks.shape}")

    # Create models
    print("\n" + "="*80)
    print("Creating Models")
    print("="*80)

    model_sparse = PatchGNNTransformer(
        num_cls_tokens=num_cls_tokens,
        hidden_channels=hidden_channels,
        spnode_hidden_channels=hidden_channels,
        num_heads=num_heads,
        depth=depth,
        mlp_expansion_factor=mlp_expansion_factor,
        dropout=dropout,
        fused_qkv=fused_qkv,
        db_precision=db_precision,
        debug=debug,
        device=device_obj
    )
    
    # Wrapper for sparse model to match benchmark signature
    class SparseModelWrapper(torch.nn.Module):
        def __init__(self, model, partition_counts):
            super().__init__()
            self.model = model
            self.partition_counts = partition_counts
        
        def forward(self, x, edge_index):
            return self.model(x, edge_index, self.partition_counts)
    
    model_sparse_wrapped = SparseModelWrapper(model_sparse, partition_counts)
    
    model_dense = PatchDenseTransformer(
        num_cls_tokens=num_cls_tokens,
        hidden_channels=hidden_channels,
        spnode_hidden_channels=hidden_channels,
        num_heads=num_heads,
        depth=depth,
        mlp_expansion_factor=mlp_expansion_factor,
        dropout=dropout,
        bias=True,
        db_precision=db_precision,
        debug=debug,
        device=device_obj
    )
    
    # Wrapper for dense model to match benchmark signature
    class DenseModelWrapper(torch.nn.Module):
        def __init__(self, model, num_active_per_batch):
            super().__init__()
            self.model = model
            self.num_active_per_batch = num_active_per_batch
        
        def forward(self, x_batched, masks):
            return self.model(x_batched, masks, self.num_active_per_batch)
    
    model_dense_wrapped = DenseModelWrapper(model_dense, num_active_per_batch)
    
    print(f"Model A (Patch GNN + Star Edges): {sum(p.numel() for p in model_sparse.parameters())} parameters")
    print(f"Model B (Patch Dense Transformer + Padded+Star Mask):      {sum(p.numel() for p in model_dense.parameters())} parameters")
    
    if compiling:
        print(f"\nCompiling models with mode='{compiling_mode}'...")
        model_sparse_wrapped = torch.compile(model_sparse_wrapped, mode=compiling_mode)
        model_dense_wrapped = torch.compile(model_dense_wrapped, mode=compiling_mode)
        print(f"Done compiling models.")
    else:
        print(f"\nNot using torch.compile (set compiling=True to enable)")
    
    # Benchmark
    print("\n" + "="*80)
    print("Benchmarking")
    print("="*80)
    
    print("\n[1] Sparse Local GNN (Vectorized Star Edges + Sparse Attention)")
    sparse_fwd, sparse_bwd, sparse_total, sparse_mem = benchmark_model(
        model_sparse_wrapped, x_flat, edge_index,
        num_iters=50, warmup=10
    )
    print(f"  Forward:  {sparse_fwd*1000:.3f} ms")
    print(f"  Backward: {sparse_bwd*1000:.3f} ms")
    print(f"  Total:    {sparse_total*1000:.3f} ms")
    print(f"  Memory:   {sparse_mem/1e9:.3f} GB")
    
    print("\n[2] Dense Local GNN (Vectorized Masking + PyTorch MHA)")
    dense_fwd, dense_bwd, dense_total, dense_mem = benchmark_model(
        model_dense_wrapped, x_batched, masks,
        num_iters=50, warmup=10
    )
    print(f"  Forward:  {dense_fwd*1000:.3f} ms")
    print(f"  Backward: {dense_bwd*1000:.3f} ms")
    print(f"  Total:    {dense_total*1000:.3f} ms")
    print(f"  Memory:   {dense_mem/1e9:.3f} GB")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Model':<50} {'Fwd (ms)':<12} {'Bwd (ms)':<12} {'Total (ms)':<12} {'Memory (GB)':<10}")
    print("-"*80)
    print(f"{'Sparse (Vectorized Edges)':<50} {sparse_fwd*1000:<12.3f} {sparse_bwd*1000:<12.3f} {sparse_total*1000:<12.3f} {sparse_mem/1e9:<10.3f}")
    print(f"{'Dense (Vectorized Masks + Torch MHA)':<50} {dense_fwd*1000:<12.3f} {dense_bwd*1000:<12.3f} {dense_total*1000:<12.3f} {dense_mem/1e9:<10.3f}")
    print("-"*80)
    print(f"\nSpeedup vs Sparse:")
    print(f"  Dense: {sparse_total/dense_total:.2f}x")
    print(f"\nMemory vs Sparse:")
    print(f"  Dense: {dense_mem/sparse_mem:.2f}x ({(dense_mem-sparse_mem)/1e9:+.3f} GB)")
    
    # Save results
    config = {
        'benchmark_type': 'local_patchgnn',
        'num_events': num_events,
        'num_partitions': num_partitions,
        'min_pmts_per_partition': min_pmts_per_partition,
        'max_pmts_per_partition': max_pmts_per_partition,
        'distribution': distribution,
        'num_cls_tokens': num_cls_tokens,
        'hidden_channels': hidden_channels,
        'num_heads': num_heads,
        'depth': depth,
        'mlp_expansion_factor': mlp_expansion_factor,
        'dropout': dropout,
        'fused_qkv': fused_qkv,
        'db_precision': db_precision,
        'compiled': compiling,
        'compiling_mode': compiling_mode,
        'device': device,
        'seed': seed,
        # Batch stats
        'total_partitions': total_partitions,
        'total_nodes': total_nodes,
        'avg_nodes_per_partition': avg_nodes_per_partition,
    }
    
    results = {
        'sparse': {
            'forward_time': sparse_fwd,
            'backward_time': sparse_bwd,
            'total_time': sparse_total,
            'peak_memory': sparse_mem,
        },
        'dense': {
            'forward_time': dense_fwd,
            'backward_time': dense_bwd,
            'total_time': dense_total,
            'peak_memory': dense_mem,
        },
    }
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_patchgnn_{timestamp}_bs{num_events}_np{num_partitions}_minp{min_pmts_per_partition}_maxp{max_pmts_per_partition}_hd{hidden_channels}.npz"
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

    import torch    
    from layers.patch_transformer import PatchDenseTransformer
    from layers.patch_gnn import PatchGNNTransformer
    from utils.set_seeds import set_seeds
    from utils.prepare_batch import prepare_sparse_batch, prepare_dense_batch, create_partition_mask_batched
    from pc_func.make_partitions import generate_batched_partitions, print_partition_statistics
    from pc_func.sparse_stars import create_star_edges_vectorized
    
    main()
