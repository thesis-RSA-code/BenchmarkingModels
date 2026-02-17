"""
Goal here is to benchmark the speed / memory usage of the DenseTransformer model 
depending on
- attention mechanism being used (fully connected edge index vs torch spda)
"""


"""
Benchmark: Global Transformer with Edge-Index vs Dense SDPA

Compares two approaches for global attention over super nodes:
- Model A: Sparse attention with fully-connected edge_index (SLOW)
- Model B: Dense batched attention with SDPA (FAST)

This isolates the global transformer bottleneck without PatchGNN complexity.
"""

import os
import time
import numpy as np
import torch

from utils.init_main import is_running_in_container


# ============================================================================
# Benchmarking
# ============================================================================

def benchmark_model(model, *args, num_iters=100, warmup=10):
    """
    Benchmark forward + backward pass.
    
    Returns:
        avg_forward_time, avg_backward_time, total_time, peak_memory
    """
    
    # Warmup
    for _ in range(warmup):
        output = model(*args)
        loss = output.sum()
        loss.backward()
        model.zero_grad()
    
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    forward_times = []
    backward_times = []
    
    for _ in range(num_iters):
        # Forward
        torch.cuda.synchronize()
        fwd_start = time.time()
        output = model(*args)
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
        'config/num_event': config['num_event'],
        'config/num_partitions': config['num_partitions'],
        'config/avg_pmts_per_event': config.get('avg_pmts_per_event', 0),
        'config/hidden_channels': config['hidden_channels'],
        'config/num_heads': config['num_heads'],
        'config/depth': config['depth'],
        'config/mlp_expansion_factor': config.get('mlp_expansion_factor', 2),
        'config/dropout': config.get('dropout', 0.0),
        'config/fused_qkv': config.get('fused_qkv', True),
        'config/use_sdpa': config.get('use_sdpa', True),
        'config/db_precision': config.get('db_precision', False),
        'config/compiled': config.get('compiled', False),
        'config/compiling_mode': config.get('compiling_mode', 'default'),
        'config/device': config['device'],
        
        # Batch stats
        'batch/total_active_partitions': config.get('total_active_partitions', 0),
        'batch/min_active_per_event': config.get('min_active_per_event', 0),
        'batch/max_active_per_event': config.get('max_active_per_event', 0),
        'batch/mean_active_per_event': config.get('mean_active_per_event', 0),
        
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
    num_event = 15
    num_partitions = 10
    avg_pmts_per_event = 50
    hidden_channels = 80
    num_heads = 8
    depth = 3
    mlp_expansion_factor = 2
    dropout = 0.0
    fused_qkv = True
    db_precision = True
    use_spda_for_dense = True
    debug = False
    device = 'cuda'
    compiling = False
    compiling_mode = "default"
    save_folder = 'outputs'

    
    seed = 42
    set_seeds(seed)
    
    print("="*80)
    print("TRANSFORMER BLOCK BENCHMARK: Edge-Index vs Dense SDPA")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Batch size: {num_event}")
    print(f"  Num partitions: {num_partitions}")
    print(f"  Hidden channels: {hidden_channels}")
    print(f"  Num heads: {num_heads}")
    print(f"  Depth: {depth}")
    print(f"  Device: {device}")
    
    # Generate dummy partition data with non-active partitions
    batched_partitions = generate_batched_partitions(
        batch_size=num_event,
        num_partitions=num_partitions,
        min_size=int(avg_pmts_per_event * 0.8),
        max_size=int(avg_pmts_per_event * 1.2),
        distribution='realistic',
        device=device,
        seed=seed
    )
    
    # For global transformer: we have ONE super node per active partition
    # Total super nodes = sum of active partitions across all events
    total_active_partitions = len(batched_partitions['partition_counts'])
    
    # Create super node features (one per active partition)
    x_super_flat = torch.randn(total_active_partitions, hidden_channels, device=device, requires_grad=False)
    
    # For dense model: pad to [num_event, num_partitions, hidden_channels]
    x_super_batched = torch.zeros(num_event, num_partitions, hidden_channels, device=device, requires_grad=False)
    offset = 0
    for b in range(num_event):
        num_active = batched_partitions['num_active_per_batch'][b].item()
        x_super_batched[b, :num_active] = x_super_flat[offset:offset+num_active]
        offset += num_active
    x_super_batched.requires_grad_(False)
    
    # Create fully connected edges for sparse model (only active partitions)
    edge_index_super = create_fully_connected_edges(
        batch_size=num_event, 
        num_active_per_batch=batched_partitions['num_active_per_batch'], 
        device=device
    )
    
    # Create mask for dense model (mask out non-active partitions)
    # Is this redundant with the x_super_batched padding ? 
    # Technically yes, conceptually no:
    # Information redundancy: Yes, both encode which partitions are active
    # Functional redundancy: No, they serve different purposes:
    # Padding: Makes the tensor shape regular for batched operations
    # Mask: Tells the attention mechanism which positions to compute/ignore
    # Otherwise we'd need the model architecture to handle ragged tensors. (See https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html#multiheadattention)

    mask_super = create_fully_connected_mask(
        batch_size=num_event, 
        num_partitions=num_partitions, 
        num_active_per_batch=batched_partitions['num_active_per_batch'], 
        device=device
    )
    
    # Create nested tensor for nested_dense model (no padding, no masking needed)
    # nested_x_list = []
    # offset = 0
    # for b in range(num_event):
    #     num_active = batched_partitions['num_active_per_batch'][b].item()
    #     nested_x_list.append(x_super_flat[offset:offset+num_active])
    #     offset += num_active
    
    # x_super_nested = torch.nested.nested_tensor(nested_x_list, layout=torch.jagged)
    
    print(f"\nData:")
    print(f"  Total active partitions: {total_active_partitions}")
    print(f"  Active per event: min={batched_partitions['num_active_per_batch'].min().item()}, "
          f"max={batched_partitions['num_active_per_batch'].max().item()}, "
          f"mean={batched_partitions['num_active_per_batch'].float().mean().item():.1f}")
    print(f"  x_super shape (flat):    {x_super_flat.shape}")
    print(f"  x_super shape (batched): {x_super_batched.shape}")
    # print(f"  x_super shape (nested):  {x_super_nested}")
    print(f"  edge_index shape:        {edge_index_super.shape}")
    print(f"  mask shape:              {mask_super.shape}")
    print(f"  Mask sparsity:           {mask_super.float().mean().item()*100:.1f}% True values")
    
    # Create models
    print("\n" + "="*80)
    print("Creating Models")
    print("="*80)
    
    model_sparse = TransformerBlock(
        hidden_channels=hidden_channels,
        kind="edge_index",
        num_heads=num_heads, 
        mlp_expansion_factor=mlp_expansion_factor,
        dropout=dropout,
        fused_qkv=fused_qkv,
        db_precision=db_precision,
        debug=debug,
        depth=depth
    ).to(device)
    
    model_dense = TransformerBlock(
        hidden_channels=hidden_channels,
        kind="dense",
        num_heads=num_heads,
        mlp_expansion_factor=mlp_expansion_factor,
        dropout=dropout,
        fused_qkv=fused_qkv,
        use_sdpa=use_spda_for_dense,
        db_precision=db_precision,
        debug=debug,
        depth=depth
    ).to(device)
    
    # model_nested = TransformerBlock(
    #     hidden_channels=hidden_channels,
    #     kind="nested_dense",
    #     num_heads=num_heads,
    #     mlp_expansion_factor=mlp_expansion_factor,
    #     dropout=dropout,
    #     bias=True,
    #     db_precision=db_precision,
    #     debug=debug,
    #     depth=depth
    # ).to(device)
    
    print(f"Model A (Sparse/Edge-Index): {sum(p.numel() for p in model_sparse.parameters())} parameters")
    print(f"Model B (Dense/Padded+Mask): {sum(p.numel() for p in model_dense.parameters())} parameters")
    # print(f"Model C (Nested):            {sum(p.numel() for p in model_nested.parameters())} parameters")

    if compiling:
        print(f"\nCompiling models...")
        model_sparse = torch.compile(model_sparse, mode=compiling_mode, fullgraph=True)
        # model_nested = torch.compile(model_nested, mode=compiling_mode, fullgraph=True)
        print(f"Done compiling models.")
        print(f"  Warning : Dense model uses internal compiled operations, so not compiling it.")

    # model_dense = torch.compile(model_dense, mode=compiling_mode, fullgraph=True)
    
    # Benchmark
    print("\n" + "="*80)
    print("Benchmarking")
    print("="*80)
    
    print("\n    Sparse Transformer (Edge-Index with Full Connectivity - Active Partitions Only)")
    sparse_fwd, sparse_bwd, sparse_total, sparse_mem = benchmark_model(
        model_sparse, x_super_flat, edge_index_super,
        num_iters=50, warmup=10
    )
    print(f"  Forward:  {sparse_fwd*1000:.3f} ms")
    print(f"  Backward: {sparse_bwd*1000:.3f} ms")
    print(f"  Total:    {sparse_total*1000:.3f} ms")
    print(f"  Memory:   {sparse_mem/1e9:.3f} GB")
    
    print("\n    Dense Transformer (Batched SDPA with Masking)")
    dense_fwd, dense_bwd, dense_total, dense_mem = benchmark_model(
        model_dense, x_super_batched, mask_super,
        num_iters=50, warmup=10
    )
    print(f"  Forward:  {dense_fwd*1000:.3f} ms")
    print(f"  Backward: {dense_bwd*1000:.3f} ms")
    print(f"  Total:    {dense_total*1000:.3f} ms")
    print(f"  Memory:   {dense_mem/1e9:.3f} GB")
    
    # print("\n    Nested Transformer (NestedTensor with torch.jagged layout)")
    # nested_fwd, nested_bwd, nested_total, nested_mem = benchmark_model(
    #     model_nested, x_super_nested,
    #     num_iters=50, warmup=10
    # )
    # print(f"  Forward:  {nested_fwd*1000:.3f} ms")
    # print(f"  Backward: {nested_bwd*1000:.3f} ms")
    # print(f"  Total:    {nested_total*1000:.3f} ms")
    # print(f"  Memory:   {nested_mem/1e9:.3f} GB")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Model':<40} {'Fwd (ms)':<12} {'Bwd (ms)':<12} {'Total (ms)':<12} {'Mem (GB)':<10}")
    print("-"*80)
    print(f"{'Sparse (Edge-Index)':<40} {sparse_fwd*1000:<12.3f} {sparse_bwd*1000:<12.3f} {sparse_total*1000:<12.3f} {sparse_mem/1e9:<10.3f}")
    print(f"{'Dense (Padded + Mask)':<40} {dense_fwd*1000:<12.3f} {dense_bwd*1000:<12.3f} {dense_total*1000:<12.3f} {dense_mem/1e9:<10.3f}")
    # print(f"{'Nested (torch.jagged)':<40} {nested_fwd*1000:<12.3f} {nested_bwd*1000:<12.3f} {nested_total*1000:<12.3f} {nested_mem/1e9:<10.3f}")
    print("-"*80)
    print(f"\nSpeedup vs Sparse (Edge-Index):")
    print(f"  Dense:  {sparse_total/dense_total:.2f}x")
    # print(f"  Nested: {sparse_total/nested_total:.2f}x")
    # print(f"\nSpeedup (Nested vs Dense): {dense_total/nested_total:.2f}x")
    print(f"\nMemory vs Sparse:")
    print(f"  Dense:  {(1 - dense_mem/sparse_mem)*100:+.1f}%")
    # print(f"  Nested: {(1 - nested_mem/sparse_mem)*100:+.1f}%")
    
    # Save results
    config = {
        'num_event': num_event,
        'num_partitions': num_partitions,
        'avg_pmts_per_event': avg_pmts_per_event,
        'hidden_channels': hidden_channels,
        'num_heads': num_heads,
        'depth': depth,
        'mlp_expansion_factor': mlp_expansion_factor,
        'dropout': dropout,
        'fused_qkv': fused_qkv,
        'use_sdpa': use_spda_for_dense,
        'db_precision': db_precision,
        'compiled': compiling,
        'compiling_mode': compiling_mode,
        'device': device,
        'seed': seed,
        # Batch stats
        'total_active_partitions': total_active_partitions,
        'min_active_per_event': batched_partitions['num_active_per_batch'].min().item(),
        'max_active_per_event': batched_partitions['num_active_per_batch'].max().item(),
        'mean_active_per_event': batched_partitions['num_active_per_batch'].float().mean().item(),
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
        # 'nested': {
        #     'forward_time': nested_fwd,
        #     'backward_time': nested_bwd,
        #     'total_time': nested_total,
    #         'peak_memory': nested_mem,
    #     },
    }
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_global_{timestamp}_bs{num_event}_np{num_partitions}_hd{hidden_channels}.npz"
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
        os.environ["HOME"] = cache_root  # Override HOME
        os.environ["TMPDIR"] = cache_root  # Override TMPDIR


    from layers.transformer import TransformerBlock
    from utils.set_seeds import set_seeds
    from pc_func.make_partitions import generate_batched_partitions
    from pc_func.make_fully_connected import create_fully_connected_edges, create_fully_connected_mask
    
    main()