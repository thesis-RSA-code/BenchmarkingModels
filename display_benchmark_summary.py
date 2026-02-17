"""
Display summary of benchmark results from NPZ files.

Supports both edge creation benchmarks and transformer benchmarks.

Usage:
    python display_benchmark_summary.py outputs/benchmark_*.npz
"""

import sys
import numpy as np
from pathlib import Path


def load_and_display(npz_path):
    """Load and display benchmark results from NPZ file."""
    data = np.load(npz_path, allow_pickle=True)
    
    print("=" * 80)
    print(f"Benchmark Results: {Path(npz_path).name}")
    print("=" * 80)
    
    # Display configuration
    print("\nConfiguration:")
    
    # Check if it's a transformer benchmark or edge creation benchmark
    is_transformer = 'config/depth' in data
    
    if is_transformer:
        # Transformer benchmark
        print(f"  Benchmark type:          Transformer")
        print(f"  Batch size (events):     {data.get('config/num_event', data.get('config/batch_size', 'N/A'))}")
        print(f"  Num partitions:          {data['config/num_partitions']}")
        if 'config/avg_pmts_per_event' in data:
            print(f"  Avg PMTs per event:      {data['config/avg_pmts_per_event']}")
        print(f"  Hidden channels:         {data['config/hidden_channels']}")
        print(f"  Num heads:               {data['config/num_heads']}")
        print(f"  Depth (layers):          {data['config/depth']}")
        print(f"  MLP expansion factor:    {data.get('config/mlp_expansion_factor', 'N/A')}")
        print(f"  Dropout:                 {data.get('config/dropout', 'N/A')}")
        print(f"  Fused QKV:               {data.get('config/fused_qkv', 'N/A')}")
        print(f"  Use SDPA:                {data.get('config/use_sdpa', 'N/A')}")
        print(f"  DB precision:            {data.get('config/db_precision', 'N/A')}")
        print(f"  Compiled:                {data.get('config/compiled', 'N/A')}")
        if 'config/compiling_mode' in data:
            print(f"  Compiling mode:          {data['config/compiling_mode']}")
    else:
        # Edge creation benchmark
        print(f"  Benchmark type:          Edge Creation")
        print(f"  Num partitions:          {data['config/num_partitions']}")
        print(f"  Avg nodes per partition: {data.get('config/avg_nodes_per_partition', 'N/A')}")
        print(f"  Hidden channels:         {data['config/hidden_channels']}")
        print(f"  Num heads:               {data['config/num_heads']}")
        if 'config/num_cls_tokens' in data:
            print(f"  Num CLS tokens:          {data['config/num_cls_tokens']}")
        print(f"  DB precision:            {data.get('config/db_precision', 'N/A')}")
        print(f"  Compiled:                {data.get('config/compiled', 'N/A')}")
    
    # Display batch stats
    print("\nBatch Statistics:")
    if is_transformer:
        print(f"  Total active partitions: {data.get('batch/total_active_partitions', 'N/A')}")
        if 'batch/min_active_per_event' in data:
            print(f"  Active per event (min):  {data['batch/min_active_per_event']}")
            print(f"  Active per event (max):  {data['batch/max_active_per_event']}")
            print(f"  Active per event (mean): {data['batch/mean_active_per_event']:.1f}")
    else:
        if 'batch/total_nodes' in data:
            print(f"  Total nodes:     {data['batch/total_nodes']}")
        if 'batch/total_cls' in data:
            print(f"  Total CLS:       {data['batch/total_cls']}")
        if 'batch/total_pmts' in data:
            print(f"  Total PMTs:      {data['batch/total_pmts']}")
    
    # Display results
    print("\n" + "-" * 80)
    print(f"{'Model':<35} {'Fwd (ms)':<12} {'Bwd (ms)':<12} {'Total (ms)':<12} {'Memory (GB)':<12}")
    print("-" * 80)
    
    # Detect which models are present
    if is_transformer:
        models = ['edge_index', 'sparse', 'dense', 'nested']
        model_names = {
            'edge_index': 'Sparse (Edge-Index)',
            'sparse': 'Sparse Transformer',
            'dense': 'Dense (Padded + Mask)',
            'nested': 'Nested (torch.jagged)',
        }
    else:
        models = ['sparse_loop', 'sparse_vec', 'dense_vanilla', 'dense_fused', 'dense_sdpa']
        model_names = {
            'sparse_loop': 'Sparse (Loop edges)',
            'sparse_vec': 'Sparse (Vec edges)',
            'dense_vanilla': 'Dense (Vanilla + Mask)',
            'dense_fused': 'Dense (Fused + Mask)',
            'dense_sdpa': 'Dense (SDPA)',
        }
    
    results = {}
    for model in models:
        try:
            fwd_key = f'results/{model}/forward_time_ms'
            bwd_key = f'results/{model}/backward_time_ms'
            total_key = f'results/{model}/total_time_ms'
            mem_key = f'results/{model}/peak_memory_gb'
            
            if fwd_key in data:
                fwd = float(data[fwd_key])
                bwd = float(data[bwd_key])
                total = float(data[total_key])
                mem = float(data[mem_key])
                
                results[model] = {'fwd': fwd, 'bwd': bwd, 'total': total, 'mem': mem}
                print(f"{model_names.get(model, model):<35} {fwd:<12.3f} {bwd:<12.3f} {total:<12.3f} {mem:<12.3f}")
        except KeyError:
            continue
    
    print("-" * 80)
    
    # Edge creation times (for edge creation benchmarks)
    if not is_transformer:
        print("\nEdge Creation Times:")
        if 'results/sparse_loop/edge_creation_time_ms' in data:
            loop_edge = float(data['results/sparse_loop/edge_creation_time_ms'])
            print(f"  Loop-based:    {loop_edge:.3f} ms")
        if 'results/sparse_vec/edge_creation_time_ms' in data:
            vec_edge = float(data['results/sparse_vec/edge_creation_time_ms'])
            print(f"  Vectorized:    {vec_edge:.3f} ms")
            if 'results/sparse_loop/edge_creation_time_ms' in data:
                speedup = loop_edge / vec_edge
                print(f"  Speedup:       {speedup:.2f}x")
    
    # Speedup analysis
    if len(results) > 1:
        # Use first model as baseline
        baseline_model = list(results.keys())[0]
        baseline_total = results[baseline_model]['total']
        baseline_mem = results[baseline_model]['mem']
        
        print(f"\nSpeedup vs {model_names.get(baseline_model, baseline_model)}:")
        for model, res in results.items():
            if model != baseline_model:
                speedup = baseline_total / res['total']
                print(f"  {model_names.get(model, model):<30} {speedup:>6.2f}x")
        
        # Memory comparison
        print(f"\nMemory vs {model_names.get(baseline_model, baseline_model)}:")
        for model, res in results.items():
            if model != baseline_model:
                mem_ratio = res['mem'] / baseline_mem
                mem_diff = res['mem'] - baseline_mem
                print(f"  {model_names.get(model, model):<30} {mem_ratio:>6.2f}x ({mem_diff:+.3f} GB)")
    
    print("\n" + "=" * 80 + "\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python display_benchmark_summary.py <npz_file> [<npz_file2> ...]")
        print("\nExample:")
        print("  python display_benchmark_summary.py outputs/benchmark_*.npz")
        sys.exit(1)
    
    npz_files = sys.argv[1:]
    
    for npz_file in npz_files:
        if not Path(npz_file).exists():
            print(f"Warning: File not found: {npz_file}")
            continue
        
        load_and_display(npz_file)


if __name__ == "__main__":
    main()
