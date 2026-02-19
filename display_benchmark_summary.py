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
        print(f"  Batch size:              {data.get('config/batch_size', 'N/A')}")
        print(f"  Num partitions:          {data['config/num_partitions']}")
        print(f"  Avg partition size:      {data.get('config/avg_partition_size', 'N/A')}")
        print(f"  Sparsity:                {data.get('config/sparsity', 'N/A')}")
        print(f"  Max partition size:      {data.get('config/max_partition_size', 'N/A')}")
        if 'config/num_cls_tokens' in data:
            print(f"  Num CLS tokens:          {data['config/num_cls_tokens']}")
        print(f"  Use compile:             {data.get('config/use_compile', 'N/A')}")
        print(f"  Warmup iters:            {data.get('config/warmup', 'N/A')}")
        print(f"  Benchmark iters:         {data.get('config/num_iters', 'N/A')}")
    
    # Display batch stats
    print("\nBatch Statistics:")
    if is_transformer:
        print(f"  Total active partitions: {data.get('batch/total_active_partitions', 'N/A')}")
        if 'batch/min_active_per_event' in data:
            print(f"  Active per event (min):  {data['batch/min_active_per_event']}")
            print(f"  Active per event (max):  {data['batch/max_active_per_event']}")
            print(f"  Active per event (mean): {data['batch/mean_active_per_event']:.1f}")
    else:
        if 'batch_stats/total_nodes' in data:
            print(f"  Total nodes:             {data['batch_stats/total_nodes']}")
        if 'batch_stats/total_edges' in data:
            print(f"  Total edges:             {data['batch_stats/total_edges']}")
        if 'batch_stats/avg_nodes_per_partition' in data:
            print(f"  Avg nodes per partition: {data['batch_stats/avg_nodes_per_partition']:.1f}")
    
    # Display results
    print("\n" + "-" * 80)
    
    if is_transformer:
        print(f"{'Model':<35} {'Fwd (ms)':<12} {'Bwd (ms)':<12} {'Total (ms)':<12} {'Memory (GB)':<12}")
        print("-" * 80)
        
        # Detect which models are present
        models = ['edge_index', 'sparse', 'dense', 'torch_mha']
        model_names = {
            'edge_index': 'Sparse (Edge-Index)',
            'sparse': 'Sparse Transformer',
            'dense': 'Dense (Padded + Mask)',
            'torch_mha': 'Torch MHA (nn.MultiheadAttention)',
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
    else:
        # Edge creation benchmark - only time and memory
        print(f"{'Method':<35} {'Time (ms)':<15} {'Memory (MB)':<15}")
        print("-" * 80)
        
        # Detect which methods are present - check both compiled and non-compiled
        methods = ['loop', 'vectorized', 'dense_mask', 'loop_compiled', 'vectorized_compiled', 'dense_mask_compiled']
        method_names = {
            'loop': 'Loop (edges)',
            'vectorized': 'Vectorized (edges)',
            'dense_mask': 'Dense (padded masks)',
            'loop_compiled': 'Loop Compiled (edges)',
            'vectorized_compiled': 'Vectorized Compiled (edges)',
            'dense_mask_compiled': 'Dense Compiled (padded masks)',
        }
        
        results = {}
        for method in methods:
            try:
                time_key = f'{method}/time'
                mem_key = f'{method}/memory'
                
                if time_key in data:
                    time_val = float(data[time_key]) * 1000  # Convert to ms
                    mem_val = float(data[mem_key]) / 1e6     # Convert to MB
                    
                    results[method] = {'time': time_val, 'mem': mem_val}
                    print(f"{method_names.get(method, method):<35} {time_val:<15.3f} {mem_val:<15.2f}")
            except KeyError:
                continue
    
    print("-" * 80)
    
    # Speedup analysis
    if len(results) > 1:
        # Find baseline
        if is_transformer:
            baseline_model = 'edge_index' if 'edge_index' in results else list(results.keys())[0]
            baseline_val = results[baseline_model]['total']
            metric = 'total'
        else:
            # For edge creation, prefer compiled loop as baseline
            if 'loop_compiled' in results:
                baseline_model = 'loop_compiled'
            elif 'loop' in results:
                baseline_model = 'loop'
            else:
                baseline_model = list(results.keys())[0]
            baseline_val = results[baseline_model]['time']
            metric = 'time'
        
        baseline_mem = results[baseline_model].get('mem', 0)
        model_names_dict = model_names if is_transformer else method_names
        
        print(f"\nSpeedup vs {model_names_dict.get(baseline_model, baseline_model)}:")
        for model, res in results.items():
            if model != baseline_model:
                speedup = baseline_val / res[metric]
                print(f"  {model_names_dict.get(model, model):<30} {speedup:>6.2f}x")
        
        # Memory comparison
        if baseline_mem > 0:
            print(f"\nMemory vs {model_names_dict.get(baseline_model, baseline_model)}:")
            for model, res in results.items():
                if model != baseline_model and 'mem' in res:
                    mem_ratio = res['mem'] / baseline_mem
                    mem_diff = res['mem'] - baseline_mem
                    mem_unit = ' GB' if is_transformer else ' MB'
                    print(f"  {model_names_dict.get(model, model):<30} {mem_ratio:>6.2f}x ({mem_diff:+.2f}{mem_unit})")
    
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
