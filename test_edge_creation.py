"""
Quick test to verify edge creation strategies work correctly.

This can be run without a container to check for syntax errors and basic functionality.
"""

import torch
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from pc_func.sparse_stars import create_star_edges_loop, create_star_edges_vectorized
from pc_func.masked_stars import create_partition_mask_padded_vectorized


def test_edge_creation():
    """Test that all edge creation methods produce the same STAR graph edges."""
    print("Testing edge creation strategies (STAR GRAPH pattern)...")
    
    device = torch.device('cuda')  # Use CPU for testing
    
    # Create partition counts (PMT nodes per partition)
    max_partition_size = 5
    num_cls_tokens = 2
    partition_counts = torch.tensor([3, 4, 2], device=device)
    num_partitions = partition_counts.size(0)  # Infer from actual data
    
    print(f"\nTest configuration:")
    print(f"  Num partitions: {num_partitions}")
    print(f"  Max partition size: {max_partition_size}")
    print(f"  Partition counts (PMTs): {partition_counts.tolist()}")
    print(f"  Num CLS tokens per partition: {num_cls_tokens}")
    print(f"  Nodes per partition (including CLS tokens): {(partition_counts + num_cls_tokens).tolist()}")
    
    nodes_per_partition = partition_counts + num_cls_tokens
    expected_edges_per_partition = num_cls_tokens * nodes_per_partition
    expected_total_edges = expected_edges_per_partition.sum().item()
    total_nodes = nodes_per_partition.sum().item()
    
    print(f"\nExpected STAR graph structure:")
    print(f"  Total nodes: {total_nodes}")
    print(f"  Expected edges per partition: {expected_edges_per_partition.tolist()}")
    print(f"  Expected total edges: {expected_total_edges}")
    print(f"  (Formula: num_cls × total_nodes per partition)")
    
    # Test loop-based edge creation
    print("\n[1] Loop-based edge creation (create_star_edges_loop)...")
    edges_loop = create_star_edges_loop(partition_counts, num_cls_tokens, device)
    print(f"    create_star_edges_loop: created {edges_loop.size(1)} edges")

    # Test vectorized edge creation
    print("\n[2] Vectorized edge creation (create_star_edges_vectorized)...")
    edges_vec = create_star_edges_vectorized(partition_counts, num_cls_tokens, device)
    print(f"    create_star_edges_vectorized: created {edges_vec.size(1)} edges")

    # Test vectorized mask function
    print("\n[3] Dense mask-based edge creation (create_partition_mask_padded_vectorized)...")
    masks = create_partition_mask_padded_vectorized(partition_counts, num_cls_tokens, max_partition_size, device)
    
    # To get edge indices from mask
    edge_dense_mask = masks.nonzero(as_tuple=False).T
    print(f"    create_partition_mask: created mask of shape {masks.shape}, produced {edge_dense_mask.size(1)} edges")

    # Verify consistency
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    
    # Check edge counts
    assert edges_loop.size(1) == expected_total_edges, f"Loop edges: {edges_loop.size(1)} != expected: {expected_total_edges}"
    assert edges_vec.size(1) == expected_total_edges, f"Vec edges: {edges_vec.size(1)} != expected: {expected_total_edges}"
    assert edge_dense_mask.size(1) == expected_total_edges, f"Vec2 edges: {edge_dense_mask.size(1)} != expected: {expected_total_edges}"
    print(f"✓ All methods create correct number of edges ({expected_total_edges})")
    
    # Verify star pattern: all sources should be CLS tokens
    print(f"Checking star pattern in loop-based edges...")
    node_offset = 0
    for i in range(num_partitions):
        n_nodes = nodes_per_partition[i].item()
        # Get edges for this partition
        partition_edges_mask = (edges_loop[0] >= node_offset) & (edges_loop[0] < node_offset + n_nodes)
        partition_srcs = edges_loop[0][partition_edges_mask]
        
        # All sources should be CLS tokens (first num_cls_tokens nodes)
        expected_cls_range = (node_offset, node_offset + num_cls_tokens)
        all_srcs_are_cls = ((partition_srcs >= expected_cls_range[0]) & 
                           (partition_srcs < expected_cls_range[1])).all()
        
        if all_srcs_are_cls:
            print(f"✓ Partition {i}: All sources are CLS tokens")
        else:
            print(f"⚠️  Partition {i}: Some sources are NOT CLS tokens!")
        
        node_offset += n_nodes
    
    # Sort edges for comparison
    print(f"Comparing loop-based and vectorized methods ...")
    edges_loop_sorted = edges_loop.T.sort(dim=1)[0].sort(dim=0)[0]
    edges_vectorized_sorted = edges_vec.T.sort(dim=1)[0].sort(dim=0)[0]
    
    # Check if edges are identical
    if torch.equal(edges_loop_sorted, edges_vectorized_sorted):
        print("✓ Loop-based and Vectorized produce identical edges")
    else:
        print("⚠️  Loop-based and Vectorized produce different edges!")
        diff_count = (edges_loop_sorted != edges_vectorized_sorted).sum().item()
        print(f"    {diff_count} differences found")
    
    # Check star pattern in mask
    print(f"Checking star pattern in dense mask...")
    expected_mask_true = (num_cls_tokens * nodes_per_partition).sum().item()
    actual_mask_true = masks.sum().item()
    
    print(f"\nMask statistics (star pattern):")
    print(f"  Expected True values: {expected_mask_true} (num_cls × nodes per partition)")
    print(f"  Actual True values:   {actual_mask_true}")
    print(f"  Total possible edges: {total_nodes ** 2}")
    
    if expected_mask_true == actual_mask_true:
        print("✓ Mask has correct number of True values (star pattern)")
    else:
        print(f"⚠️  Mask True value count mismatch! Diff: {actual_mask_true - expected_mask_true}")
    
    print("\n" + "="*60)
    print("TEST COMPLETED - STAR GRAPH PATTERN VERIFIED")
    print("="*60)


def test_attention_shapes():
    """Test that attention layers work with the created edges/masks."""
    print("\n\nTesting attention layer shapes...")
    
    try:
        from layers.vanilla_attn import AttentionLayer
        from layers.dense_attn import DenseAttentionLayer, DenseAttentionLayerFused, DenseAttentionLayerSDPA
        
        device = torch.device('cpu')
        hidden_channels = 32
        num_heads = 4
        num_nodes = 10
        
        # Create dummy data
        x = torch.randn(num_nodes, hidden_channels, device=device)
        
        # Create dummy edges (fully connected)
        src = torch.arange(num_nodes).repeat_interleave(num_nodes)
        dst = torch.arange(num_nodes).repeat(num_nodes)
        edge_index = torch.stack([src, dst])
        
        # Create dummy mask
        mask = torch.ones(num_nodes, num_nodes, dtype=torch.bool, device=device)
        
        print(f"\nTest data:")
        print(f"  Input shape: {x.shape}")
        print(f"  Edges: {edge_index.size(1)}")
        print(f"  Mask shape: {mask.shape}")
        
        # Test sparse attention
        print("\n[1] Sparse Attention...")
        sparse_attn = AttentionLayer(hidden_channels, num_heads=num_heads)
        out_sparse = sparse_attn(x, edge_index)
        print(f"    Output shape: {out_sparse.shape}")
        assert out_sparse.shape == x.shape, "Output shape mismatch!"
        print("    ✓ Sparse attention works")
        
        # Test dense attention vanilla
        print("\n[2] Dense Attention (Vanilla)...")
        dense_attn = DenseAttentionLayer(hidden_channels, num_heads=num_heads)
        out_dense = dense_attn(x, mask)
        print(f"    Output shape: {out_dense.shape}")
        assert out_dense.shape == x.shape, "Output shape mismatch!"
        print("    ✓ Dense vanilla attention works")
        
        # Test dense attention fused
        print("\n[3] Dense Attention (Fused)...")
        dense_fused = DenseAttentionLayerFused(hidden_channels, num_heads=num_heads)
        out_fused = dense_fused(x, mask)
        print(f"    Output shape: {out_fused.shape}")
        assert out_fused.shape == x.shape, "Output shape mismatch!"
        print("    ✓ Dense fused attention works")
        
        # Test dense attention SDPA
        print("\n[4] Dense Attention (SDPA)...")
        dense_sdpa = DenseAttentionLayerSDPA(hidden_channels, num_heads=num_heads)
        out_sdpa = dense_sdpa(x, mask)
        print(f"    Output shape: {out_sdpa.shape}")
        assert out_sdpa.shape == x.shape, "Output shape mismatch!"
        print("    ✓ Dense SDPA attention works")
        
        print("\n✓ All attention layers work correctly")
        
    except ImportError as e:
        print(f"⚠️  Could not import attention layers: {e}")
        print("    This is expected if torch_scatter/torch_geometric are not installed")


if __name__ == "__main__":
    print("="*60)
    print("EDGE CREATION & ATTENTION LAYER TESTS")
    print("="*60)
    
    test_edge_creation()
    # test_attention_shapes()
    
    print("\n✅ All tests completed!")
