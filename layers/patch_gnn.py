import torch
from layers.transformer import TransformerBlock


class PatchGNNTransformer(torch.nn.Module):
    """Wrapper for patch GNN transformer with star edge creation and aggregation to supernodes."""
    
    def __init__(self, num_cls_tokens, hidden_channels, spnode_hidden_channels,
                 num_heads, depth, mlp_expansion_factor, dropout, fused_qkv,
                 db_precision, debug, device):
        super().__init__()
        
        self.num_cls_tokens = num_cls_tokens
        self.spnode_hidden_channels = spnode_hidden_channels
        self.device = device
        
        self.transformer = TransformerBlock(
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
        
        # MLP for aggregating CLS tokens into supernodes
        self.supernode_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * num_cls_tokens, hidden_channels * mlp_expansion_factor),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels * mlp_expansion_factor, spnode_hidden_channels),
        ).to(device)
    
    def forward(self, x, edge_index, partition_counts):
        """
        Forward with input features, edge_index, and partition info, then aggregate CLS tokens to supernodes.
        
        Args:
            x: [total_nodes, hidden_channels] - flat input features for all nodes across all partitions
            edge_index: [2, num_edges] - edge indices
            partition_counts: [total_active_partitions] - number of PMTs per partition (flatten across batch). 
                               Needed to extract the cls tokens from the hit pmt node features.
            
        Returns:
            supernodes: [total_active_partitions, spnode_hidden_channels] - aggregated supernodes (flat across batch)
            
        Note:
            The output is flat across the batch. Use batch indices to separate events if needed.
            In sparse graph representations (like PyG), this is the standard format.
        """
        # Apply transformer
        x_out = self.transformer(x, edge_index=edge_index)
        
        # Extract CLS tokens and aggregate to supernodes (fully vectorized, no loops)
        # CLS tokens are the first num_cls_tokens nodes in each partition
        num_partitions = partition_counts.size(0)
        nodes_per_partition = partition_counts + self.num_cls_tokens
        hidden_channels = x_out.size(1)
        
        # Compute starting indices for each partition (vectorized)
        partition_starts = torch.cat([
            torch.tensor([0], device=self.device),
            torch.cumsum(nodes_per_partition[:-1], dim=0)
        ])
        
        # Create indices for all CLS tokens across all partitions
        # For each partition, we want indices [start, start+1, ..., start+num_cls_tokens-1]
        cls_indices = partition_starts.unsqueeze(1) + torch.arange(
            self.num_cls_tokens, device=self.device
        ).unsqueeze(0)  # [num_partitions, num_cls_tokens]
        
        # Flatten indices and gather all CLS tokens at once
        cls_indices_flat = cls_indices.flatten()  # [num_partitions * num_cls_tokens]
        cls_tokens = x_out[cls_indices_flat]  # [num_partitions * num_cls_tokens, hidden_channels]
        
        # Reshape to [num_partitions, num_cls_tokens * hidden_channels]
        cls_tokens = cls_tokens.reshape(num_partitions, -1)
        
        # Process all partitions through MLP in one batched call
        supernodes = self.supernode_mlp(cls_tokens)  # [num_partitions, spnode_hidden_channels]
        
        return supernodes