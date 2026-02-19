import torch
from layers.transformer import TransformerBlock


"""

"""

class PatchDenseTransformer(torch.nn.Module):
    """Wrapper for patch dense transformer with masking and aggregation to supernodes.
    
    Uses proper batch dimensions [B, num_partitions, ...] unlike the sparse model.
    """
    
    def __init__(self, num_cls_tokens, hidden_channels, spnode_hidden_channels, num_heads, depth, 
                 mlp_expansion_factor, dropout, bias, db_precision, debug, device):
        super().__init__()

        self.num_cls_tokens = num_cls_tokens
        self.hidden_channels = hidden_channels
        self.spnode_hidden_channels = spnode_hidden_channels
        self.device = device
        
        self.transformer = TransformerBlock(
            hidden_channels=hidden_channels,
            kind="torch_mha",
            num_heads=num_heads,
            mlp_expansion_factor=mlp_expansion_factor,
            dropout=dropout,
            bias=bias,
            db_precision=db_precision,
            debug=debug,
            depth=depth
        ).to(device)
        
        # MLP for aggregating CLS tokens into supernodes (processes each partition independently)
        self.supernode_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * num_cls_tokens, hidden_channels * mlp_expansion_factor),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels * mlp_expansion_factor, spnode_hidden_channels),
        ).to(device)
    
    def forward(self, x_batched, masks, num_active_per_batch):
        """
        Forward with input features and masks, then aggregate CLS tokens to supernodes.
        num_partitions is the maximum possiblenumber of partitions per event (for padding purposes).
        
        Args:
            x_batched: [B, num_partitions, max_partition_size, hidden_channels] - input features (padded)
            masks: [B, num_partitions, max_partition_size, max_partition_size] - attention masks
            num_active_per_batch: [B] - number of active partitions per event
            
        Returns:
            supernodes: [B, num_partitions, spnode_hidden_channels] - supernodes (inactive partitions are zeros)
        """
        B, num_partitions, max_partition_size, hidden_channels = x_batched.shape
        
        # Flatten batch and partition dims for transformer: [B*num_partitions, max_partition_size, hidden_channels]
        x_flat = x_batched.reshape(B * num_partitions, max_partition_size, hidden_channels)
        masks_flat = masks.reshape(B * num_partitions, max_partition_size, max_partition_size)
        
        # Apply transformer with masking
        x_out = self.transformer(x_flat, mask=masks_flat)
        
        # Extract CLS tokens and aggregate to supernodes
        # x_out: [B*num_partitions, max_partition_size, hidden_channels]
        cls_tokens = x_out[:, :self.num_cls_tokens, :]  # [B*num_partitions, num_cls_tokens, hidden_channels]
        
        # Flatten CLS tokens per partition: [B*num_partitions, num_cls_tokens * hidden_channels]
        cls_flat = cls_tokens.reshape(B * num_partitions, -1)
        
        # Process all partitions through MLP in one batched call
        supernodes_flat = self.supernode_mlp(cls_flat)  # [B*num_partitions, spnode_hidden_channels]
        
        # Reshape back to batch dimensions: [B, num_partitions, spnode_hidden_channels]
        supernodes = supernodes_flat.reshape(B, num_partitions, self.spnode_hidden_channels)
        
        # Zero out inactive partitions
        # Create mask: [B, num_partitions]
        partition_indices = torch.arange(num_partitions, device=self.device).unsqueeze(0).expand(B, -1)
        active_mask = partition_indices < num_active_per_batch.unsqueeze(1)
        
        # Apply mask: [B, num_partitions, spnode_hidden_channels]
        supernodes = supernodes * active_mask.unsqueeze(-1)
        
        return supernodes
