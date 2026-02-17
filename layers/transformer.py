import torch.nn as nn
from utils.activation_functions import get_activation
from layers.dense_attn import DenseVanillaAttentionLayer
from layers.sparse_attn import SparseVanillaAttentionLayer


class DenseTransformerLayer(nn.Module):
    """Transformer layer with dense attention."""
    
    def __init__(self, hidden_channels, num_heads=4, activation='relu',
                 mlp_expansion_factor=2, dropout=0.0, fused_qkv=True, 
                 use_sdpa=True, db_precision=False, debug=False):
        
        super().__init__()
        self.attn = DenseVanillaAttentionLayer(
            hidden_channels, num_heads, 
            fused_qkv=fused_qkv, use_sdpa=use_sdpa,
            activation=activation, db_precision=db_precision, debug=debug)
        
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.dropout = nn.Dropout(dropout)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * mlp_expansion_factor),
            get_activation(activation),
            nn.Linear(hidden_channels * mlp_expansion_factor, hidden_channels),
        )
    
    @classmethod
    def from_config(cls, hidden_channels, num_heads, activation, 
                    mlp_expansion_factor, dropout, fused_qkv,
                    use_sdpa, db_precision, debug, **kwargs):
        """Create layer from full config, extracting only relevant parameters."""
        return cls(
            hidden_channels=hidden_channels,
            num_heads=num_heads,
            activation=activation,
            mlp_expansion_factor=mlp_expansion_factor,
            dropout=dropout,
            fused_qkv=fused_qkv,
            use_sdpa=use_sdpa,
            db_precision=db_precision,
            debug=debug,
        )
    
    def forward(self, x, mask=None):
        # x: [B, N, D]
        x = self.norm1(x + self.attn(x, mask))
        x = self.norm2(x + self.dropout(self.mlp(x)))
        return x


class SparseTransformerLayer(nn.Module):
    """Transformer layer with edge_index attention."""
    
    def __init__(self, hidden_channels, num_heads=4, activation='relu',
                 mlp_expansion_factor=2, dropout=0.0, fused_qkv=True, db_precision=False, 
                 debug=False):
        super().__init__()

        self.attn = SparseVanillaAttentionLayer(
            hidden_channels, num_heads, fused_qkv=fused_qkv,
            activation=activation, db_precision=db_precision, debug=debug)

        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.dropout = nn.Dropout(dropout)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * mlp_expansion_factor),
            get_activation(activation),
            nn.Linear(hidden_channels * mlp_expansion_factor, hidden_channels),
        )
    
    @classmethod
    def from_config(cls, hidden_channels, num_heads, activation,
                    mlp_expansion_factor, dropout, fused_qkv,
                    db_precision, debug, **kwargs):
        """Create layer from full config, extracting only relevant parameters."""
        return cls(
            hidden_channels=hidden_channels,
            num_heads=num_heads,
            activation=activation,
            mlp_expansion_factor=mlp_expansion_factor,
            dropout=dropout,
            fused_qkv=fused_qkv,
            db_precision=db_precision,
            debug=debug,
        )
    
    def forward(self, x, edge_index):
        x = self.norm1(x + self.attn(x, edge_index))
        x = self.norm2(x + self.dropout(self.mlp(x)))
        return x


class NestedTransformerLayer(nn.Module):
    """Transformer layer with nested tensor attention (torch.jagged layout)."""
    
    def __init__(self, hidden_channels, num_heads=4, activation='relu',
                 mlp_expansion_factor=2, dropout=0.0):

        super().__init__()
        # Use PyTorch's MultiHeadAttention for nested tensors
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_channels, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True, # See "in addition" from https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
            bias=False,
        )

        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.dropout = nn.Dropout(dropout)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * mlp_expansion_factor),
            get_activation(activation),
            nn.Linear(hidden_channels * mlp_expansion_factor, hidden_channels),
        )
    
    @classmethod
    def from_config(cls, hidden_channels, num_heads, activation,
                    mlp_expansion_factor, dropout, **kwargs):
        """Create layer from full config, extracting only relevant parameters."""
        return cls(
            hidden_channels=hidden_channels,
            num_heads=num_heads,
            activation=activation,
            mlp_expansion_factor=mlp_expansion_factor,
            dropout=dropout,
        )
    
    def forward(self, x):
        """
        Forward pass with nested tensor.
        
        Args:
            x: NestedTensor with torch.jagged layout
            
        Returns:
            x: NestedTensor with same structure
        """
        x = self.norm1(x + self.attn(x, x, x, is_causal=False))
        x = self.norm2(x + self.dropout(self.mlp(x)))
        return x


class TransformerBlock(nn.Module):
    """Stack of transformer layers supporting multiple attention types."""
    
    def __init__(self, hidden_channels, kind="dense", num_heads=4,
                 activation='relu', mlp_expansion_factor=2,
                 dropout=0.0, fused_qkv=True,
                 use_sdpa=True, db_precision=False,
                 depth=3, bias=True, debug=False):

        super().__init__()

        self.kind = kind
        
        # Map kind to layer class
        layer_classes = {
            "dense": DenseTransformerLayer,
            "edge_index": SparseTransformerLayer,
            "nested_dense": NestedTransformerLayer,
        }
        
        if kind not in layer_classes:
            raise ValueError(f"Invalid kind: {kind}. Must be one of {list(layer_classes.keys())}.")
        
        LayerClass = layer_classes[kind]
        
        # Create layers using from_config classmethod
        # Each layer extracts only the parameters it needs
        self.layers = nn.ModuleList([
            LayerClass.from_config(
                hidden_channels=hidden_channels,
                num_heads=num_heads,
                activation=activation,
                mlp_expansion_factor=mlp_expansion_factor,
                dropout=dropout,
                fused_qkv=fused_qkv,
                use_sdpa=use_sdpa,
                bias=bias,
                db_precision=db_precision,
                debug=debug,
            )
            for _ in range(depth)
        ])

    def forward(self, x, edge_index=None, mask=None):
        """
        Forward pass.
        
        Args:
            x: Input tensor (format depends on kind)
               - dense: [B, N, D] padded tensor
               - edge_index: [N, D] flat tensor
               - nested_dense: NestedTensor with torch.jagged layout
            edge_index: [2, E] for edge_index kind (optional)
            mask: [B, N, N] for dense kind (optional)
            
        Returns:
            x: Output in same format as input
        """
        for layer in self.layers:
            if self.kind == "dense":
                x = layer(x, mask)
            elif self.kind == "edge_index":
                x = layer(x, edge_index)
            elif self.kind == "nested_dense":
                x = layer(x)  # No edge_index or mask needed!

        return x
