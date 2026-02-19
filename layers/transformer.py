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


class TorchMHATransformerLayer(nn.Module):
    """Transformer layer using PyTorch's nn.MultiheadAttention."""
    
    def __init__(self, hidden_channels, num_heads=4, activation='relu',
                 mlp_expansion_factor=2, dropout=0.0, bias=True, db_precision=False, 
                 debug=False):

        super().__init__()
        # Use PyTorch's standard MultiheadAttention
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            batch_first=True  # Input is [B, N, D]
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
                    mlp_expansion_factor, dropout, bias,
                    db_precision, debug, **kwargs):
        """Create layer from full config, extracting only relevant parameters."""
        return cls(
            hidden_channels=hidden_channels,
            num_heads=num_heads,
            activation=activation,
            mlp_expansion_factor=mlp_expansion_factor,
            dropout=dropout,
            bias=bias,
            db_precision=db_precision,
            debug=debug,
        )
    
    def forward(self, x, mask=None):
        """
        Forward pass.
        
        Args:
            x: [B, N, D] padded tensor
            mask: Optional [B, N, N] boolean mask (True = attend, False = ignore)
            
        Returns:
            x: [B, N, D] output tensor
        """
        # nn.MultiheadAttention expects key_padding_mask: [B, N] where True positions are ignored
        # Our mask: [B, N, N] where True means can attend
        # Convert to key_padding_mask format
        key_padding_mask = None
        if mask is not None:
            # A key position is padding if it cannot attend to anything (all False in that column)
            key_padding_mask = ~mask[:, 0, :]  # [B, N] - invert to get padding positions
        
        # Self-attention: query = key = value
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.norm1(x + attn_out)
        
        # MLP
        mlp_out = self.dropout(self.mlp(x))
        x = self.norm2(x + mlp_out)
        
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
            "torch_mha": TorchMHATransformerLayer,
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
               - torch_mha: [B, N, D] padded tensor
            edge_index: [2, E] for edge_index kind (optional)
            mask: [B, N, N] for dense/torch_mha kinds (optional)
            
        Returns:
            x: Output in same format as input
        """
        for layer in self.layers:
            if self.kind == "dense":
                x = layer(x, mask)
            elif self.kind == "edge_index":
                x = layer(x, edge_index)
            elif self.kind == "torch_mha":
                x = layer(x, mask)

        return x
