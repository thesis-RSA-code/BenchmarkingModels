import torch.nn as nn
from torch_scatter import scatter_add
from torch_geometric.utils import softmax


class SparseVanillaAttentionLayer(nn.Module):
    def __init__(self, hidden_channels, num_heads=4, fused_qkv=True, activation='relu', 
                 db_precision=False, debug=False):
        super().__init__()
        self.db_precision = db_precision
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads
        assert hidden_channels % num_heads == 0, "hidden_channels must be divisible by num_heads"
        self.debug = debug
        self.fused_qkv = fused_qkv
        
        # QKV projections
        if fused_qkv:
            self.qkv_proj = nn.Linear(hidden_channels, hidden_channels * 3)
        else:
            self.q_proj = nn.Linear(hidden_channels, hidden_channels)
            self.k_proj = nn.Linear(hidden_channels, hidden_channels)
            self.v_proj = nn.Linear(hidden_channels, hidden_channels)
        
        self.o_proj = nn.Linear(hidden_channels, hidden_channels)
    
    def forward(self, x, edge_index):
        row, col = edge_index
        
        # Get Q, K, V
        if self.fused_qkv:
            qkv = self.qkv_proj(x)  # [N, hidden_channels * 3]
            q, k, v = qkv.chunk(3, dim=-1)  # Each: [N, hidden_channels]
        else:
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
        
        # Reshape for multi-head
        if self.db_precision:
            Q = q.view(-1, self.num_heads, self.head_dim).double()
            K = k.view(-1, self.num_heads, self.head_dim).double()
            V = v.view(-1, self.num_heads, self.head_dim).double()
        else:
            Q = q.view(-1, self.num_heads, self.head_dim)
            K = k.view(-1, self.num_heads, self.head_dim)
            V = v.view(-1, self.num_heads, self.head_dim)
        
        # Compute attention (sparse over edges)
        attn_scores = (Q[row] * K[col]).sum(dim=-1) / self.head_dim**0.5
        attn_weights = softmax(attn_scores, index=row, num_nodes=x.size(0))
        
        # Aggregate values
        weighted_v = attn_weights.unsqueeze(-1) * V[col]
        z = scatter_add(weighted_v, row, dim=0, dim_size=x.size(0))
        
        if self.debug:
            print("--- Debug Sparse Attention ---")
            print(f"Q mean: {Q.mean().item():.6f}, Q var: {Q.var().item():.6f}")
            print(f"K mean: {K.mean().item():.6f}, K var: {K.var().item():.6f}")
            print(f"V mean: {V.mean().item():.6f}, V var: {V.var().item():.6f}")
            print(f"attn_scores mean: {attn_scores.mean().item():.6f}, var: {attn_scores.var().item():.6f}")
            print(f"attn_weights mean: {attn_weights.mean().item():.6f}, var: {attn_weights.var().item():.6f}")
            print(f"weighted_v mean: {weighted_v.mean().item():.6f}, var: {weighted_v.var().item():.6f}")
            print(f"z mean: {z.mean().item():.6f}, var: {z.var().item():.6f}")
        
        # Concatenate heads and project
        z_concat = z.view(-1, self.num_heads * self.head_dim).float()
        out = self.o_proj(z_concat)
        
        return out