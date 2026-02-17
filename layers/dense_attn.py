import torch.nn as nn
import torch.nn.functional as F


class DenseVanillaAttentionLayer(nn.Module):
    """
    Dense multi-head attention layer with optional fused QKV projection.
    
    Args:
        hidden_channels: hidden dimension size
        num_heads: number of attention heads
        fused_qkv: if True, uses single linear layer for QKV (faster)
        use_sdpa: if True, uses F.scaled_dot_product_attention (fastest)
    """
    
    def __init__(self, hidden_channels, num_heads=4, fused_qkv=True, use_sdpa=True, 
                 activation='relu', db_precision=False, debug=False):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads
        assert hidden_channels % num_heads == 0, "hidden_channels must be divisible by num_heads"
        
        self.fused_qkv = fused_qkv
        self.use_sdpa = use_sdpa
        self.db_precision = db_precision
        self.debug = debug
        
        # QKV projections
        if fused_qkv:
            self.qkv_proj = nn.Linear(hidden_channels, hidden_channels * 3)
        else:
            self.q_proj = nn.Linear(hidden_channels, hidden_channels)
            self.k_proj = nn.Linear(hidden_channels, hidden_channels)
            self.v_proj = nn.Linear(hidden_channels, hidden_channels)
        
        self.o_proj = nn.Linear(hidden_channels, hidden_channels)
    
    def forward(self, x, mask=None):
        """
        Forward pass with dense attention.
        
        Args:
            x: [N, hidden_channels] OR [B, N, hidden_channels]
            mask: [N, N] OR [B, N, N] boolean tensor (True = attend)
        
        Returns:
            out: same shape as input
        """
        # Handle both batched and unbatched input
        if x.dim() == 2:
            # Unbatched: [N, hidden_channels] -> add batch dim
            x = x.unsqueeze(0)  # [1, N, hidden_channels]
            if mask is not None:
                mask = mask.unsqueeze(0)  # [1, N, N]
            squeeze_output = True
        else:
            squeeze_output = False
        
        B, N, _ = x.shape
        
        # Get Q, K, V
        if self.fused_qkv:
            qkv = self.qkv_proj(x)  # [B, N, hidden_channels * 3]
            q, k, v = qkv.split([self.hidden_channels]*3, dim=-1)  # Each: [B, N, hidden_channels]
        else:
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
        
        # Reshape for multi-head: [B, N, hidden_channels] -> [B, num_heads, N, head_dim]
        if self.db_precision:
            Q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2).double()
            K = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2).double()
            V = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2).double()
        else:
            Q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            K = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            V = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        if self.use_sdpa:
            # SDPA wants [batch, num_heads, seq_len, head_dim]
            z = F.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=mask,  # mask should be [B, 1, N, N] or broadcastable
                dropout_p=0.0,
                is_causal=False
            )  # [B, num_heads, N, head_dim]
        else:
            # Manual attention computation with matmul - supposed to be faster than einsum
            scale = self.head_dim ** -0.5
            attn_scores = (q @ k.transpose(-2, -1)) * scale  # [B, num_heads, N, N]
            
            # Apply mask
            if mask is not None:
                # mask: [B, N, N] -> [B, 1, N, N] for broadcasting across heads
                if mask.dim() == 3:
                    mask = mask.unsqueeze(1)
                attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
            
            attn_weights = F.softmax(attn_scores, dim=-1)  # [B, num_heads, N, N]
            
            if self.debug:
                print(f"\n--- DenseAttention Debug ---")
                print(f"Q: mean={q.mean():.4f}, std={q.std():.4f}")
                print(f"K: mean={k.mean():.4f}, std={k.std():.4f}")
                print(f"V: mean={v.mean():.4f}, std={v.std():.4f}")
                print(f"Attn scores: mean={attn_scores.mean():.4f}, std={attn_scores.std():.4f}")
                print(f"Attn weights: mean={attn_weights.mean():.4f}, std={attn_weights.std():.4f}")
            
            z = attn_weights @ v  # [B, num_heads, N, head_dim]
        
        # Reshape back: [B, num_heads, N, head_dim] -> [B, N, hidden_channels]
        z = z.transpose(1, 2).reshape(B, N, self.hidden_channels).float()
        
        # Output projection
        out = self.o_proj(z)
        
        # Remove batch dim if input was unbatched
        if squeeze_output:
            out = out.squeeze(0)
        
        return out