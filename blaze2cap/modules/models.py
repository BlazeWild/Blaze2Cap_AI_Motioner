import torch
import torch.nn as nn
import math

class LayerNorm(nn.LayerNorm):
    """
    Subclass torch's LayerNorm to handle fp16.
    Forces the normalization to run in float32 to prevent overflow/underflow,
    then casts back to the original dtype (fp16/bf16).
    """
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        # Always upcast to float32 for the normalization stats
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
    
    
class QuickGELU(nn.Module):
    """
    Quick Gaussian Error Linear Unit activation function.
    """
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)
    
class FeedForward(nn.Module):
    """
    Standard Point-wise Feed Forward Network (FFN).
    Input -> Linear -> GELU -> Dropout -> Linear -> Output
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.activation = QuickGELU()
        self.dropout = nn.Dropout(dropout)
        self.w_2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))
    
class CausalSelfAttention(nn.Module):
    """
    Calculates Attention with a fixed Triangular Mask (Causal) 
    AND a variable Padding Mask (Virtual Wall).
    """
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, key_padding_mask=None):
        B, N, C = x.shape
        
        # generate causal mask (time constraints)
        # prevents looking at future tokens/frames
        causal_mask = torch.triu(
            torch.ones(N, N, device=x.device, dtype=torch.bool), diagonal=1
        )
        if key_padding_mask is None:
            out, weights = self.attn(
                query=x,
                key=x,
                value=x,
                attn_mask=causal_mask,
                key_padding_mask=None,
                need_weights=True
            )
            return self.dropout(out), weights

        # Build a per-query mask to avoid all-masked rows (NaNs) on padded queries.
        # Mask rule: True = masked. Combine causal + padding keys.
        combined = causal_mask.unsqueeze(0) | key_padding_mask.unsqueeze(1)

        # For padded query positions, allow self-attend to avoid all-masked rows.
        diag = torch.eye(N, device=x.device, dtype=torch.bool).unsqueeze(0)
        combined = combined & ~(diag & key_padding_mask.unsqueeze(2))

        # Expand mask for all heads (expected shape: B*n_head, N, N)
        combined = combined.unsqueeze(1).expand(B, self.n_head, N, N)
        combined = combined.reshape(B * self.n_head, N, N)

        out, weights = self.attn(
            query=x,
            key=x,
            value=x,
            attn_mask=combined,
            key_padding_mask=None,
            need_weights=True
        )

        # Zero out padded queries so they do not leak into later layers.
        out = out.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return self.dropout(out), weights
    
class TransformerBlock(nn.Module):
    """
    A single layer containing the full Residual connection logic.
    Structure: Pre-Norm -> Attention -> Add -> Pre-Norm -> MLP -> Add
    """
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        # attention
        self.norm1 = LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, dropout)
        # feed forward
        self.norm2 = LayerNorm(d_model)
        self.mlp = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, key_padding_mask=None):
        # residual + attention
        residual = x #  residual
        x_norm = self.norm1(x) #pre layer norm
        # attnention out and weights
        attn_out, weights = self.attn(x_norm, key_padding_mask=key_padding_mask) # calulate attention
        x = residual + attn_out #residual connection
 
        # another residual + mlp
        residual = x
        x_norm = self.norm2(x) # pre layer norm
        mlp_out = self.mlp(x_norm) # feed forward
        x = residual + self.dropout(mlp_out) # residual connection
        
        #return output and attention weights
        return x, weights
    
class TemporalTransfomerEncoder(nn.Module):
    """ Stacking n_layers of TransformerBlock modules. """
    def __init__(self, num_layers, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_head, d_ff, dropout) for _ in range(num_layers)
        ])
        # final prenorm 
        self.norm_final = LayerNorm(d_model)
        
    def forward(self, x, key_padding_mask=None, return_all_weights = False):
        all_weights = []
        for layer in self.layers:
            x, weights = layer(x, key_padding_mask=key_padding_mask)
            if return_all_weights:
                all_weights.append(weights)
        
        x = self.norm_final(x)
        
        if return_all_weights:
            return x, all_weights
        return x


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding for Temporal Sequences.
    Injects position information into the input embeddings.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not learnable, but saved in state_dict)
        self.register_buffer('pe', pe.unsqueeze(0))  # Shape: (1, max_len, d_model)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [Batch, Seq_Len, D_Model]
        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:, :x.size(1), :]


class MotionTransformer(nn.Module):
    """
    Split-Head Transformer for Motion Prediction.
    
    Input: [Batch, Seq, 25 joints, 18 features]
    Output: 
        - Root trajectory: [Batch, Seq, 2, 6] (position delta + rotation delta)
        - Body pose: [Batch, Seq, 20, 6] (local 6D rotations)
    """
    def __init__(self, 
                 num_joints=25, 
                 input_feats=18, 
                 d_model=256, 
                 num_layers=4, 
                 n_head=4, 
                 d_ff=512, 
                 dropout=0.1, 
                 max_len=100):
        super().__init__()
        
        self.num_joints = num_joints
        self.input_feats = input_feats
        
        # 1. Input Projection
        # Flatten input (25 * 18 = 450) -> Project to d_model (256)
        input_dim = num_joints * input_feats
        self.input_projection = nn.Linear(input_dim, d_model)
        self.dropout_input = nn.Dropout(dropout)
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        
        # 3. Encoder Backbone
        self.encoder = TemporalTransfomerEncoder(
            num_layers=num_layers, 
            d_model=d_model, 
            n_head=n_head, 
            d_ff=d_ff, 
            dropout=dropout
        )
        
        # 4. Split-Head Decoder
        # Head A: Root Trajectory (2 joints × 6D = 12 outputs)
        # Joint 0: Root position delta (dx, dy, dz, pad, pad, pad)
        # Joint 1: Root rotation delta (6D representation)
        self.head_root = nn.Linear(d_model, 12)
        
        # Head B: Body Pose (20 joints × 6D = 120 outputs)
        # Local 6D rotations for child joints
        self.head_body = nn.Linear(d_model, 120)

    def forward(self, x, key_padding_mask=None):
        """
        Args:
            x: Input tensor of shape [Batch, Seq, 25, 18] or [Batch, Seq, 450]
            key_padding_mask: Optional mask [Batch, Seq] where True = padding
        Returns:
            root_out: [Batch, Seq, 2, 6] - Root trajectory
            body_out: [Batch, Seq, 20, 6] - Body pose
        """
        # Handle both 4D and 3D input
        if x.dim() == 4:
            B, S, J, F = x.shape
            # Flatten spatial dimensions: [B, S, 25, 18] -> [B, S, 450]
            x = x.view(B, S, -1)
        else:
            B, S, _ = x.shape
        
        # --- 1. Input Projection ---
        # [B, S, 450] -> [B, S, 256]
        x = self.input_projection(x)
        
        # --- 2. Add Positional Encoding ---
        x = self.pos_encoder(x)
        x = self.dropout_input(x)
        
        # --- 3. Transformer Encoder ---
        # latent: [B, S, 256]
        latent = self.encoder(x, key_padding_mask=key_padding_mask)
        
        # --- 4. Split-Head Decoding ---
        
        # Head A: Root Output
        # [B, S, 256] -> [B, S, 12] -> [B, S, 2, 6]
        root_out = self.head_root(latent)
        root_out = root_out.view(B, S, 2, 6)
        
        # Head B: Body Output  
        # [B, S, 256] -> [B, S, 120] -> [B, S, 20, 6]
        body_out = self.head_body(latent)
        body_out = body_out.view(B, S, 20, 6)
        
        return root_out, body_out
    
    def forward_combined(self, x, key_padding_mask=None):
        """
        Alternative forward that returns combined output matching GT format.
        Returns: [Batch, Seq, 22, 6]
        """
        root_out, body_out = self.forward(x, key_padding_mask)
        # Concatenate: [B, S, 2, 6] + [B, S, 20, 6] -> [B, S, 22, 6]
        return torch.cat([root_out, body_out], dim=2)