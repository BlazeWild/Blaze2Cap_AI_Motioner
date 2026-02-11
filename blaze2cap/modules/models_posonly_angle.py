import torch
import torch.nn as nn
import math

class LayerNorm(nn.Module):
    """
    Standard LayerNorm. 
    """
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape, eps=eps)

    def forward(self, x):
        return self.layer_norm(x)

class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.activation = QuickGELU()
        self.dropout = nn.Dropout(dropout)
        self.w_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class CausalSelfAttention(nn.Module):
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

    def forward(self, x, src_mask=None, key_padding_mask=None):
        B, N, C = x.shape
        if src_mask is None:
            src_mask = torch.triu(
                torch.ones(N, N, device=x.device, dtype=torch.bool), diagonal=1
            )
        out, weights = self.attn(
            query=x, key=x, value=x,
            attn_mask=src_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=True 
        )
        return self.dropout(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, dropout)
        self.norm2 = LayerNorm(d_model)
        self.mlp = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None, key_padding_mask=None):
        x_norm = self.norm1(x)
        attn_out = self.attn(x_norm, src_mask=src_mask, key_padding_mask=key_padding_mask)
        x = x + attn_out
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = x + self.dropout(mlp_out)
        return x

class MotionTransformer(nn.Module):
    """
    Canonical Motion Transformer (Rotation Output).
    Input:  19 Joints x 14 Channels [pos, vel, parent, child, vis, anc]
    Output: 20 Joints x 6 Channels (6D Rotations)
    """
    def __init__(self, 
                 num_joints=19,      
                 input_feats=14,     # CHANGED: 8 -> 14
                 output_joints=20,   
                 d_model=512, 
                 num_layers=6, 
                 n_head=8, 
                 d_ff=1024, 
                 dropout=0.1, 
                 max_len=512):
        super().__init__()
        
        # 1. Input Projection
        # Flattened Input: 19 * 14 = 266
        self.input_dim = num_joints * input_feats
        self.embedding = nn.Linear(self.input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.dropout_input = nn.Dropout(dropout)
        
        # 2. Transformer Encoder
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_head, d_ff, dropout) for _ in range(num_layers)
        ])
        self.norm_final = LayerNorm(d_model)
        
        # 3. Output Projection
        # CHANGED: Output 6 values (Rotation) per joint instead of 3 (Position)
        # Output: 20 * 6 = 120 values
        self.output_dim = output_joints * 6 
        
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, self.output_dim)
        )
        
        self.output_joints = output_joints

    def forward(self, x, src_mask=None, key_padding_mask=None):
        """
        x: [Batch, Seq, 19, 14] -> Flattened internally
        Returns: [Batch, Seq, 20, 6] (Rotations)
        """
        B, T = x.shape[0], x.shape[1]

        if x.dim() == 4:
            x = x.reshape(B, T, -1)

        # Embed
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.dropout_input(x)
        
        # Transformer
        for layer in self.layers:
            x = layer(x, src_mask=src_mask, key_padding_mask=key_padding_mask)
            
        x = self.norm_final(x)
        
        # Project
        output = self.output_head(x) # [B, T, 120]
        
        # Reshape to (B, T, 20, 6)
        return output.reshape(B, T, self.output_joints, 6)