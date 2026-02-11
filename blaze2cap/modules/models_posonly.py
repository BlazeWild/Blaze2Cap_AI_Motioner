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
        # Create constant 'pe' matrix with values dependent on pos and i
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: [Batch, Seq, d_model]
        # Add PE to input
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
        # x: [Batch, Seq, Feature]
        B, N, C = x.shape
        
        # Create Causal Mask (Triangular) if not provided
        if src_mask is None:
            # 1s on diagonal and below, -inf above.
            # PyTorch MHA expects boolean mask where TRUE = IGNORE
            src_mask = torch.triu(
                torch.ones(N, N, device=x.device, dtype=torch.bool), diagonal=1
            )

        out, weights = self.attn(
            query=x,
            key=x,
            value=x,
            attn_mask=src_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False, # We usually don't need weights for training
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
        # Pre-Norm Architecture
        x_norm = self.norm1(x)
        attn_out = self.attn(x_norm, src_mask=src_mask, key_padding_mask=key_padding_mask)
        x = x + attn_out # Residual 1

        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = x + self.dropout(mlp_out) # Residual 2
        return x

class MotionTransformer(nn.Module):
    """
    Simplified Canonical Motion Transformer.
    Input:  19 Joints (BlazePose) x 8 Channels [wx,wy,wz, vx,vy,vz, vis, anc]
    Output: 20 Joints (TotalCapture) x 3 Channels [x, y, z] (Positions)
    """
    def __init__(self, 
                 num_joints=19,      # Updated Input Joints
                 input_feats=8,      # Updated Input Channels
                 output_joints=20,   # Updated Output Joints
                 d_model=512, 
                 num_layers=6, 
                 n_head=8, 
                 d_ff=1024, 
                 dropout=0.1, 
                 max_len=512):
        super().__init__()
        
        # 1. Input Projection
        # Flattened Input: 19 * 8 = 152
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
        # Output: 20 * 3 = 60 values (Pure Position)
        self.output_dim = output_joints * 3
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, self.output_dim)
        )
        
        # Store output joint count for reshaping
        self.output_joints = output_joints

    def forward(self, x, src_mask=None, key_padding_mask=None):
        """
        x: [Batch, Seq, 19, 8]  or [Batch, Seq, 152]
        Returns: [Batch, Seq, 20, 3]
        """
        B, T = x.shape[0], x.shape[1]

        # Flatten input if needed: [B, T, 19, 8] -> [B, T, 152]
        if x.dim() == 4:
            x = x.reshape(B, T, -1)

        # 1. Embed
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.dropout_input(x)
        
        # 2. Transformer Layers
        for layer in self.layers:
            x = layer(x, src_mask=src_mask, key_padding_mask=key_padding_mask)
            
        x = self.norm_final(x)
        
        # 3. Project to Output
        output = self.output_head(x) # [B, T, 60]
        
        # Reshape to [Batch, Seq, 20, 3]
        return output.reshape(B, T, self.output_joints, 3)