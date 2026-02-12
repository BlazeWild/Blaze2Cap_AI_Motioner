import torch
import torch.nn as nn
import math

# --- LAYERS (Standard Transformer Components) ---

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape, eps=eps)
    def forward(self, x): return self.layer_norm(x)

class QuickGELU(nn.Module):
    def forward(self, x): return x * torch.sigmoid(1.702 * x)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.activation = QuickGELU()
        self.dropout = nn.Dropout(dropout)
        self.w_2 = nn.Linear(d_ff, d_model)
    def forward(self, x): return self.w_2(self.dropout(self.activation(self.w_1(x))))

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        causal_mask = torch.triu(torch.ones(N, N, device=x.device, dtype=torch.bool), diagonal=1)
        out, _ = self.attn(query=x, key=x, value=x, attn_mask=causal_mask, need_weights=False, is_causal=True)
        return self.dropout(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, dropout)
        self.norm2 = LayerNorm(d_model)
        self.mlp = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x

class TemporalTransfomerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(d_model, n_head, d_ff, dropout) for _ in range(num_layers)])
        self.norm_final = LayerNorm(d_model)
    def forward(self, x):
        for layer in self.layers: x = layer(x)
        return self.norm_final(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1), :]

# --- MAIN MODEL ---

class MotionTransformer(nn.Module):
    """
    Blaze2Cap Motion Transformer.
    
    Inputs:
      - 27 Joints (BlazePose + Virtual)
      - 20 Features (Canonical Pose + 6D Alignment)
      
    Outputs:
      - 21 Joints (TotalCapture subset, excluding World Root)
      - 6 Features (6D Rotation)
        * Index 0: Hip/Pelvis Orientation Delta
        * Index 1-20: Body Joint Rotations
    """
    def __init__(self, 
                 num_joints=28, 
                 input_feats=14, # UPDATED: 20 features
                 num_joints_out=21, # UPDATED: 21 output joints (Index 0=Hip, 1-20=Body)
                 d_model=512, 
                 num_layers=6, 
                 n_head=8, 
                 d_ff=1024, 
                 dropout=0.1, 
                 max_len=512):
        super().__init__()
        
        # 1. Input Projection
        # Flattens (Batch, Time, 27*20) -> (Batch, Time, d_model)
        self.input_projection = nn.Linear(num_joints * input_feats, d_model)
        self.dropout_input = nn.Dropout(dropout)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        
        # 2. Encoder
        self.encoder = TemporalTransfomerEncoder(num_layers, d_model, n_head, d_ff, dropout)
        
        # 3. Output Heads (Split)
        
        # Head A: Hip Orientation (Index 0)
        # Predicts 6 values (1 Joint * 6D)
        self.head_hip = nn.Linear(d_model, 6)
        
        # Head B: Body Pose (Indices 1-20)
        # Predicts 120 values (20 Joints * 6D)
        self.head_body = nn.Linear(d_model, 20 * 6)

    def forward(self, x):
        B, S = x.shape[0], x.shape[1]
        
        # Flatten Input: (B, S, 27, 20) -> (B, S, 540)
        if x.dim() == 4: x = x.view(B, S, -1)
        
        # Input Embedding
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.dropout_input(x)
        
        # Transformer Pass
        latent = self.encoder(x) # [B, S, d_model]
        
        # --- DECODE HEADS ---
        
        # 1. Hip Head -> (B, S, 6) -> Reshape to (B, S, 1, 6)
        hip_out = self.head_hip(latent).view(B, S, 1, 6)
        
        # 2. Body Head -> (B, S, 120) -> Reshape to (B, S, 20, 6)
        body_out = self.head_body(latent).view(B, S, 20, 6)
        
        # --- CONCATENATE ---
        # Final Output: [B, S, 21, 6]
        # Index 0 is Hip, Indices 1-20 are Body
        full_out = torch.cat([hip_out, body_out], dim=2)
        
        return full_out