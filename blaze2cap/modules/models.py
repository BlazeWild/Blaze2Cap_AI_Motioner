import torch
import torch.nn as nn
import math

# ... (Keep LayerNorm, QuickGELU, FeedForward, CausalSelfAttention, TransformerBlock, TemporalTransfomerEncoder, PositionalEncoding as before) ...
# (Copy them from previous responses or keep your existing ones, they are fine)

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
    Split-Head Transformer.
    Input: 27 Joints * 19 Feats
    Output: 22 Joints * 6 Feats
      - Idx 0: Root Lin Vel [vx, vy, vz, 0, 0, 0]
      - Idx 1: Root Ang Vel [6D Rot]
      - Idx 2-21: Body Rot [6D Rot]
    """
    def __init__(self, 
                 num_joints=27, 
                 input_feats=19, # Corrected Default
                 num_joints_out=22, # Corrected Default (Includes Root)
                 d_model=512, 
                 num_layers=6, 
                 n_head=8, 
                 d_ff=1024, 
                 dropout=0.1, 
                 max_len=512):
        super().__init__()
        
        # Input Projection
        self.input_projection = nn.Linear(num_joints * input_feats, d_model)
        self.dropout_input = nn.Dropout(dropout)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        
        # Encoder
        self.encoder = TemporalTransfomerEncoder(num_layers, d_model, n_head, d_ff, dropout)
        
        # --- SPLIT HEADS ---
        
        # Head A: Root Motion (Predicts 3 LinVel + 6 AngVel = 9 values)
        self.head_root = nn.Linear(d_model, 9)
        
        # Head B: Body Motion (Predicts 20 joints * 6D Rot = 120 values)
        self.head_body = nn.Linear(d_model, 20 * 6)

    def forward(self, x):
        B, S = x.shape[0], x.shape[1]
        
        # Flatten Input
        if x.dim() == 4: x = x.view(B, S, -1)
        
        # Encode
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.dropout_input(x)
        latent = self.encoder(x) # [B, S, d_model]
        
        # --- DECODE ROOT ---
        root_raw = self.head_root(latent) # [B, S, 9]
        
        # Split: [vx, vy, vz] and [rot_6d]
        pred_lin_vel = root_raw[:, :, 0:3] 
        pred_ang_vel = root_raw[:, :, 3:9]
        
        # Pad LinVel to 6D: [vx, vy, vz, 0, 0, 0]
        pad_zeros = torch.zeros_like(pred_lin_vel)
        root_lin_vel_6d = torch.cat([pred_lin_vel, pad_zeros], dim=2) # [B, S, 6]
        
        # Stack Root: Index 0 (Lin), Index 1 (Ang) -> [B, S, 2, 6]
        root_out = torch.stack([root_lin_vel_6d, pred_ang_vel], dim=2)
        
        # --- DECODE BODY ---
        body_out = self.head_body(latent).view(B, S, 20, 6) # [B, S, 20, 6]
        
        # --- CONCATENATE ---
        # Final Output: [B, S, 22, 6]
        full_out = torch.cat([root_out, body_out], dim=2)
        
        return full_out