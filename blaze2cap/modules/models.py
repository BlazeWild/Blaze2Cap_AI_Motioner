import torch
import torch.nn as nn
import math

class LayerNorm(nn.Module):
    """
    Standard LayerNorm. 
    Note: PyTorch's native nn.LayerNorm already handles mixed precision (autocast) 
    correctly in recent versions. We don't need to force float32 manually unless 
    you are on very old hardware. Using standard implementation for speed.
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

class CausalSelfAttention(nn.Module):
    """
    Simplified Causal Attention.
    REMOVED: Key Padding Mask logic (since we use repetition padding).
    """
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        # batch_first=True expects [Batch, Seq, Feature]
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True 
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        
        # Create Causal Mask (Triangular)
        # 1s on diagonal and below, -inf above.
        # PyTorch MHA expects boolean mask where TRUE = IGNORE (Mask out)
        # So we want Upper Triangle (excluding diagonal) to be True.
        causal_mask = torch.triu(
            torch.ones(N, N, device=x.device, dtype=torch.bool), diagonal=1
        )

        out, weights = self.attn(
            query=x,
            key=x,
            value=x,
            attn_mask=causal_mask,
            need_weights=True,
            is_causal=True # Optimization hint for PyTorch 2.0+
        )
        return self.dropout(out), weights

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, dropout)
        self.norm2 = LayerNorm(d_model)
        self.mlp = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-Norm Architecture (More stable for Motion)
        x_norm = self.norm1(x)
        attn_out, weights = self.attn(x_norm)
        x = x + attn_out # Residual 1

        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = x + self.dropout(mlp_out) # Residual 2
        return x, weights

class TemporalTransfomerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_head, d_ff, dropout) for _ in range(num_layers)
        ])
        self.norm_final = LayerNorm(d_model)

    def forward(self, x, return_all_weights=False):
        all_weights = []
        for layer in self.layers:
            x, weights = layer(x)
            if return_all_weights:
                all_weights.append(weights)
        
        x = self.norm_final(x)
        
        if return_all_weights:
            return x, all_weights
        return x

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
        # Slice to current sequence length
        return x + self.pe[:, :x.size(1), :]

class MotionTransformer(nn.Module):
    """
    Split-Head Transformer for Motion Prediction.
    
    Optimized Head Structure:
    - Root Head: Predicts 9 values (3 Pos + 6 Rot) -> Pads to 12.
    - Body Head: Predicts 120 values (20 joints * 6 Rot).
    """
    def __init__(self, 
                 num_joints=27, 
                 input_feats=18, 
                 d_model=256, 
                 num_layers=4, 
                 n_head=4, 
                 d_ff=512, 
                 dropout=0.1, 
                 max_len=1024):
        super().__init__()
        
        self.num_joints = num_joints
        
        # 1. Input Projection
        # 27 * 18 = 486 features
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
        
        # Head A: Root Trajectory 
        # We only predict 9 REAL values:
        # - 3 for Translation Delta (dx, dy, dz)
        # - 6 for Rotation Delta (6D continuous)
        # We will manually pad the Translation to 6 dims later.
        self.head_root = nn.Linear(d_model, 9) 
        
        # Head B: Body Pose 
        # 20 joints * 6D rotation = 120 values
        self.head_body = nn.Linear(d_model, 120)

    def forward(self, x, key_padding_mask=None):
        # Extract dimensions
        B, S = x.shape[0], x.shape[1]

        # x shape: [Batch, Seq, 27, 18]
        if x.dim() == 4:
            x = x.view(B, S, -1) # Flatten -> [B, S, 486]

        # 1. Project & Encode
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.dropout_input(x)
        latent = self.encoder(x) # [B, S, 256]
        
        # --- 2. Head A: Root Processing (The fix) ---
        # Predict 9 values
        root_raw = self.head_root(latent) # [B, S, 9]
        
        # Split into Position (3) and Rotation (6)
        pred_pos_delta = root_raw[:, :, 0:3] # dx, dy, dz
        pred_rot_delta = root_raw[:, :, 3:9] # 6D rotation
        
        # Create Padding (0, 0, 0) for Position
        # Use zeros_like to match device/dtype of input automatically
        pad_zeros = torch.zeros_like(pred_pos_delta) 
        
        # Concatenate Position parts: [dx, dy, dz] + [0, 0, 0] -> [B, S, 6]
        root_pos_final = torch.cat([pred_pos_delta, pad_zeros], dim=2)
        
        # Stack Position and Rotation: 
        # [B, S, 6] (Pos) + [B, S, 6] (Rot) -> [B, S, 2, 6]
        root_out = torch.stack([root_pos_final, pred_rot_delta], dim=2)
        
        # --- 3. Head B: Body Processing ---
        # [B, S, 120] -> [B, S, 20, 6]
        body_out = self.head_body(latent).view(B, S, 20, 6)
        
        return root_out, body_out