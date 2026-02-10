
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt


# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, PROJECT_ROOT)

from blaze2cap.modules.models import MotionTransformer
from blaze2cap.modules.pose_processing import process_blazepose_frames
from blaze2cap.modeling.loss import MotionCorrectionLoss
from blaze2cap.utils.skeleton_config import get_totalcapture_skeleton

# --- Configuration ---
DATA_ROOT = os.path.join(PROJECT_ROOT, "blaze2cap/dataset/Totalcapture_blazepose_preprocessed/Dataset")
SOURCE_REL = "blazepose_final/S1/acting1/cam1/blazepose_S1_acting1_cam1_seg0_s1_o0.npy"
TARGET_REL = "gt_final/S1/acting1/cam1/gt_S1_acting1_cam1_seg0_s1_o0.npy"

SOURCE_PATH = os.path.join(DATA_ROOT, SOURCE_REL)
TARGET_PATH = os.path.join(DATA_ROOT, TARGET_REL)

WINDOW_SIZE = 64

print(f"=== STEP 1: LOADING DATA ===")
print(f"Source File: {SOURCE_PATH}")
print(f"Target File: {TARGET_PATH}")

if not os.path.exists(SOURCE_PATH):
    print(f"ERROR: Source file not found at {SOURCE_PATH}")
    sys.exit(1)
if not os.path.exists(TARGET_PATH):
    print(f"ERROR: Target file not found at {TARGET_PATH}")
    sys.exit(1)

# Load NPY
src_data = np.nan_to_num(np.load(SOURCE_PATH).astype(np.float32))
gt_data = np.nan_to_num(np.load(TARGET_PATH).astype(np.float32))

print(f"Loaded Source Shape: {src_data.shape}")  # Expected (Frames, 25, 7)
print(f"Loaded Target Shape: {gt_data.shape}")  # Expected (Frames, 132 or similar)

# Align lengths
min_len = min(len(src_data), len(gt_data))
src_data = src_data[:min_len]
gt_data = gt_data[:min_len]
print(f"Aligned Length: {min_len}")

print(f"\n=== STEP 2: PREPROCESSING (Creating Windows) ===")
# Process steps similar to PoseSequenceDataset
# 1. process_blazepose_frames converts raw (F, 25, 7) -> (F, Window, Features)
# It adds virtual joints, pads, extracts features, calculates deltas, etc.
X, M = process_blazepose_frames(src_data, WINDOW_SIZE)
print(f"Output X Shape (Windows): {X.shape}") # (Frames, WindowSize, Features=486)
print(f"Output M Shape (Masks): {M.shape}")

# Prepare Target
Y_flat = gt_data.reshape(min_len, -1)
pad_gt = np.repeat(Y_flat[0:1], WINDOW_SIZE-1, axis=0)
full_gt = np.concatenate([pad_gt, Y_flat], axis=0)
strides_gt = (full_gt.strides[0], full_gt.strides[0], full_gt.strides[1])
Y = np.lib.stride_tricks.as_strided(
    full_gt, shape=(min_len, WINDOW_SIZE, full_gt.shape[1]), strides=strides_gt
)
print(f"Output Y Shape (Windows): {Y.shape}")

# Select a single window for demonstration
SAMPLE_IDX = 100 # Arbitrary frame index
if SAMPLE_IDX >= len(X): SAMPLE_IDX = 0

sample_x = torch.from_numpy(X[SAMPLE_IDX:SAMPLE_IDX+1]) # [1, Window, Features]
sample_y = torch.from_numpy(Y[SAMPLE_IDX:SAMPLE_IDX+1]) # [1, Window, TargetFeats]
sample_m = torch.from_numpy(M[SAMPLE_IDX:SAMPLE_IDX+1]) # [1, Window]

print(f"\nProcessing Window Index: {SAMPLE_IDX}")
print(f"Sample Input Shape: {sample_x.shape}")

print(f"\n=== STEP 3: MODEL INITIALIZATION (Random Weights) ===")
model = MotionTransformer(
    num_joints=27,
    input_feats=18,
    d_model=256,
    num_layers=4,
    n_head=4,
    d_ff=512,
    dropout=0.0
)
print("Model initialized with random weights.")

# Monkey-patch the encoder to return weights to us
# The MotionTransformer uses self.encoder (TemporalTransfomerEncoder)
# TemporalTransfomerEncoder.forward has a `return_all_weights` arg.
# But MotionTransformer.forward doesn't expose it.
# We will manually call the sub-modules to show step-by-step.

print(f"\n=== STEP 4: FORWARD PASS (Step-by-Step) ===")

# 4.1 Input Projection
print("--- 4.1 Input Projection ---")
B, S, _ = sample_x.shape
x_flat = sample_x.view(B, S, -1)
print(f"Flattened Input: {x_flat.shape}")
x_proj = model.input_projection(x_flat)
print(f"Projected Input: {x_proj.shape}")

# 4.2 Positional Encoding
print("--- 4.2 Positional Encoding ---")
x_pos = model.pos_encoder(x_proj)
x_drop = model.dropout_input(x_pos)
print(f"After PosEnc & Dropout: {x_drop.shape}")

# 4.3 Transformer Encoder (With Attention Visualization)
print("--- 4.3 Transformer Encoder Layers ---")
# Access the encoder instance
encoder = model.encoder
latent = x_drop
attn_weights_list = []

for i, layer in enumerate(encoder.layers):
    print(f"  Processing Layer {i}...")
    # layer(x) returns (x, weights)
    latent, weights = layer(latent)
    attn_weights_list.append(weights)
    print(f"    Layer {i} Output: {latent.shape}")
    print(f"    Layer {i} Attn Weights: {weights.shape}")

latent = encoder.norm_final(latent)
print(f"Final Latent Representation: {latent.shape}")

print(f"\n=== STEP 5: CAUSAL MASK VIEWING ===")
# The weights returned by CausalSelfAttention already have the mask applied (softmaxed).
# But let's visualize the raw mask structure that caused this.
# The code in models.py does:
#   causal_mask = torch.triu(torch.ones(N, N), diagonal=1)
# Let's show the first head's attention map for the last layer.
# last_layer_attn is [Batch, Seq, Seq] -> [1, 64, 64]
# We take the first batch item: [64, 64]
attn_matrix = attn_weights_list[-1][0].detach().numpy()

print("Visualizing Attention Map for Layer 3 (Averaged Heads) (Check 'attn_map.png')")
plt.figure(figsize=(10, 8))
plt.imshow(attn_matrix, cmap="viridis", aspect='auto')
plt.colorbar()
plt.title(f"Causal Attention Map (Layer 3, Head 0)\nLower triangle populated, Upper triangle zeroed")
plt.xlabel("Key Position")
plt.ylabel("Query Position")
plt.savefig("attn_map.png")
print("Saved attn_map.png")

print(f"Verifying Causality:")
print(f"Top-Right Corner (Should be ~0.0): {attn_matrix[0, -1]}")
print(f"Bottom-Left Corner (Should be >0): {attn_matrix[-1, 0]}")

print(f"\n=== STEP 6: OUTPUT HEADS ===")
# 6.1 Root Head
root_raw = model.head_root(latent)
pred_pos_delta = root_raw[:, :, 0:3]
pred_rot_delta = root_raw[:, :, 3:9]
pad_zeros = torch.zeros_like(pred_pos_delta)
root_pos_final = torch.cat([pred_pos_delta, pad_zeros], dim=2) # 3 -> 6
root_out = torch.stack([root_pos_final, pred_rot_delta], dim=2) # [B, S, 2, 6]
print(f"Root Output: {root_out.shape}")

# 6.2 Body Head
body_out = model.head_body(latent).view(B, S, 20, 6)
print(f"Body Output: {body_out.shape}")

# Combine
pred_combined = torch.cat([root_out, body_out], dim=2)
print(f"Final Prediction Shape: {pred_combined.shape}")

print(f"\n=== STEP 7: CALCULATING LOSS ===")
# Setup Loss
skel_config = get_totalcapture_skeleton()
parents = torch.tensor(skel_config['parents'], dtype=torch.long)
offsets = skel_config['offsets']

criterion = MotionCorrectionLoss(
    parents=parents,
    offsets=offsets,
    lambda_root_vel=50.0,
    lambda_root_rot=10.0,
    lambda_pose_rot=2.0,
    lambda_pose_pos=10.0,
    lambda_smooth=0.0,
    lambda_accel=0.0
)

# Reshape target if necessary
# Target is usually flattened in dataset, but here we kept dimensions in `sample_y`?
# sample_y shape: [1, 64, 132]
# We need to reshape it to [1, 64, 22, 6] to match prediction
target_reshaped = sample_y.view(B, S, 22, 6)
print(f"Target Reshaped: {target_reshaped.shape}")

loss_dict = criterion((root_out, body_out), target_reshaped, sample_m)

print("\n--- Loss Components ---")
total_loss = loss_dict["loss"].item()
print(f"Total Loss: {total_loss:.6f}")
for k, v in loss_dict.items():
    if k != "loss":
        print(f"  {k}: {v.item():.6f}")

print("\n=== STEP 8: CALCULATING WEIGHT UPDATES (Simulation) ===")
# Show a gradient calculation
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
optimizer.zero_grad()
loss_dict["loss"].backward()

print("Gradients computed.")
# Check gradient of the first layer weight
first_layer_grad = model.input_projection.weight.grad
print(f"Input Projection Gradient Norm: {first_layer_grad.norm().item():.6f}")

optimizer.step()
print("Optimizer step performed. Weights updated.")

print("\n=== DONE ===")
