# Blaze2Cap

**Real-Time 3D Motion Prediction from 2D/3D Pose Landmarks**

Transform BlazePose keypoints into smooth, accurate 3D skeletal motion using deep learning.

---

## 🎯 Overview

Blaze2Cap is a PyTorch-based motion prediction system that converts 2D/3D pose landmarks (from MediaPipe BlazePose) into full 3D skeletal motion with 22 joints. The model uses a Transformer architecture optimized for temporal consistency and motion smoothing.

**Key Features:**
- ✅ **Temporal Transformer** - Causal self-attention for sequential motion prediction
- ✅ **Motion Smoothing** - High smoothness loss weight for natural, jitter-free output
- ✅ **Mixed Precision Training** - FP16 support for faster training on modern GPUs
- ✅ **L4 GPU Optimized** - Configured for NVIDIA L4 (24GB VRAM)
- ✅ **Comprehensive Testing** - Full test suite for models, data loaders, and loss functions

---

## 📊 Dataset

**TotalCapture Dataset** (Augmented)
- **Input:** BlazePose landmarks (25 keypoints × 7 channels → 18 features)
- **Output:** 3D skeletal motion (22 joints × 6D rotation representation)
- **Samples:** 5,164 total (2,775 train / 2,068 test)
- **Augmentation:** Temporal subsampling at multiple strides (60fps, 30fps, 20fps)

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/BlazeWild/Blaze2Cap.git
cd Blaze2Cap

# Install dependencies
pip install -e .

# Or install manually
pip install torch torchvision torchaudio mediapipe==0.10.14 numpy tqdm
```

### Dataset Setup

**Download from Hugging Face:**

The dataset (6.89GB) is hosted on Hugging Face for easier access:

```bash
# Install huggingface-hub
pip install huggingface-hub

# Download the dataset
huggingface-cli download Blazewild/Totalcap-blazepose --repo-type dataset --local-dir blaze2cap/dataset/Totalcapture_blazepose_preprocessed/Dataset
```

**Expected structure:**

```
blaze2cap/dataset/Totalcapture_blazepose_preprocessed/Dataset/
├── blaze_augmented/     # Input: BlazePose keypoints (5,164 samples)
├── gt_augmented/        # Output: Ground truth motion (5,164 samples)
└── dataset_map.json     # Train/test split mapping
```

**Generate dataset mapping** (if not included in download):

```bash
python blaze2cap/data/generate_json.py
```

### Training

```bash
# Run tests first
python -m test.test_model
python -m test.test_loss

# Start training
python -m tools.train
```

### Evaluation

```bash
# Evaluate best model
python -m test.evaluate --checkpoint ./checkpoints/best_model.pth

# Evaluate on specific split
python -m test.evaluate --checkpoint ./checkpoints/best_model.pth --split test
```

---

## 🏗️ Architecture

### Model: MotionTransformer

```
Input: [Batch, Seq, 25, 18]  # 25 joints × 18 features
  ↓
Flatten & Project: [B, S, 256]
  ↓
Positional Encoding
  ↓
Transformer Encoder (4 layers)
  - Causal Self-Attention
  - Feed-Forward Network
  ↓
Split-Head Decoder:
  - Root Head: [B, S, 2, 6]   # Position + Rotation deltas
  - Body Head: [B, S, 20, 6]  # 20 joint local rotations
  ↓
Output: [B, S, 22, 6]  # Combined 3D skeletal motion
```

**Parameters:** ~2.5M trainable parameters

### Loss Function

```python
L_total = λ_rot × L_rotation + λ_smooth × L_smoothness

# L_rotation: MSE between predicted and GT 6D rotations
# L_smoothness: MSE of velocity differences (penalizes jitter)
```

**L4 GPU Configuration:**
- `λ_rot = 1.0` - Keep geometry grounded
- `λ_smooth = 5.0` - **High weight for smooth motion**

---

## ⚙️ Configuration

### Hyperparameters (L4 GPU Optimized)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `batch_size` | 512 | Maximize L4 throughput |
| `num_workers` | 8 | Parallel data loading |
| `window_size` | 64 | Larger temporal context |
| `d_model` | 256 | Transformer hidden size |
| `num_layers` | 4 | Transformer depth |
| `n_head` | 4 | Multi-head attention |
| `lr` | 1e-4 | Learning rate |
| `epochs` | 100 | Training epochs |
| `use_amp` | True | Mixed precision (FP16) |

Edit `tools/train.py` to modify these settings.

---

## 📁 Project Structure

```
Blaze2Cap/
├── blaze2cap/
│   ├── __init__.py              # Package exports
│   ├── data/
│   │   ├── data_loader.py       # PoseSequenceDataset
│   │   └── generate_json.py     # Dataset map generator
│   ├── modules/
│   │   └── models.py            # MotionTransformer
│   ├── modeling/
│   │   ├── loss.py              # MotionCorrectionLoss
│   │   ├── eval_motion.py       # MPJPE/MARE metrics
│   │   └── optimization.py      # Optimizer configs
│   ├── utils/
│   │   ├── checkpoint.py        # Save/load checkpoints
│   │   ├── train_utils.py       # Timer, CudaPreFetcher
│   │   ├── logging.py           # Setup logging
│   │   └── visualization.py     # Render pose videos
│   └── dataset/
│       └── Totalcapture_blazepose_preprocessed/
│           └── Dataset/         # Training data
├── tools/
│   └── train.py                 # Main training script
├── test/
│   ├── test_model.py            # Model architecture tests
│   ├── test_loss.py             # Loss function tests
│   ├── test_dataloader.py       # Data loader tests
│   ├── evaluate.py              # Evaluation script
│   └── run_all.py               # Run all tests
├── pyproject.toml               # Project metadata
└── README.md                    # This file
```

---

## 🧪 Testing

```bash
# Test model architecture
python -m test.test_model

# Test loss functions
python -m test.test_loss

# Test data loader
python -m test.test_dataloader

# Run all tests
python -m test.run_all
```

---

## 📈 Metrics

The model is evaluated using:
- **MPJPE** (Mean Per Joint Position Error) - 3D position accuracy in mm
- **MARE** (Mean Absolute Rotation Error) - Rotation accuracy in radians

---

## 🔧 Troubleshooting

### CUDA Out of Memory

Reduce batch size in `tools/train.py`:
```python
"batch_size": 256,  # Reduce from 512
```

### Import Errors

Make sure to activate your virtual environment:
```bash
source venv/bin/activate  # Linux/Mac
# or
conda activate your_env
```

### Dataset Not Found

Verify dataset path and regenerate mapping:
```bash
ls -la blaze2cap/dataset/Totalcapture_blazepose_preprocessed/Dataset/
python blaze2cap/data/generate_json.py
```

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{blaze2cap2026,
  author = {BlazeWild},
  title = {Blaze2Cap: Real-Time 3D Motion Prediction from Pose Landmarks},
  year = {2026},
  url = {https://github.com/BlazeWild/Blaze2Cap}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **TotalCapture Dataset** - For providing high-quality motion capture data
- **MediaPipe BlazePose** - For real-time pose estimation
- **PyTorch Team** - For the deep learning framework

---

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the maintainer.

**Repository:** https://github.com/BlazeWild/Blaze2Cap

---

**Built with ❤️ for smooth, natural motion prediction**
