# 1. The Data Pipeline (`data_loader.py` + `pose_processing.py`)

**Input:** Raw BlazePose **(F, 25, 7)**.

**Topology:**  
You correctly convert to **(F, 27, 7)** by adding **Neck** and **MidHip**.

**Features:**  
You extract **18 features per joint**.

**Edge Case 1 (Start of File):**  
`is_anchor` forces velocity to **0.0**. ✅ Correct.

**Edge Case 2 (Leaf Nodes):**  
`child_vecs` copies `parent_vecs` for endpoints (**fingers / toes / head**), preventing “dead” zero vectors. ✅ Correct.

**Edge Case 3 (Padding):**  
You use **Repetition Padding** (copying the first frame).  
This means the model always sees valid data, so **no padding mask is needed**. ✅ Correct.

**Final Output:**  
**(Batch, Seq, 27, 18)**


# 2. The Model (`model.py`)

**Input Dimension:**  
The model expects **27 × 18 = 486** input features.  
This matches your data loader exactly.

**Masking:**  
You use a **Causal Mask (Upper Triangular)** only.  
This allows the model to learn temporal dependencies without “cheating” by looking at future frames.  
Since padding is valid repeated data, this is the **correct approach**.

**Output Head:**  
The model predicts **9 values for the Root**:
- **3 position**
- **6 rotation**

Inside the forward pass, the position is manually padded to **(6)**.  
This matches your **Ground Truth format (22, 6)** perfectly without wasting model capacity on zeros.


# Final `blaze2cap/modules/models.py`

Here is the **complete, final** code for the model file.  
It defaults to **`num_joints = 27`** and includes the **9-value Root Head logic** discussed above.
