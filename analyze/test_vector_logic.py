import numpy as np
from blaze2cap.modules.pose_processing import PARENTS, CHILDREN, NUM_JOINTS

# Mock data: 2 Frames, 27 Joints, 3 Coords
F = 2
J = 27
world_centered = np.random.rand(F, J, 3)

# 1. Test indexing
parents_data = world_centered[:, PARENTS]
expected_p0 = world_centered[0, PARENTS[0]] # Parent of Node 0
actual_p0 = parents_data[0, 0]

# 2. Test Vector Calc
parent_vecs = world_centered - parents_data
# Vector for Node 5 (L_ELB). Parent is 3 (L_SH).
# Expected: Pos(L_ELB) - Pos(L_SH)
vec_5_manual = world_centered[:, 5] - world_centered[:, 3]
vec_5_auto = parent_vecs[:, 5]

# 3. Test Leaf Logic
child_vecs = world_centered[:, CHILDREN] - world_centered
# Leaf: L_WRIST -> L_INDEX (Wait, L_INDEX is leaf)
# L_INDEX = 11. Parent = 7 (L_WRIST). Child[11] = 11.
# Initial Child Vec for 11: Pos(11) - Pos(11) = 0.
# Logic replaces it with Parent Vec: Pos(11) - Pos(7).

leaf_mask = (CHILDREN == np.arange(NUM_JOINTS))
child_result = child_vecs.copy()
child_result[:, leaf_mask, :] = parent_vecs[:, leaf_mask, :]

is_leaf = leaf_mask[11] # Should be True
final_child_vec_11 = child_result[:, 11]
parent_vec_11 = parent_vecs[:, 11]

print(f"Indexing Match: {np.allclose(expected_p0, actual_p0)}")
print(f"Vector Calc Match: {np.allclose(vec_5_manual, vec_5_auto)}")
print(f"Node 11 is Leaf: {is_leaf}")
print(f"Leaf Extension Match: {np.allclose(final_child_vec_11, parent_vec_11)}")

# 4. Check Root
# Root is 26 (MidHip). Parent is 26.
# Parent Vec: 26 - 26 = 0.
vec_root = parent_vecs[:, 26]
print(f"Root Parent Vec is Zero: {np.allclose(vec_root, 0)}")
