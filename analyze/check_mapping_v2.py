import numpy as np

# 1. Output Hierarchy (Model - 22 Joints)
output_names = [
    "Hips_pos", "Hips_rot", "Spine", "Spine1", "Spine2", "Spine3", "Neck", "Head",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightUpLeg", "RightLeg", "RightFoot",
    "LeftUpLeg", "LeftLeg", "LeftFoot"
]
# Provided in skeleton_config.py
output_parents = [-1, 0, 1, 2, 3, 4, 5, 6, 5, 8, 9, 10, 5, 12, 13, 14, 1, 16, 17, 1, 19, 20]

print("--- MODEL SKELETON (22 Joints) ---")
for i, (name, p) in enumerate(zip(output_names, output_parents)):
    p_name = output_names[p] if p != -1 else "None"
    print(f"IDX {i}: {name} (Parent {p}: {p_name})")

print("\n--- INPUT DATA HIERARCHY (25 Joints) ---")
# From data_loader.py (Deduced)
parents = np.arange(25)
# children logic implies parents
# CHILDREN[3], CHILDREN[4] = 5, 6
# PARENTS[5], PARENTS[6] = 3, 4
parents[5] = 3
parents[6] = 4

# CHILDREN[5], CHILDREN[6] = 7, 8
# PARENTS[7], PARENTS[8] = 5, 6
parents[7] = 5
parents[8] = 6

# CHILDREN[7], CHILDREN[8] = 11, 12
# PARENTS[[9, 11, 13]] = 7 
# PARENTS[[10, 12, 14]] = 8 
parents[[9, 11, 13]] = 7
parents[[10, 12, 14]] = 8

# Lower Body
# CHILDREN[15], CHILDREN[16] = 17, 18
# PARENTS[17], PARENTS[18] = 15, 16
parents[17] = 15
parents[18] = 16

# CHILDREN[17], CHILDREN[18] = 19, 20
# PARENTS[19], PARENTS[20] = 17, 18
parents[19] = 17
parents[20] = 18

# CHILDREN[19], CHILDREN[20] = 23, 24
# PARENTS[[21, 23]] = 19
# PARENTS[[22, 24]] = 20 
parents[[21, 23]] = 19
parents[[22, 24]] = 20

# Let's print the structure to infer names
# Assume 0, 1, 2 are root-like or static?
# 3, 4 start upper chains.
# 15, 16 start lower chains.

def get_chain(idx):
    chain = [str(idx)]
    # find children
    children = np.where(parents == idx)[0]
    children = children[children != idx] # Remove self-parenting (default)
    if len(children) > 0:
        child_strs = [get_chain(c) for c in children]
        chain.append(f"[{', '.join(child_strs)}]")
    return "".join(chain)

print("\nChains:")
# Base nodes (those that are their own parents)
roots = [i for i in range(25) if parents[i] == i]
for r in roots:
    print(f"Root {r}: {get_chain(r)}")
