Here is the breakdown of the exact mathematical transformations applied to the original data in the script to achieve the Z-Up, In-Place visualization.

The transformation used is a **Change of Basis** (Coordinate System Transformation) defined by the matrix .

$$M = \begin{bmatrix} 
1 & 0 & 0 \\ 
0 & 0 & -1 \\ 
0 & -1 & 0 
\end{bmatrix}$$
### The Transformation Matrix ()

This matrix converts your original **Y-Down** data to **Z-Up**.

---

### 1. Index 0: Root Position / Velocity

**Status:** **Calculated, then Discarded (Forced to 0,0,0)**

Although we force the position to zero for the "In-Place" view, the calculation for the velocity vector (before discarding) was:

1. **Inversion:** The raw velocity from TotalCapture data is usually inverted, so we flip it.
$$V_{temp} = V_{raw} \times -1.0$$

2. **Coordinate Transform:** We rotate the velocity vector into the new Z-Up world.

$$V_{final} = V_{temp} \cdot M^T$$
3. **In-Place Override:**
$$Position_{final} = (0, 0, 0)$$


---

### 2. Index 1: Root Rotation (The Hips)

**Status:** **Transformed (Change of Basis)**

This determines the "Delta" orientation (how much the character turns per frame).

1. **6D to Matrix:** Convert raw 6D data to a 3x3 Rotation Matrix ().
2. **Change of Basis:** We must transform the rotation matrix so it operates in the Z-Up world. This wraps the rotation in the transform matrix.

$$R_{final} = M \cdot R_{raw} \cdot M^T$$

*This  is what is displayed as "HIP DELTA ORIENTATION".*

---

### 3. Index 2-21: Body Joint Rotations

**Status:** **Transformed (Change of Basis)**

These are the local rotations for the spine, arms, legs, etc.

1. **6D to Matrix:** Convert raw 6D data to 3x3 Rotation Matrices ().
2. **Change of Basis:** Apply the same basis transformation as the root.

$$R_{final\_local} = M \cdot R_{raw\_local} \cdot M^T$$

**Why $M \cdot R \cdot M^T$?**
- $M^T$ converts a vector from the New (Z-Up) world back to the Old (Y-Down) world.
- $R$ applies the rotation in the Old world.
- $M$ converts the result back to the New (Z-Up) world.