# Pose Feature Specification (`pose_processing.py`)

This document defines the **19-dimensional feature vector** used per joint in the pose processing pipeline.

The features are intentionally split into **local body shape** and **global motion context**, allowing the model to **cleanly disentangle pose from trajectory** and avoid spatial drift or sliding artifacts (e.g. the *“Sharon problem”*).

---

## Overview

- **Total Features per Joint:** `19`
- **Total Joints per Frame:** `27`
- **Feature Groups:**
  - **Group A:** Canonical Pose (Local Body Shape)
  - **Group B:** View Context (Global Root Motion)

---

## Group A — Canonical Pose (Indices `0–13`)

These features describe **how the body is shaped**, independent of where the character exists in the world.

All values are computed in **canonical space**:
- Mid-Hip is positioned at `(0, 0, 0)`
- Hips are aligned to the **X-axis**
- World translation and rotation are removed

| Index Range | Feature Name | Dimensions | Description |
|------------|-------------|------------|-------------|
| `0 – 2` | **Canonical Position** | 3 | Joint position relative to the Mid-Hip origin in canonical space. |
| `3 – 5` | **Canonical Velocity** | 3 | Per-frame joint velocity computed in canonical space. |
| `6 – 8` | **Parent Vector** | 3 | Vector from this joint to its **parent** joint (e.g., Elbow → Shoulder). |
| `9 – 11` | **Child Vector** | 3 | Vector from this joint to its **child** joint (e.g., Elbow → Wrist). |
| `12` | **Visibility Score** | 1 | BlazePose confidence value (`0.0 – 1.0`). Used to suppress unreliable joints. |
| `13` | **Anchor Flag** | 1 | Binary indicator (`0 / 1`) denoting whether the joint is considered stable in this frame. |

### Purpose

Group A encodes **pure skeletal configuration**:
- Limb orientation
- Joint relationships
- Temporal consistency

> These features allow the model to understand *pose* without being affected by global movement or camera motion.

---

## Group B — View Context (Indices `14–18`)

These features encode **global motion and camera-relative context**.

> ⚠️ **Important:**  
> Group B features are **identical for all 27 joints within the same frame**.  
> They provide global trajectory information without contaminating local pose learning.

| Index Range | Feature Name | Dimensions | Description |
|------------|-------------|------------|-------------|
| `14 – 15` | **Alignment Angle** | 2 | World-space hip orientation encoded as `(sin θ, cos θ)`. Indicates facing direction. |
| `16 – 17` | **Screen Velocity** | 2 | Lateral motion of the hips in screen space. Encodes strafing movement. |
| `18` | **Scale Delta** | 1 | Change in hip width across frames. Approximates depth movement (forward/backward). |

### Purpose

Group B provides **trajectory awareness**:
- Facing direction
- Sideways movement
- Camera depth change

> This separation prevents pose distortion when the character moves through space.

---

## Final Feature Layout

```text
[ Canonical Pose (14) | View Context (5) ] = 19 Features
