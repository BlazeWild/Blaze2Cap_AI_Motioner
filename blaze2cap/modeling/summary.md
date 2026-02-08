| **Loss Term** | **What it Controls** | **Why we need it (The Failure Case)** |
|--------------|----------------------|--------------------------------------|
| **l_root_vel** | Hip Speed | Without this, the character **slides** (skating effect). It might move its legs fast but the hips don’t move forward, like Michael Jackson’s *Moonwalk*. |
| **l_root_rot** | Facing Direction | Without this, the character might walk **North while facing East** (incorrect strafing). |
| **l_pose_rot** | Joint Angles | Ensures correct bone structure. Without this, elbows may bend backward or knees may twist unnaturally. |
| **l_pose_pos** | 3D Position | MPJPE equivalent. Needed because small rotation errors at the hip cause huge position errors at the foot (**Lever Arm Effect**). |
| **l_smooth** | Velocity | **Anti-jitter.** Without this, the character vibrates. Two frames may both be “accurate” but slightly offset, making the video look like an earthquake. |
| **l_accel** | Acceleration | **Weight / Physics.** Without this, motion looks robotic and weightless. Real humans cannot stop instantly; momentum exists. |
