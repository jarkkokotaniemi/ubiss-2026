# people_avoidance — teaching skeleton

A ROS 2 Python package for a LiDAR-based people-avoidance pipeline.
The node wires four stages together and runs end-to-end from day one.
**Students fill in the four TODO stages** — the node publishes a safe zero
command until each stage is implemented.

---

## Pipeline overview

```
/scan (LaserScan)
    │
    ▼
[Stage 1] leg_detection.py  ──  segment_scan() + detect_legs()
    │  List[LegMeasurement]  { x, y, Rxx, Rxy, Ryy }
    ▼
[Stage 2] tracking.py  ──  KalmanTracker.predict() / associate() / update()
    │  List[Track]  each Track: m=[x,y,vx,vy] (4,), P=(4×4)
    ▼
[Stage 3] controller.py  ──  obstacle_radius() + compute_velocity()
    │  geometry_msgs/Twist
    ▼
/cmd_vel
```

Robot pose (x, y, θ) is read from `/odom` and fed into the controller.

---

## Data contract between stages

### Stage 1 output — `LegMeasurement`

| Field | Type  | Meaning |
|-------|-------|---------|
| `x`   | float | Person position x in the laser frame (m) |
| `y`   | float | Person position y in the laser frame (m) |
| `Rxx` | float | Measurement covariance R[0,0] (m²) |
| `Rxy` | float | Measurement covariance R[0,1] = R[1,0] (m²) |
| `Ryy` | float | Measurement covariance R[1,1] (m²) |

R is the 2×2 symmetric observation noise matrix passed to the Kalman update step.

### Stage 2 output — `Track`

| Field      | Type          | Meaning |
|------------|---------------|---------|
| `m`        | `np.ndarray (4,)` | State mean [x, y, vx, vy] in the odom frame |
| `P`        | `np.ndarray (4,4)` | State covariance |
| `track_id` | int           | Unique track identifier |
| `misses`   | int           | Consecutive unmatched frames |

Observation model: H = [[1,0,0,0],[0,1,0,0]] — only position is observed.

### Stage 3 output — `geometry_msgs/Twist`

| Field            | Meaning |
|------------------|---------|
| `linear.x`       | Forward velocity (m/s) |
| `angular.z`      | Rotation rate (rad/s, positive = left) |

---

## The four TODO stages

### Stage 1 — `leg_detection.py`

**`segment_scan(points, distance_threshold)`**
Split a sorted (N,2) point cloud into contiguous clusters.
Two consecutive points belong to the same cluster when their distance < threshold.

**`detect_legs(scan, ...)`**
From the segmented clusters, identify leg-like blobs, pair them hip-width apart,
and return one `LegMeasurement` per person.
Also assign the observation covariance R (Rxx, Rxy, Ryy) from range uncertainty.

### Stage 2 — `tracking.py`

**`KalmanTracker.__init__()`**
Define the constant-velocity state transition matrix F and process noise Q.

**`KalmanTracker.predict()`**
Propagate all tracks: `m = F @ m`, `P = F @ P @ F.T + Q`.

**`KalmanTracker.associate(measurements)`**
Build a cost matrix (Euclidean or Mahalanobis distance) between tracks and
measurements. Solve with `scipy.optimize.linear_sum_assignment`. Gate large costs.

**`KalmanTracker.update(measurements)`**
Apply the KF update equations for each matched pair, spawn new tracks for
unmatched measurements, prune tracks that exceed `max_misses`.

### Stage 3 — `controller.py`

**`obstacle_radius(track, sigma_scale)`**
Return `sigma_scale × √(λ_max)` where λ_max is the largest eigenvalue of
`track.P[:2, :2]`.  This inflates the safety bubble with positional uncertainty.

**`compute_velocity(tracks, robot_x, robot_y, robot_theta, ...)`**
Implement an avoidance policy (potential fields, VFH, DWA, or a simple
reactive rule). Return a `Twist` with `linear.x` and `angular.z` clipped to the
declared speed limits.

---

## Parameters

All parameters are declared in the node with defaults and can be overridden at
launch — no code changes needed.

| Parameter              | Default | Description |
|------------------------|---------|-------------|
| `scan_topic`           | `/scan` | LiDAR input |
| `cmd_vel_topic`        | `/cmd_vel` | Velocity output |
| `odom_topic`           | `/odom` | Robot pose source |
| `dt`                   | 0.1 s  | KF time step |
| `max_misses`           | 5      | Track deletion threshold |
| `distance_threshold`   | 0.1 m  | Segmentation gap |
| `leg_radius`           | 0.10 m | Expected single-leg radius |
| `max_leg_width`        | 0.25 m | Max leg-pair separation |
| `max_linear_speed`     | 0.2 m/s | Forward speed cap |
| `max_angular_speed`    | 1.0 rad/s | Rotation rate cap |
| `obstacle_radius_scale`| 2.0    | Uncertainty inflation factor k |

---

## How to run against the TurtleBot4 simulation

### Terminal 1 — start the simulator

```zsh
conda deactivate
source /opt/ros/jazzy/setup.zsh
export TURTLEBOT4_MODEL=standard      # or: lite
ros2 launch turtlebot4_gz_bringup turtlebot4.launch.py
```

Wait until Gazebo opens and the robot is visible (~20–40 s on first launch).
You should see the TB4 standard robot (tall cylindrical body) in the default world.

### Terminal 2 — build and run the skeleton node

```zsh
conda deactivate
source /opt/ros/jazzy/setup.zsh
cd ~/ros2_ws
colcon build --packages-select people_avoidance --symlink-install
source install/setup.zsh
ros2 launch people_avoidance people_avoidance.launch.py
```

Expected output (with stubs, no implementation):
```
[people_avoidance] PeopleAvoidanceNode ready — listening on '/scan', publishing to '/cmd_vel'
```

### Terminal 3 — verify topics

```zsh
source /opt/ros/jazzy/setup.zsh
ros2 topic echo --once /cmd_vel     # should show zero Twist
ros2 topic hz /cmd_vel              # should match /scan rate (~5 Hz)
ros2 topic echo --once /scan        # should show RPLidar A1 ranges
```

### Overriding a parameter at launch

```zsh
ros2 launch people_avoidance people_avoidance.launch.py \
    max_linear_speed:=0.3 obstacle_radius_scale:=3.0
```
