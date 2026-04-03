# Autonomous Navigation via Artificial Potential Fields

ROS 2 (Jazzy) navigation stack for a holonomic Mecanum-wheeled robot using Artificial Potential Fields (APF).

## Simulation

![Simulation](media/Screenshot%202026-04-03%20191926.png)

## How to Run

```bash
colcon build --packages-select robot
source install/setup.bash
ros2 launch robot sim_gazebo.launch.py
```

Set a navigation goal using the **2D Goal Pose** tool in RViz.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k_att` | 1.0 | Attractive gain |
| `k_rep` | 0.1 | Repulsive gain |
| `d0` | 1.0 | Obstacle influence distance (m) |
| `max_vel` | 0.5 | Maximum velocity (m/s) |

## Dependencies

- ROS 2 Jazzy
- `ros-jazzy-ros-gz-*` (bridge packages)
- `rclpy`, `sensor_msgs`, `geometry_msgs`, `nav_msgs`
