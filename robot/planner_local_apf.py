import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from geometry_msgs.msg import Twist, PoseStamped, Point
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import math
import numpy as np

class APFPlannerNode(Node):
    def __init__(self):
        super().__init__('planner_local_apf')
        
        # Declare Parameters for Live Tuning
        self.declare_parameter('k_att', 1.2, ParameterDescriptor(description='Attractive force gain'))
        self.declare_parameter('max_vel', 0.8)
        self.declare_parameter('safety_bubble', 0.35)
        self.declare_parameter('slow_zone', 0.8)
        self.declare_parameter('k_rep', 0.08, ParameterDescriptor(description='Repulsive force gain'))
        self.declare_parameter('k_curl', 0.1, ParameterDescriptor(description='Tangential/Vortex force gain'))
        self.declare_parameter('d0', 1.0, ParameterDescriptor(description='Influence range of obstacles (meters)'))
        self.declare_parameter('v_max', 0.7, ParameterDescriptor(description='Max linear velocity'))
        self.declare_parameter('w_max', 6.0, ParameterDescriptor(description='Max angular velocity (TURBO)'))
        self.declare_parameter('nav_mode', 0, ParameterDescriptor(description='0: Standard APF, 1: Harmonic (Trap-Free)'))
        self.declare_parameter('cluster_dist', 0.25, ParameterDescriptor(description='Threshold for jump-clustering LIDAR points'))

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/force_markers', 10)
        self.pub_nav_markers = self.create_publisher(MarkerArray, '/navigation_markers', 10)
        self.pub_obs_markers = self.create_publisher(MarkerArray, '/detected_obstacles', 10)
        
        # Subscribers
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.create_subscription(Path, '/plan', self.path_callback, 10)
        
        # State
        self.current_pose = None
        self.goal_pose = None
        self.waypoints = []
        self.latest_scan = None
        
        # 3. Visualization State
        self.start_pose = None
        self.actual_path = Path()
        self.actual_path.header.frame_id = "odom"
        self.trail_pub = self.create_publisher(Path, '/robot_trail', 10)
        
        self.timer = self.create_timer(0.05, self.control_loop) # Faster 20Hz loop
        self.get_logger().info("APF Local Planner (Mecanum Artificial Potential Field) Started.")

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose
        if self.start_pose is None:
            self.start_pose = msg.pose.pose
        if not hasattr(self, '_odom_count'): self._odom_count = 0
        self._odom_count += 1
        if self._odom_count % 100 == 0:
            self.get_logger().info("Odometry Heartbeat [OK]")

    def goal_callback(self, msg):
        # We still accept direct goals, but A* /plan will override them
        self.goal_pose = msg.pose
        self.waypoints = []
        self.get_logger().info(f"Direct APF Goal: {msg.pose.position.x}, {msg.pose.position.y}")

    def path_callback(self, msg):
        self.waypoints = msg.poses
        self.goal_pose = None
        self.get_logger().info(f"Received A* Path with {len(self.waypoints)} waypoints.")

    def scan_callback(self, msg):
        self.latest_scan = msg

    def create_force_marker(self, id, fx, fy, color, label, scale=0.5):
        marker = Marker()
        marker.header.frame_id = "base_footprint"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "forces"
        marker.id = id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        start = Point(x=0.0, y=0.0, z=0.1)
        end = Point(x=fx*scale, y=fy*scale, z=0.1)
        marker.points = [start, end]
        marker.scale.x = 0.05
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color = color
        return marker

    def create_safety_ring(self, radius):
        marker = Marker()
        marker.header.frame_id = "base_footprint"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "safety_zone"
        marker.id = 10
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position.z = 0.01
        marker.scale.x = radius * 2.0
        marker.scale.y = radius * 2.0
        marker.scale.z = 0.01
        marker.color = ColorRGBA(r=0.0, g=0.5, b=1.0, a=0.2) # Transparent Blue
        return marker

    def cluster_obstacles(self, msg):
        """Perform Jump-Distance clustering on raw LIDAR scan."""
        if not msg: return []
        
        ranges = np.array(msg.ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))
        cluster_dist = self.get_parameter('cluster_dist').value
        
        clusters = []
        current_cluster = []
        
        for i in range(len(ranges)):
            r = ranges[i]
            if not (0.1 < r < 5.0): # Skip noise
                if current_cluster:
                    clusters.append(current_cluster)
                    current_cluster = []
                continue
            
            p_x = r * math.cos(angles[i])
            p_y = r * math.sin(angles[i])
            point = np.array([p_x, p_y])
            
            if not current_cluster:
                current_cluster.append(point)
            else:
                last_point = current_cluster[-1]
                dist = np.linalg.norm(point - last_point)
                if dist < cluster_dist:
                    current_cluster.append(point)
                else:
                    clusters.append(current_cluster)
                    current_cluster = [point]
        
        if current_cluster: clusters.append(current_cluster)
        
        # 2. Geometric Feature Extraction
        obstacles = []
        for c in clusters:
            points = np.array(c)
            centroid = np.mean(points, axis=0) # [X, Y] in Body Frame
            # Radius is max distance from centroid to any point in cluster
            radius = np.max(np.linalg.norm(points - centroid, axis=1)) 
            obstacles.append({
                'center': centroid,
                'radius': max(0.05, radius) # Minimum size for point obstacles
            })
        
        return obstacles

    def control_loop(self):
        if self.current_pose is None:
            return
            
        # [IMMEDIATE VISUALS] Publish even if no goal exists
        self.publish_visuals()
            
        target_x, target_y = None, None
            
        if len(self.waypoints) > 0:
            target_x = self.waypoints[0].pose.position.x
            target_y = self.waypoints[0].pose.position.y
        elif self.goal_pose is not None:
            target_x = self.goal_pose.position.x
            target_y = self.goal_pose.position.y
        else:
            return

        # Fetch Parameters (Live)
        k_att = self.get_parameter('k_att').value
        k_rep = self.get_parameter('k_rep').value
        k_curl = self.get_parameter('k_curl').value
        d0 = self.get_parameter('d0').value
        v_max = self.get_parameter('v_max').value
        w_max = self.get_parameter('w_max').value

        # 1. Orientation
        q = self.current_pose.orientation
        yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y**2 + q.z**2))

        # 2. Attraction
        dx_world = target_x - self.current_pose.position.x
        dy_world = target_y - self.current_pose.position.y
        dist_to_goal = math.sqrt(dx_world**2 + dy_world**2)

        # --- DIAGNOSTIC DASHBOARD ---
        if not hasattr(self, '_log_tick'): self._log_tick = 0
        self._log_tick += 1
        if self._log_tick % 20 == 0: # 1 update per second
            yaw_deg = math.degrees(yaw)
            type_str = "WAYPOINT" if len(self.waypoints) > 0 else "DIRECT GOAL"
            
            # Format the remaining path for readability
            if len(self.waypoints) > 0:
                path_str = " -> ".join([f"({p.pose.position.x:.1f},{p.pose.position.y:.1f})" for p in self.waypoints[:5]])
                if len(self.waypoints) > 5: path_str += " ..."
            else:
                path_str = "Final Approach" if dist_to_goal > 0.3 else "Goal Reached"
            
            self.get_logger().info("\n" + "="*50 + 
                                   f"\n[ROBOT STATE]  Pos: ({self.current_pose.position.x:.2f}, {self.current_pose.position.y:.2f}) | Rot: {yaw_deg:.1f}°" +
                                   f"\n[NAVIGATION]   Target: {type_str} at ({target_x:.2f}, {target_y:.2f})" +
                                   f"\n[DIST TO GO]   {dist_to_goal:.2f} meters" +
                                   f"\n[PLANNED PATH] {path_str}" +
                                   "\n" + "="*50)

        if dist_to_goal < 0.25:
            if len(self.waypoints) > 0:
                self.waypoints.pop(0)
                if len(self.waypoints) == 0:
                    self.get_logger().info("Final A* Waypoint Reached!")
                    self.stop_robot()
            else:
                self.get_logger().info("Direct Goal Reached!")
                self.stop_robot()
                self.goal_pose = None
            return

        # 2. Attractive Force (World Frame)
        safe_dist = max(0.1, dist_to_goal)
        f_att_world_x = k_att * (dx_world / safe_dist)
        f_att_world_y = k_att * (dy_world / safe_dist)
        
        # Transform World Attraction to Body Frame (Single Rotation)
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        f_att_body_x = f_att_world_x * cos_y + f_att_world_y * sin_y
        f_att_body_y = -f_att_world_x * sin_y + f_att_world_y * cos_y

        # 3. Dynamic d0 (Adaptive Safety Radius - "Squeeze Mode")
        d0_raw = self.get_parameter('d0').value
        robot_radius = 0.26 # Physical shell radius
        # Allow passing through gaps by shrinking d0 slowly to just above robot size
        min_allowed_d0 = robot_radius + 0.02 
        
        f_gap_x, f_gap_y = 0.0, 0.0

        if self.latest_scan is not None:
            # 3a. GAP SENSING (Passage Detection)
            ranges = np.nan_to_num(np.array(self.latest_scan.ranges), nan=0.0, posinf=5.0, neginf=0.0)
            angles = np.linspace(self.latest_scan.angle_min, self.latest_scan.angle_max, len(ranges))
            
            jumps = np.where(np.abs(np.diff(ranges)) > 0.5)[0]
            if len(jumps) >= 2:
                best_gap_angle = None
                smallest_angle_diff = float('inf')
                target_yaw_rel = math.atan2(f_att_body_y, f_att_body_x) if (f_att_body_x != 0 or f_att_body_y != 0) else 0.0
                
                for i in range(len(jumps)-1):
                    idx1, idx2 = jumps[i], jumps[i+1]
                    r1, r2 = ranges[idx1], ranges[idx2]
                    a1, a2 = angles[idx1], angles[idx2]
                    gap_width = abs(a2 - a1) * ((r1 + r2) / 2.0)
                    
                    if gap_width > 0.45: # Tighter fit allowed!
                        mid_angle = (a1 + a2) / 2.0
                        angle_diff = abs(mid_angle - target_yaw_rel)
                        if angle_diff < smallest_angle_diff:
                            smallest_angle_diff = angle_diff
                            best_gap_angle = mid_angle
                            
                if best_gap_angle is not None:
                    f_gap_x += 1.5 * math.cos(best_gap_angle) # Stronger pull through gaps
                    f_gap_y += 1.5 * math.sin(best_gap_angle)

            valid_ranges = [r for r in self.latest_scan.ranges if 0.1 < r < 5.0]
            if valid_ranges:
                min_r = min(valid_ranges)
                # Slowly descend toward min_allowed_d0 if blocked
                target_d0 = min(d0_raw, max(min_allowed_d0, min_r * 0.95))
                if not hasattr(self, 'current_d0'): self.current_d0 = d0_raw
                self.current_d0 = 0.98 * self.current_d0 + 0.02 * target_d0 # Slow smooth decay
            else:
                self.current_d0 = d0_raw
        else:
            self.current_d0 = d0_raw

        # 4. Geometric Repulsion (Cluster-Based)
        f_rep_x, f_rep_y = 0.0, 0.0
        self.detected_obstacles = self.cluster_obstacles(self.latest_scan)
        
        for obs in self.detected_obstacles:
            cx, cy = obs['center'][0], obs['center'][1]
            rad = obs['radius']
            dist_to_center = math.sqrt(cx**2 + cy**2)
            
            # Distance from robot edge to obstacle "surface"
            r_surface = max(0.02, dist_to_center - (rad + robot_radius))
            d0_limit = self.current_d0
            
            if r_surface < d0_limit:
                # Geometric Repulsion Magnitude
                rep_mag = k_rep * (1.0/r_surface - 1.0/d0_limit) * (1.0/r_surface**2)
                rep_mag = min(rep_mag, 15.0) # slightly higher cap for clusters
                
                angle_to_obs = math.atan2(cy, cx)
                f_rep_x -= rep_mag * math.cos(angle_to_obs)
                f_rep_y -= rep_mag * math.sin(angle_to_obs)
                
                # Dynamic "Revolving" Vortex Force
                side = 1.0 if cy > 0 else -1.0
                vortex_gain = k_curl * (2.0 / (r_surface + 0.1))
                f_rep_x += vortex_gain * rep_mag * math.sin(angle_to_obs) * side
                f_rep_y -= vortex_gain * rep_mag * math.cos(angle_to_obs) * side

        # 5. Total Force Integration (Body Frame Mixing)
        # f_rep and f_gap are natively in Body Frame (LiDAR relative)
        
        # f_rep and f_gap are natively in Body Frame (LiDAR relative)
        # Combine all for final command
        f_raw_x = f_att_body_x + f_rep_x + f_gap_x
        f_raw_y = f_att_body_y + f_rep_y + f_gap_y
        
        mag = math.sqrt(f_raw_x**2 + f_raw_y**2)
        if mag > 2.0:
            f_raw_x = (f_raw_x / mag) * 2.0
            f_raw_y = (f_raw_y / mag) * 2.0
            
        if not hasattr(self, 'prev_f_x'):
            self.prev_f_x, self.prev_f_y = f_raw_x, f_raw_y
            
        # High Damping (0.95) to stop oscillations
        f_total_x = 0.95 * self.prev_f_x + 0.05 * f_raw_x
        f_total_y = 0.95 * self.prev_f_y + 0.05 * f_raw_y
        self.prev_f_x, self.prev_f_y = f_total_x, f_total_y
        
        # 6. Holonomic Velocity Mapping (Body Frame)
        # Reverted Sign Flip - Returning to standard ROS convention
        arrival_scaling = min(1.0, dist_to_goal / 0.5)
        vel_x_target = max(min(f_total_x * arrival_scaling, v_max), -v_max)
        vel_y_target = max(min(f_total_y * arrival_scaling, v_max), -v_max)
        
        # [TELEMETRY DIAGNOSTICS] - Every 1.0s
        now = self.get_clock().now()
        if not hasattr(self, '_last_log_t'):
            self._last_log_t = now
        
        if (now - self._last_log_t).nanoseconds > 1e9:
            self._last_log_t = now
            self.get_logger().info(f"\n[CONTROL] World_F: ({f_att_world_x:.2f}, {f_att_world_y:.2f}) | Body_F: ({f_total_x:.2f}, {f_total_y:.2f})")
            self.get_logger().info(f"[CONTROL] Final Cmd (vx, vy): ({vel_x_target:.2f}, {vel_y_target:.2f})")
            self.get_logger().info(f"[CONTROL] Orientation (Yaw): {math.degrees(yaw):.1f}°")

        msg = Twist()
        # Acceleration Limiting (Slew Rate)
        if not hasattr(self, 'current_vx'): self.current_vx, self.current_vy = 0.0, 0.0
        accel_step = 0.04 # 0.8 m/s^2 ramping
        
        if vel_x_target > self.current_vx: self.current_vx = min(vel_x_target, self.current_vx + accel_step)
        else: self.current_vx = max(vel_x_target, self.current_vx - accel_step)
        
        if vel_y_target > self.current_vy: self.current_vy = min(vel_y_target, self.current_vy + accel_step)
        else: self.current_vy = max(vel_y_target, self.current_vy - accel_step)

        msg.linear.x = self.current_vx
        msg.linear.y = self.current_vy
        
        # Orient toward goal (Holonomic Damping)
        # We only rotate if the error is large (> 15 degrees)
        target_yaw_world = math.atan2(dy_world, dx_world)
        yaw_err = target_yaw_world - yaw
        while yaw_err > math.pi: yaw_err -= 2*math.pi
        while yaw_err < -math.pi: yaw_err += 2*math.pi
        
        if abs(yaw_err) > 0.25: # Only rotate if significantly off
            msg.angular.z = max(min(2.5 * yaw_err, w_max), -w_max)
        else:
            msg.angular.z = 0.0
            
        self.cmd_vel_pub.publish(msg)

    def publish_visuals(self):
        # 1. State/Markers Container
        markers = MarkerArray()
        
        # 2. Potential Field Force Markers (Dynamic)
        # These are only valid if we are actually calculating forces (not shown here for brevity, 
        # but in a real fix we'd only add them if f_total exists)
        if hasattr(self, 'prev_f_x'):
            markers.markers.append(self.create_force_marker(0, self.prev_f_x, self.prev_f_y, ColorRGBA(r=0.4, g=0.4, b=1.0, a=1.0), "Total"))
            # Could add others here
        
        # 3. Trail Visualization (Breadcrumbs)
        if not hasattr(self, '_path_tick'): self._path_tick = 0
        self._path_tick += 1
        if self._path_tick % 4 == 0:
            p_stamped = PoseStamped()
            p_stamped.header.frame_id = "odom"
            p_stamped.header.stamp = self.get_clock().now().to_msg()
            p_stamped.pose = self.current_pose
            self.actual_path.poses.append(p_stamped)
            self.trail_pub.publish(self.actual_path)
            
        # 4. Detected Obstacles (Geometric Visualization)
        if hasattr(self, 'detected_obstacles'):
            obs_markers = MarkerArray()
            for i, obs in enumerate(self.detected_obstacles):
                m = Marker()
                m.header.frame_id = "base_footprint"
                m.header.stamp = self.get_clock().now().to_msg()
                m.ns = "detections"
                m.id = i
                m.type = Marker.CYLINDER
                m.action = Marker.ADD
                m.pose.position.x = obs['center'][0]
                m.pose.position.y = obs['center'][1]
                m.pose.position.z = 0.05
                m.scale.x = obs['radius'] * 2.0
                m.scale.y = obs['radius'] * 2.0
                m.scale.z = 0.1
                m.color = ColorRGBA(r=0.0, g=0.8, b=1.0, a=0.3) # Semi-transparent Light Blue
                obs_markers.markers.append(m)
            self.pub_obs_markers.publish(obs_markers)
        
        self.marker_pub.publish(markers)
        
        # 9. Reference Static Markers (Start & Origin)
        if self.start_pose:
            nav_markers = MarkerArray()
            
            # Start Marker (Green Disc)
            start_m = Marker()
            start_m.header.frame_id = "odom"
            start_m.header.stamp = self.get_clock().now().to_msg()
            start_m.ns = "reference"
            start_m.id = 0
            start_m.type = Marker.CYLINDER
            start_m.action = Marker.ADD
            start_m.pose = self.start_pose
            start_m.pose.position.z = 0.005
            start_m.scale.x, start_m.scale.y, start_m.scale.z = 0.4, 0.4, 0.01
            start_m.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.6)
            nav_markers.markers.append(start_m)
            
            # Origin Marker (Red Disc)
            orig_m = Marker()
            orig_m.header.frame_id = "odom"
            orig_m.header.stamp = self.get_clock().now().to_msg()
            orig_m.ns = "reference"
            orig_m.id = 1
            orig_m.type = Marker.CYLINDER
            orig_m.action = Marker.ADD
            orig_m.pose.position.x, orig_m.pose.position.y, orig_m.pose.position.z = 0.0, 0.0, 0.0
            orig_m.scale.x, orig_m.scale.y, orig_m.scale.z = 0.5, 0.5, 0.01
            orig_m.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.5)
            nav_markers.markers.append(orig_m)
            
            self.pub_nav_markers.publish(nav_markers)

    def stop_robot(self):
        self.cmd_vel_pub.publish(Twist())
        self.marker_pub.publish(MarkerArray())

def main(args=None):
    rclpy.init(args=args)
    node = APFPlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
