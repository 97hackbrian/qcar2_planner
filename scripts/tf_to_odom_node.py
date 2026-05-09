#!/usr/bin/env python3
"""
TF to Odom + EKF Node

This node provides a stable, high-rate (100Hz) odometry source by fusing:
1. Low-rate Pose updates from Cartographer (via TF odom->base_link).
2. High-rate Angular Velocity from IMU.

Algorithm (Simplified EKF/Complementary Filter) with Anti-Oscillation:
- Prediction (100Hz): Integrate IMU gyro for heading. Dead-reckon position using last known velocity.
- Correction (~5-20Hz): When TF updates, correct position/heading ONLY if change exceeds tolerance.

Tolerances:
- position_tolerance: min distance change (m) required to update position target.
- orientation_tolerance: min angle change (rad) required to update orientation target.
- filter_alpha_*: Smoothing factor (0.01 - 1.0). Lower = smoother but slower response.

Publishes:
- /odom_filtered (nav_msgs/Odometry) - Local smooth odometry (frame: odom)
- /odom_filtered_pose (geometry_msgs/PoseStamped) - Global smooth pose (frame: map) for Nvblox.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, PoseStamped
import math

def euler_from_quaternion(q):
    t3 = +2.0 * (q.w * q.z + q.x * q.y)
    t4 = +1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(t3, t4)

def quaternion_from_euler(roll, pitch, yaw):
    q = Quaternion()
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy
    return q

def angle_diff(a, b):
    d = a - b
    while d > math.pi: d -= 2*math.pi
    while d < -math.pi: d += 2*math.pi
    return d

class TfToOdomEKF(Node):
    def __init__(self):
        super().__init__('tf_to_odom_ekf')

        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('output_topic', '/odom_filtered')
        
        # Filter parameters - Tuned for stability
        self.declare_parameter('filter_alpha_pos', 0.05) 
        self.declare_parameter('filter_alpha_yaw', 0.02)
        # Tolerance parameters (Anti-Oscillation)
        self.declare_parameter('position_tolerance', 0.02) # meters
        self.declare_parameter('orientation_tolerance', 0.02) # radians

        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.map_frame = self.get_parameter('map_frame').value
        self.output_topic = self.get_parameter('output_topic').value
        self.alpha_pos = self.get_parameter('filter_alpha_pos').value
        self.alpha_yaw = self.get_parameter('filter_alpha_yaw').value
        self.pos_tol = self.get_parameter('position_tolerance').value
        self.ori_tol = self.get_parameter('orientation_tolerance').value

        # TF Listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # State [x, y, theta] (Local in odom frame)
        self.x = 0.0
        self.y = 0.0
        self.th = 0.0
        
        # Velocity estimation state
        self.v = 0.0
        self.w = 0.0
        
        # IMU state
        self.gyro_z = 0.0
        self.last_imu_time = self.get_clock().now()

        # Calibration
        self.gyro_bias = 0.0
        self.cal_samples = 200
        self.cal_buffer = []
        self.is_calibrating = True

        # Last TF update timestamp
        self.last_tf_time = Time(seconds=0)
        self.last_accepted_tf_pos = (0.0, 0.0)
        self.last_accepted_tf_yaw = 0.0
        self.tf_initialized = False

        # QoS
        qos_imu = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Imu, '/qcar2_imu', self._imu_cb, qos_imu)
        
        self.odom_pub = self.create_publisher(Odometry, self.output_topic, 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/odom_filtered_pose', 10)

        # Main Loop (100Hz) - Prediction & Publishing
        self.create_timer(0.01, self._loop)
        
        self.get_logger().info(f"🚀 TF-to-Odom EKF Started! PosTol={self.pos_tol} OriTol={self.ori_tol}")

    def _imu_cb(self, msg):
        self.gyro_z = msg.angular_velocity.z
        
        if self.is_calibrating:
            self.cal_buffer.append(self.gyro_z)
            if len(self.cal_buffer) >= self.cal_samples:
                self.gyro_bias = sum(self.cal_buffer) / len(self.cal_buffer)
                self.is_calibrating = False
                self.get_logger().info(f"✅ Gyro Calibrated! Bias: {self.gyro_bias:.5f} rad/s")

    def _loop(self):
        if self.is_calibrating: return

        now = self.get_clock().now()
        dt = (now - self.last_imu_time).nanoseconds * 1e-9
        self.last_imu_time = now
        
        if dt > 0.1 or dt <= 0.0: return 

        # ── 1. Update from TF (Correction Step) ─────────────────────
        try:
            t = self.tf_buffer.lookup_transform(
                self.odom_frame,
                self.base_frame,
                Time()) 
            
            px = t.transform.translation.x
            py = t.transform.translation.y
            q = t.transform.rotation
            pth = euler_from_quaternion(q)
            
            tf_time = Time.from_msg(t.header.stamp)
            
            if not self.tf_initialized:
                self.x, self.y, self.th = px, py, pth
                self.last_tf_time = tf_time
                self.last_accepted_tf_pos = (px, py)
                self.last_accepted_tf_yaw = pth
                self.tf_initialized = True
                self.get_logger().info("✅ Initial Pose acquired from TF")
            
            else:
                dt_tf = (tf_time - self.last_tf_time).nanoseconds * 1e-9
                if dt_tf > 0.001:
                    last_px, last_py = self.last_accepted_tf_pos
                    dist_sq = (px - last_px)**2 + (py - last_py)**2
                    yaw_diff = abs(angle_diff(pth, self.last_accepted_tf_yaw))
                    
                    is_moved = True
                    if dist_sq < (self.pos_tol**2) and yaw_diff < self.ori_tol:
                        is_moved = False
                        if abs(self.v) < 0.01 and abs(self.w) < 0.01: pass
                    
                    if is_moved:
                        dist = math.sqrt(dist_sq)
                        v_meas = dist / dt_tf
                        
                        dx = px - last_px
                        dy = py - last_py
                        heading_vec = (math.cos(pth), math.sin(pth))
                        dot = dx*heading_vec[0] + dy*heading_vec[1]
                        if dot < -0.01: v_meas = -v_meas 
                        
                        self.v = 0.95 * self.v + 0.05 * v_meas
                        
                        err_x = px - self.x
                        err_y = py - self.y
                        err_th = angle_diff(pth, self.th)
                        
                        self.x += self.alpha_pos * err_x
                        self.y += self.alpha_pos * err_y
                        self.th += self.alpha_yaw * err_th
                        
                        self.last_accepted_tf_pos = (px, py)
                        self.last_accepted_tf_yaw = pth
                        self.last_tf_time = tf_time
                    else:
                        self.last_tf_time = tf_time
                        self.v *= 0.9

        except Exception:
            pass 

        # ── 2. Prediction Step (Dead Reckoning) ─────────────────────
        omega = self.gyro_z - self.gyro_bias
        if abs(omega) < 0.005: omega = 0.0
        
        self.th += omega * dt
        self.th = math.atan2(math.sin(self.th), math.cos(self.th))

        if abs(self.v) < 0.005: self.v = 0.0

        self.x += self.v * math.cos(self.th) * dt
        self.y += self.v * math.sin(self.th) * dt
        self.w = omega 

        # ── 3. Publish ──────────────────────────────────────────────
        self._publish_odom(now)
        self._publish_map_pose(now) # Publish global pose for Nvblox

    def _publish_odom(self, now):
        msg = Odometry()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = self.odom_frame
        msg.child_frame_id = self.base_frame
        msg.pose.pose.position.x = self.x
        msg.pose.pose.position.y = self.y
        msg.pose.pose.orientation = quaternion_from_euler(0, 0, self.th)
        msg.twist.twist.linear.x = self.v
        msg.twist.twist.angular.z = self.w
        
        cov_p = 0.05 if abs(self.v) > 0.01 else 0.001
        msg.pose.covariance = [cov_p]*36 # Simplified but valid
        self.odom_pub.publish(msg)

    def _publish_map_pose(self, now):
        """
        Calculates Global Pose (in map frame) by combining:
        1. Local Filtered Pose (odom -> base_link_filtered)
        2. Map Correction (map -> odom) from Cartographer Look up
        """
        try:
            # Lookup T_map_odom
            t_map_odom = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.odom_frame,
                Time())

            # Transform our local pose (x, y, th) to global frame
            # P_map = T_map_odom * P_local
            
            # Extract Map->Odom transform components
            tx, ty = t_map_odom.transform.translation.x, t_map_odom.transform.translation.y
            q = t_map_odom.transform.rotation
            th_map_odom = euler_from_quaternion(q)

            # Composition logic (2D)
            # Global Theta = MapOdom_Theta + Local_Theta
            global_th = th_map_odom + self.th
            global_th = math.atan2(math.sin(global_th), math.cos(global_th))

            # Global Position = MapOdom_Pos + Rotate(Local_Pos)
            # Rotate local pos by map_odom theta
            c = math.cos(th_map_odom)
            s = math.sin(th_map_odom)
            
            global_x = tx + (c * self.x - s * self.y)
            global_y = ty + (s * self.x + c * self.y)

            # Publish PoseStamped
            pose_msg = PoseStamped()
            pose_msg.header.stamp = now.to_msg()
            pose_msg.header.frame_id = self.map_frame
            pose_msg.pose.position.x = global_x
            pose_msg.pose.position.y = global_y
            pose_msg.pose.orientation = quaternion_from_euler(0, 0, global_th)

            self.pose_pub.publish(pose_msg)

        except Exception as e:
            # Throttle error logging
            now_sec = now.nanoseconds * 1e-9
            if not hasattr(self, 'last_map_error_time') or (now_sec - self.last_map_error_time > 5.0):
                self.get_logger().warn(f"⚠️ Could not publish map pose: {e}. Is Cartographer running?")
                self.last_map_error_time = now_sec

def main(args=None):
    rclpy.init(args=args)
    node = TfToOdomEKF()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
