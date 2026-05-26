#!/usr/bin/env python3
# =============================================================================
# map_overlay_node.py — Map Overlay & Alignment Node (TF Tree Master)
# =============================================================================
#
# Loads a previously saved 2D map (PGM + YAML) containing lane boundaries.
# Creates a new root frame ('pgm_map') and publishes the PGM map there.
# Computes and broadcasts the TF: pgm_map -> map (Cartographer).
#
# Alignment modes:
#   - 'auto':   Uses Point-to-Point ICP (SVD) to align the maps automatically.
#               (Aligns Carto points to PGM points)
#   - 'manual': Waits for a 2D Pose Estimate (/initialpose) from RViz2.
#               (Computes offset between clicked pose and Carto robot pose)
#
# Publishes:
#   - /grid_map (GridMap): for directional_planner_server
#   - /planner_occupancy (OccupancyGrid): for RViz2
#   - /planner_uncertainty (OccupancyGrid): for RViz2
#   - TF: pgm_map -> map
# =============================================================================

import os
import math
import yaml
import traceback
import numpy as np
import cv2
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from nav_msgs.msg import OccupancyGrid
from grid_map_msgs.msg import GridMap as GridMapMsg
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from std_srvs.srv import Trigger

import tf2_ros
from tf2_ros import Buffer, TransformListener, TransformBroadcaster

def get_rigid_transform(A, B):
    """
    Computes optimal rigid transform (R, t) mapping points A to B using SVD.
    A, B are Nx2 numpy arrays.
    Returns R (2x2), t (2x1).
    """
    assert len(A) == len(B)
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    AA = A - centroid_A
    BB = B - centroid_B

    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = np.dot(Vt.T, U.T)

    t = centroid_B.T - np.dot(R, centroid_A.T)
    return R, t

def icp_2d(source, target, max_iterations=50, tolerance=0.001, max_dist=0.5):
    """
    Point-to-point ICP in 2D with dynamic thresholding (annealing).
    """
    src = np.copy(source)
    total_R = np.eye(2)
    total_t = np.zeros(2)
    prev_error = float('inf')

    for i in range(max_iterations):
        # Annealing: start with a large search radius to pull distant walls, then shrink to max_dist
        current_max_dist = max_dist
        if i < max_iterations // 2:
            current_max_dist = max_dist + (max_dist * 4.0) * (1.0 - (i / (max_iterations // 2)))

        distances = np.linalg.norm(src[:, np.newaxis, :] - target[np.newaxis, :, :], axis=2)
        indices = np.argmin(distances, axis=1)
        min_distances = np.min(distances, axis=1)

        valid = min_distances < current_max_dist
        matched_src = src[valid]
        matched_tgt = target[indices[valid]]

        if len(matched_src) < 10:
            return None, "Not enough matches"

        R, t = get_rigid_transform(matched_src, matched_tgt)
        src = np.dot(src, R.T) + t

        total_R = np.dot(R, total_R)
        total_t = np.dot(R, total_t) + t

        mean_error = np.mean(min_distances[valid])
        if abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    theta = math.atan2(total_R[1, 0], total_R[0, 0])
    return (total_t[0], total_t[1], theta), None

def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))

class MapOverlayNode(Node):
    def __init__(self):
        super().__init__('map_overlay_node')
        self.get_logger().info('=== MapOverlayNode (TF Master) __init__ START ===')

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter('map_yaml_path', '')
        self.declare_parameter('alignment_mode', 'manual')
        self.declare_parameter('icp_max_iterations', 50)
        self.declare_parameter('icp_convergence_threshold', 0.001)
        self.declare_parameter('icp_max_correspondence_dist', 1.0)
        self.declare_parameter('icp_downsample_resolution', 0.2)
        self.declare_parameter('icp_min_cartographer_points', 50)
        self.declare_parameter('overlay_max_lidar_range_m', 6.0)
        self.declare_parameter('icp_min_wall_component_size', 25)
        self.declare_parameter('morph_kernel_size', 3)
        self.declare_parameter('border_dilation_px', 2)
        self.declare_parameter('pgm_scale_factor', 0.495) # New scale factor parameter
        self.declare_parameter('publish_rate', 0.5)
        self.declare_parameter('map_frame', 'map')       # Cartographer frame
        self.declare_parameter('pgm_frame', 'pgm_map')   # New Root Frame
        self.declare_parameter('occupancy_out_topic', '/planner_occupancy')
        self.declare_parameter('uncertainty_out_topic', '/planner_uncertainty')
        self.declare_parameter('gridmap_out_topic', '/grid_map')

        self.map_yaml_path = str(self.get_parameter('map_yaml_path').value)
        self.alignment_mode = str(self.get_parameter('alignment_mode').value).lower()
        self.icp_max_iter = int(self.get_parameter('icp_max_iterations').value)
        self.icp_tol = float(self.get_parameter('icp_convergence_threshold').value)
        self.icp_max_dist = float(self.get_parameter('icp_max_correspondence_dist').value)
        self.icp_ds_res = float(self.get_parameter('icp_downsample_resolution').value)
        self.icp_min_pts = int(self.get_parameter('icp_min_cartographer_points').value)
        self.overlay_max_lidar_range_m = float(self.get_parameter('overlay_max_lidar_range_m').value)
        self.icp_min_wall_component_size = int(self.get_parameter('icp_min_wall_component_size').value)
        self.morph_kernel_size = int(self.get_parameter('morph_kernel_size').value)
        self.border_dilation_px = int(self.get_parameter('border_dilation_px').value)
        self.pgm_scale_factor = float(self.get_parameter('pgm_scale_factor').value)
        self.publish_rate = float(self.get_parameter('publish_rate').value)
        self.map_frame = str(self.get_parameter('map_frame').value)
        self.pgm_frame = str(self.get_parameter('pgm_frame').value)
        
        occ_topic = str(self.get_parameter('occupancy_out_topic').value)
        unc_topic = str(self.get_parameter('uncertainty_out_topic').value)
        gm_topic = str(self.get_parameter('gridmap_out_topic').value)

        # ── State (TF: pgm_map -> map) ──────────────────────────────────────
        self.lock = threading.Lock()
        self.transform_locked = False
        self.tf_dx = 0.0
        self.tf_dy = 0.0
        self.tf_dtheta = 0.0
        
        # Lane layers (static)
        self.lane_mask = None
        self.lane_dir_x = None
        self.lane_dir_y = None
        self.independent_lanes = {} # internal_id -> {mask, dir_x, dir_y, name, original_name}
        
        # PGM Map data (static)
        self.pgm_occupancy = None
        self.pgm_uncertainty = None
        self.pgm_points = None
        self.pgm_resolution = 0.05
        self.pgm_origin_x = 0.0
        self.pgm_origin_y = 0.0
        self.pgm_rows = 0
        self.pgm_cols = 0

        # Cartographer map data
        self.carto_points = None

        # ── TF2 ─────────────────────────────────────────────────────────────
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        # ── Load Map ────────────────────────────────────────────────────────
        if not self.map_yaml_path or not os.path.isfile(self.map_yaml_path):
            self.get_logger().fatal(f'Valid map_yaml_path required. Got: {self.map_yaml_path}')
            raise FileNotFoundError(f'{self.map_yaml_path}')
        self._load_pgm_map()
        self._load_lanes()

        # ── Publishers ──────────────────────────────────────────────────────
        qos_latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.occ_pub = self.create_publisher(OccupancyGrid, occ_topic, qos_latched)
        self.unc_pub = self.create_publisher(OccupancyGrid, unc_topic, qos_latched)
        self.gm_pub = self.create_publisher(GridMapMsg, gm_topic, 10)

        # ── Subscribers ─────────────────────────────────────────────────────
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.cartographer_map_cb, qos_latched
        )
        self.initialpose_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/initialpose', self.initialpose_cb, 10
        )

        # ── Services ────────────────────────────────────────────────────────
        self.align_srv = self.create_service(Trigger, '/align_map', self.align_srv_cb)

        # ── Timers ──────────────────────────────────────────────────────────
        self.pub_timer = self.create_timer(1.0 / max(0.1, self.publish_rate), self._publish_all)
        self.align_timer = self.create_timer(1.0, self._auto_align_tick)
        self.tf_timer = self.create_timer(0.02, self._publish_tf) # 50 Hz

        self.get_logger().info(f'Node ready. Mode: {self.alignment_mode}. Root Frame: {self.pgm_frame}')

    # =====================================================================
    # Map Loading (PGM)
    # =====================================================================
    def _load_pgm_map(self):
        yaml_dir = os.path.dirname(os.path.abspath(self.map_yaml_path))
        with open(self.map_yaml_path, 'r') as f:
            meta = yaml.safe_load(f)

        pgm_name = meta.get('image', '')
        self.pgm_resolution = float(meta.get('resolution', 0.05))
        origin = meta.get('origin', [0.0, 0.0, 0.0])
        self.pgm_origin_x = float(origin[0])
        self.pgm_origin_y = float(origin[1])
        negate = int(meta.get('negate', 0))
        occ_thresh = float(meta.get('occupied_thresh', 0.65))
        free_thresh = float(meta.get('free_thresh', 0.196))

        pgm_path = os.path.join(yaml_dir, pgm_name)
        raw_img = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)

        if negate:
            img_f = raw_img.astype(np.float32) / 255.0
        else:
            img_f = (255.0 - raw_img.astype(np.float32)) / 255.0

        img_f = np.flipud(img_f)

        if self.pgm_scale_factor != 1.0:
            new_w = int(img_f.shape[1] * self.pgm_scale_factor)
            new_h = int(img_f.shape[0] * self.pgm_scale_factor)
            img_f = cv2.resize(img_f, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        k = self.morph_kernel_size
        if k > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            img_f = cv2.morphologyEx(img_f, cv2.MORPH_OPEN, kernel)
            img_f = cv2.morphologyEx(img_f, cv2.MORPH_CLOSE, kernel)

        self.pgm_rows, self.pgm_cols = img_f.shape
        self.pgm_occupancy = np.full((self.pgm_rows, self.pgm_cols), -1.0, dtype=np.float32)
        self.pgm_occupancy[img_f <= free_thresh] = 0.0
        
        wall_mask = img_f >= occ_thresh
        if self.border_dilation_px > 0:
            dk = self.border_dilation_px * 2 + 1
            dkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dk, dk))
            wall_mask = cv2.dilate(wall_mask.astype(np.uint8), dkernel) > 0
            
        self.pgm_occupancy[wall_mask] = 1.0
        self.pgm_uncertainty = np.ones((self.pgm_rows, self.pgm_cols), dtype=np.float32)
        free_mask = self.pgm_occupancy == 0.0
        self.pgm_uncertainty[free_mask] = np.clip(img_f[free_mask] / max(free_thresh, 0.01), 0.0, 1.0)

        y_idx, x_idx = np.where(wall_mask)
        pts_x = self.pgm_origin_x + (x_idx + 0.5) * self.pgm_resolution
        pts_y = self.pgm_origin_y + (y_idx + 0.5) * self.pgm_resolution
        points = np.column_stack((pts_x, pts_y))
        self.pgm_points = self._voxel_downsample(points, self.icp_ds_res)

        self.get_logger().info(f'PGM loaded. Origin: ({self.pgm_origin_x}, {self.pgm_origin_y}). Points: {len(self.pgm_points)}')

    def _voxel_downsample(self, points, voxel_size):
        if len(points) == 0: return points
        voxel_indices = np.floor(points / voxel_size).astype(np.int32)
        _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
        return points[unique_indices]

    def _load_lanes(self):
        self.lane_mask = np.zeros((self.pgm_rows, self.pgm_cols), dtype=np.float32)
        self.lane_dir_x = np.zeros((self.pgm_rows, self.pgm_cols), dtype=np.float32)
        self.lane_dir_y = np.zeros((self.pgm_rows, self.pgm_cols), dtype=np.float32)
        self.independent_lanes = {}
        
        lanes_yaml_path = os.path.join(os.path.dirname(self.map_yaml_path), 'lanes.yaml')
        if not os.path.isfile(lanes_yaml_path):
            self.get_logger().warn(f'No lanes.yaml found at {lanes_yaml_path}. Lane layers will be empty.')
            return
            
        with open(lanes_yaml_path, 'r') as f:
            lanes_data = yaml.safe_load(f)
            
        if not lanes_data or 'lanes' not in lanes_data:
            self.get_logger().warn('lanes.yaml is empty or invalid format.')
            return
            
        lanes = lanes_data['lanes']
        self.get_logger().info(f'Loaded {len(lanes)} lanes from config.')
        
        # 1) Build independent lane grids
        for idx, lane in enumerate(lanes):
            internal_id = f"lane_{idx + 1:03d}"
            original_name = lane.get('name', '')
            name = original_name if original_name else internal_id
            
            width = float(lane.get('width', 0.0))
            points = lane.get('points', [])
            
            if not points or width <= 0.0:
                self.get_logger().warn(f'Skipping {internal_id} ({name}): missing points or width.')
                continue
                
            if len(points) < 2:
                continue
                
            width_px = width / self.pgm_resolution
            
            lane_mask = np.zeros((self.pgm_rows, self.pgm_cols), dtype=np.float32)
            lane_dir_x = np.zeros((self.pgm_rows, self.pgm_cols), dtype=np.float32)
            lane_dir_y = np.zeros((self.pgm_rows, self.pgm_cols), dtype=np.float32)
            
            for i in range(len(points) - 1):
                p1 = points[i]
                p2 = points[i+1]
                
                c1 = int((p1[0] - self.pgm_origin_x) / self.pgm_resolution - 0.5)
                r1 = int((p1[1] - self.pgm_origin_y) / self.pgm_resolution - 0.5)
                c2 = int((p2[0] - self.pgm_origin_x) / self.pgm_resolution - 0.5)
                r2 = int((p2[1] - self.pgm_origin_y) / self.pgm_resolution - 0.5)
                
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                mag = math.sqrt(dx*dx + dy*dy)
                if mag < 1e-6:
                    continue
                ux = dx / mag
                uy = dy / mag
                
                thickness = max(1, int(width_px))
                cv2.line(lane_mask, (c1, r1), (c2, r2), 1.0, thickness)
                cv2.line(lane_dir_x, (c1, r1), (c2, r2), float(ux), thickness)
                cv2.line(lane_dir_y, (c1, r1), (c2, r2), float(uy), thickness)
            
            self.independent_lanes[internal_id] = {
                'original_name': original_name,
                'name': name,
                'mask': lane_mask,
                'dir_x': lane_dir_x,
                'dir_y': lane_dir_y
            }
            
            # Combine into global visualization mask
            cv2.max(self.lane_mask, lane_mask, self.lane_mask)
            mask_indices = lane_mask > 0.0
            self.lane_dir_x[mask_indices] = lane_dir_x[mask_indices]
            self.lane_dir_y[mask_indices] = lane_dir_y[mask_indices]
                
        n_mask = int(np.sum(self.lane_mask > 0))
        self.get_logger().info(f'Total lane_mask cells: {n_mask}')

    # =====================================================================
    # Callbacks
    # =====================================================================
    def cartographer_map_cb(self, msg: OccupancyGrid):
        if self.transform_locked and self.alignment_mode == 'auto':
            return

        res = msg.info.resolution
        w, h = msg.info.width, msg.info.height
        ox, oy = msg.info.origin.position.x, msg.info.origin.position.y

        data = np.array(msg.data, dtype=np.int8).reshape((h, w))
        wall_mask = data >= 50

        # Filter out tiny disconnected wall speckles caused by noisy /scan returns.
        # This keeps the ICP input focused on real map structures.
        if self.icp_min_wall_component_size > 1 and np.any(wall_mask):
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                wall_mask.astype(np.uint8), connectivity=8
            )
            cleaned = np.zeros_like(wall_mask)
            for label in range(1, num_labels):
                area = int(stats[label, cv2.CC_STAT_AREA])
                if area >= self.icp_min_wall_component_size:
                    cleaned[labels == label] = True
            wall_mask = cleaned

        y_idx, x_idx = np.where(wall_mask)
        
        pts_x = ox + (x_idx + 0.5) * res
        pts_y = oy + (y_idx + 0.5) * res
        points = np.column_stack((pts_x, pts_y))

        if self.overlay_max_lidar_range_m > 0.0 and len(points) > 0:
            try:
                t = self.tf_buffer.lookup_transform(self.map_frame, 'base_link', rclpy.time.Time())
                rx = t.transform.translation.x
                ry = t.transform.translation.y
                d2 = (points[:, 0] - rx) ** 2 + (points[:, 1] - ry) ** 2
                points = points[d2 <= (self.overlay_max_lidar_range_m ** 2)]
            except Exception:
                pass

        self.carto_points = self._voxel_downsample(points, self.icp_ds_res)

    def initialpose_cb(self, msg: PoseWithCovarianceStamped):
        # Allow /initialpose in BOTH modes.
        # In manual mode: it locks the transform exactly as clicked.
        # In auto mode: it uses the click as an INITIAL GUESS and lets ICP refine it.
        try:
            # Robot pose in Cartographer's map frame
            t = self.tf_buffer.lookup_transform(self.map_frame, 'base_link', rclpy.time.Time())
            rx = t.transform.translation.x
            ry = t.transform.translation.y
            rq = t.transform.rotation
            ryaw = math.atan2(2.0*(rq.w*rq.z + rq.x*rq.y), 1.0 - 2.0*(rq.y**2 + rq.z**2))
            
            # User clicked pose in pgm_map frame
            cx = msg.pose.pose.position.x
            cy = msg.pose.pose.position.y
            cq = msg.pose.pose.orientation
            cyaw = math.atan2(2.0*(cq.w*cq.z + cq.x*cq.y), 1.0 - 2.0*(cq.y**2 + cq.z**2))

            # T_pgm_map = T_pgm_robot * T_map_robot^-1
            dyaw = normalize_angle(cyaw - ryaw)
            dx = cx - (rx * math.cos(dyaw) - ry * math.sin(dyaw))
            dy = cy - (rx * math.sin(dyaw) + ry * math.cos(dyaw))
            
            with self.lock:
                self.tf_dx, self.tf_dy, self.tf_dtheta = dx, dy, dyaw
                if self.alignment_mode == 'manual':
                    self.transform_locked = True
                    self.get_logger().info(f'Manual alignment locked! TF: dx={dx:.2f}, dy={dy:.2f}, dth={math.degrees(dyaw):.1f}°')
                else:
                    self.transform_locked = False # Unlock to let ICP run from this new seed
                    self.get_logger().info(f'Seed alignment applied! ICP will now refine from: dx={dx:.2f}, dy={dy:.2f}, dth={math.degrees(dyaw):.1f}°')
            
        except Exception as e:
            self.get_logger().error(f'Failed to compute initial alignment: {e}')

    def align_srv_cb(self, request, response):
        with self.lock:
            self.transform_locked = False
        response.success = True
        response.message = f"Unlocked alignment. Mode: {self.alignment_mode}"
        self.get_logger().info('Alignment unlocked via service call.')
        return response

    # =====================================================================
    # Auto Alignment (ICP)
    # =====================================================================
    def _auto_align_tick(self):
        if self.alignment_mode != 'auto' or self.transform_locked:
            return

        if self.carto_points is None or len(self.carto_points) < self.icp_min_pts:
            return

        if self.pgm_points is None or len(self.pgm_points) < 10:
            return

        self.get_logger().info('Starting ICP alignment...')
        
        # Apply the CURRENT tf_dx, tf_dy, tf_dtheta as the initial guess for ICP
        initial_R = np.array([
            [math.cos(self.tf_dtheta), -math.sin(self.tf_dtheta)],
            [math.sin(self.tf_dtheta),  math.cos(self.tf_dtheta)]
        ])
        # Transform carto points by current guess
        seeded_carto = np.dot(self.carto_points, initial_R.T) + np.array([self.tf_dx, self.tf_dy])

        res, err = icp_2d(
            seeded_carto, self.pgm_points,
            max_iterations=self.icp_max_iter,
            tolerance=self.icp_tol,
            max_dist=self.icp_max_dist
        )

        if res is not None:
            dx, dy, dtheta = res
            # Accumulate the transform: T_final = T_icp * T_initial
            # final_theta = initial_theta + icp_theta
            final_theta = normalize_angle(self.tf_dtheta + dtheta)
            
            # final_translation = R_icp * initial_translation + t_icp
            R_icp = np.array([
                [math.cos(dtheta), -math.sin(dtheta)],
                [math.sin(dtheta),  math.cos(dtheta)]
            ])
            t_icp = np.array([dx, dy])
            t_initial = np.array([self.tf_dx, self.tf_dy])
            final_t = np.dot(R_icp, t_initial) + t_icp

            with self.lock:
                self.tf_dx, self.tf_dy, self.tf_dtheta = final_t[0], final_t[1], final_theta
                self.transform_locked = True
            self.get_logger().info(f'ICP Converged! Final TF: dx={final_t[0]:.2f}, dy={final_t[1]:.2f}, dth={math.degrees(final_theta):.1f}°')
        else:
            self.get_logger().warn(f'ICP failed: {err}')

    # =====================================================================
    # High-Rate TF Broadcaster
    # =====================================================================
    def _publish_tf(self):
        with self.lock:
            dx, dy, dtheta = self.tf_dx, self.tf_dy, self.tf_dtheta

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.pgm_frame
        t.child_frame_id = self.map_frame
        t.transform.translation.x = dx
        t.transform.translation.y = dy
        t.transform.translation.z = 0.0
        t.transform.rotation.z = math.sin(dtheta / 2.0)
        t.transform.rotation.w = math.cos(dtheta / 2.0)
        self.tf_broadcaster.sendTransform(t)

    # =====================================================================
    # PGM Map Publishing (Static, No Transformations!)
    # =====================================================================
    def _publish_all(self):
        try:
            stamp = self.get_clock().now().to_msg()
            
            if self.pgm_occupancy is None: return

            self._pub_occ(stamp, self.pgm_occupancy, self.occ_pub, 100)
            self._pub_unc(stamp, self.pgm_occupancy, self.pgm_uncertainty)
            self._pub_gridmap(stamp, self.pgm_occupancy, self.pgm_uncertainty)
            
        except Exception as e:
            self.get_logger().error(f'Publish error: {e}\n{traceback.format_exc()}')

    def _pub_occ(self, stamp, occ, publisher, wall_val):
        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = self.pgm_frame
        msg.info.resolution = float(self.pgm_resolution)
        msg.info.width = int(self.pgm_cols)
        msg.info.height = int(self.pgm_rows)
        msg.info.origin.position.x = float(self.pgm_origin_x)
        msg.info.origin.position.y = float(self.pgm_origin_y)
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0

        grid = np.full(self.pgm_rows * self.pgm_cols, -1, dtype=np.int8)
        flat_occ = occ.ravel()
        grid[flat_occ == 0.0] = 0
        grid[flat_occ == 1.0] = wall_val
        msg.data = grid.tolist()
        publisher.publish(msg)
        
    def _pub_unc(self, stamp, occ, unc):
        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = self.pgm_frame
        msg.info.resolution = float(self.pgm_resolution)
        msg.info.width = int(self.pgm_cols)
        msg.info.height = int(self.pgm_rows)
        msg.info.origin.position.x = float(self.pgm_origin_x)
        msg.info.origin.position.y = float(self.pgm_origin_y)
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0

        flat_occ = occ.ravel()
        flat_unc = unc.ravel()
        grid = np.full(self.pgm_rows * self.pgm_cols, -1, dtype=np.int8)
        free_mask = flat_occ == 0.0
        grid[free_mask] = np.clip(flat_unc[free_mask] * 100.0, 0, 100).astype(np.int8)
        
        wall_mask = flat_occ == 1.0
        grid[wall_mask] = 100
        
        msg.data = grid.tolist()
        self.unc_pub.publish(msg)

    def _pub_gridmap(self, stamp, occ, unc):
        msg = GridMapMsg()
        msg.header.stamp = stamp
        msg.header.frame_id = self.pgm_frame

        msg.info.resolution = float(self.pgm_resolution)
        size_x = self.pgm_cols * self.pgm_resolution
        size_y = self.pgm_rows * self.pgm_resolution
        msg.info.length_x = float(size_x)
        msg.info.length_y = float(size_y)
        
        msg.info.pose.position.x = float(self.pgm_origin_x + size_x / 2.0)
        msg.info.pose.position.y = float(self.pgm_origin_y + size_y / 2.0)
        msg.info.pose.position.z = 0.0
        msg.info.pose.orientation.w = 1.0

        if self.lane_mask is None:
            lane_mask = np.zeros_like(occ)
            dir_x = np.zeros_like(occ)
            dir_y = np.zeros_like(occ)
        else:
            lane_mask = self.lane_mask
            dir_x = self.lane_dir_x
            dir_y = self.lane_dir_y

        names = ['occupancy', 'uncertainty', 'lane_mask', 'dir_x', 'dir_y']
        arrays = [occ, unc, lane_mask, dir_x, dir_y]
        msg.layers = names
        msg.basic_layers = ['occupancy', 'uncertainty']

        n_x, n_y = self.pgm_cols, self.pgm_rows

        for layer_data in arrays:
            arr = Float32MultiArray()
            d0 = MultiArrayDimension(label='column_index', size=n_y, stride=n_x * n_y)
            d1 = MultiArrayDimension(label='row_index', size=n_x, stride=n_x)
            arr.layout.dim = [d0, d1]
            arr.layout.data_offset = 0
            
            eigen_data = layer_data[::-1, ::-1].T.astype(np.float32)
            arr.data = eigen_data.flatten(order='F').tolist()
            msg.data.append(arr)

        msg.outer_start_index = 0
        msg.inner_start_index = 0
        self.gm_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = MapOverlayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
