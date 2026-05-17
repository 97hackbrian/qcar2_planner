#!/usr/bin/env python3
# =============================================================================
# map_loader_node.py — Saved Map Loader (Node 4)
# =============================================================================
#
# Loads a previously saved 2D uncertainty map (PGM + YAML from
# nav2_map_server format), applies morphological opening + closing,
# and publishes it on:
#   /planner_uncertainty  (OccupancyGrid)  — for RViz2 visualization
#   /grid_map             (GridMap)         — for directional_planner_server
#
# This node replaces map_processor_node when running in saved-map mode.
#
# GUARDRAIL: This node does NOT publish any motor commands.
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
from std_msgs.msg import Float32MultiArray, MultiArrayDimension


class MapLoaderNode(Node):
    """Loads a saved PGM+YAML map, post-processes, and publishes."""

    def __init__(self):
        super().__init__('map_loader_node')
        self.get_logger().info('=== MapLoaderNode __init__ START ===')

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter('map_yaml_path', '/workspaces/isaac_ros-dev/ros2/utils/mapV4uncertainty.yaml')
        self.declare_parameter('morph_kernel_size', 0)
        self.declare_parameter('publish_rate', 1.0)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('uncertainty_out_topic', '/planner_uncertainty')
        self.declare_parameter('gridmap_out_topic', '/grid_map')
        self.declare_parameter('lanes_yaml_path', '')

        self.map_yaml_path = str(self.get_parameter('map_yaml_path').value)
        self.morph_kernel_size = int(self.get_parameter('morph_kernel_size').value)
        self.publish_rate = float(self.get_parameter('publish_rate').value)
        self.map_frame = str(self.get_parameter('map_frame').value)
        unc_topic = str(self.get_parameter('uncertainty_out_topic').value)
        gm_topic = str(self.get_parameter('gridmap_out_topic').value)

        # ── Validate path ───────────────────────────────────────────────────
        if not self.map_yaml_path:
            self.get_logger().fatal('map_yaml_path is EMPTY. Cannot load map.')
            raise RuntimeError('map_yaml_path parameter is required.')

        if not os.path.isfile(self.map_yaml_path):
            self.get_logger().fatal(f'Map YAML not found: {self.map_yaml_path}')
            raise FileNotFoundError(f'{self.map_yaml_path}')

        # ── Load map ────────────────────────────────────────────────────────
        self.get_logger().info(f'Loading map from: {self.map_yaml_path}')
        self._load_map()
        self._load_lanes()

        # ── Publishers ──────────────────────────────────────────────────────
        # Transient-local so late subscribers get the last message
        qos_latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.unc_pub = self.create_publisher(
            OccupancyGrid, unc_topic, qos_latched
        )
        self.gm_pub = self.create_publisher(
            GridMapMsg, gm_topic, 10
        )

        # ── Timer ───────────────────────────────────────────────────────────
        period = 1.0 / max(0.1, self.publish_rate)
        self.pub_timer = self.create_timer(period, self._publish_all)

        self.get_logger().info(
            f'MapLoaderNode ready: {self.cols}x{self.rows} cells, '
            f'res={self.resolution:.3f}m, publishing at {self.publish_rate}Hz'
        )

    # =====================================================================
    # Load PGM + YAML
    # =====================================================================
    def _load_map(self):
        """Load the YAML metadata and PGM image, apply morphological ops."""
        yaml_dir = os.path.dirname(os.path.abspath(self.map_yaml_path))

        with open(self.map_yaml_path, 'r') as f:
            meta = yaml.safe_load(f)

        # ── YAML fields ─────────────────────────────────────────────────
        pgm_name = meta.get('image', '')
        self.resolution = float(meta.get('resolution', 0.05))
        origin = meta.get('origin', [0.0, 0.0, 0.0])
        self.origin_x = float(origin[0])
        self.origin_y = float(origin[1])
        negate = int(meta.get('negate', 0))
        occ_thresh = float(meta.get('occupied_thresh', 0.65))
        free_thresh = float(meta.get('free_thresh', 0.196))

        # ── Load PGM ────────────────────────────────────────────────────
        pgm_path = os.path.join(yaml_dir, pgm_name)
        if not os.path.isfile(pgm_path):
            raise FileNotFoundError(f'PGM not found: {pgm_path}')

        raw_img = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)
        if raw_img is None:
            raise RuntimeError(f'Failed to read PGM: {pgm_path}')

        self.get_logger().info(
            f'PGM loaded: {raw_img.shape[1]}x{raw_img.shape[0]}, '
            f'resolution={self.resolution}m'
        )

        # ── Normalize to occupancy [0, 1] — match nav2_map_server ──────
        # nav2 convention: occ = (255 - pixel) / 255  when negate=0
        #   white pixel (254) → occ ≈ 0.004 → FREE
        #   black pixel (0)   → occ = 1.0   → OCCUPIED
        #   gray  pixel (205) → occ ≈ 0.196 → UNKNOWN
        if negate:
            img_f = raw_img.astype(np.float32) / 255.0
        else:
            img_f = (255.0 - raw_img.astype(np.float32)) / 255.0

        # nav2_map_server convention: image is stored top-down,
        # but OccupancyGrid row 0 = bottom. Flip vertically.
        img_f = np.flipud(img_f)

        # ── Morphological opening + closing ─────────────────────────────
        k = self.morph_kernel_size
        if k > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            img_f = cv2.morphologyEx(img_f, cv2.MORPH_OPEN, kernel)
            img_f = cv2.morphologyEx(img_f, cv2.MORPH_CLOSE, kernel)
            self.get_logger().info(
                f'Morphological opening+closing applied (kernel={k}x{k})'
            )

        # ── Classify cells ──────────────────────────────────────────────
        # nav2 convention: pixel value p maps to occupancy:
        #   p >= occ_thresh  → occupied (wall)
        #   p <= free_thresh → free
        #   otherwise        → unknown (-1)
        self.rows, self.cols = img_f.shape
        self.occupancy = np.full((self.rows, self.cols), -1.0, dtype=np.float32)
        self.occupancy[img_f <= free_thresh] = 0.0    # free (explored)
        self.occupancy[img_f >= occ_thresh] = 1.0     # wall

        # Uncertainty: use the raw image value directly for free cells
        # free_thresh maps to uncertainty = 0 (fully explored)
        # Closer to occ_thresh = higher uncertainty
        self.uncertainty = np.ones((self.rows, self.cols), dtype=np.float32)
        free_mask = self.occupancy == 0.0
        # Scale: 0.0 = fully certain (white), 1.0 = uncertain (gray)
        self.uncertainty[free_mask] = np.clip(
            img_f[free_mask] / max(free_thresh, 0.01), 0.0, 1.0
        )

        # No direction info from saved map (will be populated by lanes if present)
        self.lane_mask = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.dir_x = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.dir_y = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.independent_lanes = {}

        # Grid geometry for GridMap (center-based)
        self.size_x = self.cols * self.resolution
        self.size_y = self.rows * self.resolution
        self.center_x = self.origin_x + self.size_x / 2.0
        self.center_y = self.origin_y + self.size_y / 2.0

        n_free = int(np.sum(free_mask))
        n_wall = int(np.sum(self.occupancy == 1.0))
        n_unk = int(np.sum(self.occupancy == -1.0))
        self.get_logger().info(
            f'Map classified: free={n_free}, wall={n_wall}, unknown={n_unk}'
        )

    def _load_lanes(self):
        lanes_yaml_path = str(self.get_parameter('lanes_yaml_path').value)
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
                
            width_px = width / self.resolution
            
            lane_mask = np.zeros((self.rows, self.cols), dtype=np.float32)
            lane_dir_x = np.zeros((self.rows, self.cols), dtype=np.float32)
            lane_dir_y = np.zeros((self.rows, self.cols), dtype=np.float32)
            
            for i in range(len(points) - 1):
                p1 = points[i]
                p2 = points[i+1]
                
                c1 = int((p1[0] - self.origin_x) / self.resolution - 0.5)
                r1 = int((p1[1] - self.origin_y) / self.resolution - 0.5)
                c2 = int((p2[0] - self.origin_x) / self.resolution - 0.5)
                r2 = int((p2[1] - self.origin_y) / self.resolution - 0.5)
                
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
            self.dir_x[mask_indices] = lane_dir_x[mask_indices]
            self.dir_y[mask_indices] = lane_dir_y[mask_indices]
                
        n_mask = int(np.sum(self.lane_mask > 0))
        self.get_logger().info(f'Total lane_mask cells: {n_mask}')

    # =====================================================================
    # Publish all outputs
    # =====================================================================
    def _publish_all(self):
        """Publish OccupancyGrid and GridMap."""
        try:
            stamp = self.get_clock().now().to_msg()
            self._pub_uncertainty(stamp)
            self._pub_gridmap(stamp)
        except Exception as e:
            self.get_logger().error(
                f'Publish error: {e}\n{traceback.format_exc()}'
            )

    # ── OccupancyGrid: uncertainty ──────────────────────────────────────
    def _pub_uncertainty(self, stamp):
        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = self.map_frame
        msg.info.resolution = float(self.resolution)
        msg.info.width = int(self.cols)
        msg.info.height = int(self.rows)
        msg.info.origin.position.x = float(self.origin_x)
        msg.info.origin.position.y = float(self.origin_y)
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0

        # RViz OccupancyGrid convention:
        #   -1  = unknown (gray)
        #    0  = free    (white) — navigable
        #   100 = wall    (black) — obstacle
        grid = np.full(self.rows * self.cols, -1, dtype=np.int8)
        flat_occ = self.occupancy.ravel()
        flat_unc = self.uncertainty.ravel()

        # Free cells: 0 (white)
        free_mask = flat_occ == 0.0
        grid[free_mask] = np.clip(
            flat_unc[free_mask] * 100.0, 0, 100
        ).astype(np.int8)

        # Wall cells: 100 (black)
        wall_mask = flat_occ == 1.0
        grid[wall_mask] = 100

        msg.data = grid.tolist()
        self.unc_pub.publish(msg)

    # ── GridMap: multi-layer for planner ────────────────────────────────
    def _pub_gridmap(self, stamp):
        msg = GridMapMsg()
        msg.header.stamp = stamp
        msg.header.frame_id = self.map_frame

        msg.info.resolution = float(self.resolution)
        msg.info.length_x = float(self.size_x)
        msg.info.length_y = float(self.size_y)
        msg.info.pose.position.x = float(self.center_x)
        msg.info.pose.position.y = float(self.center_y)
        msg.info.pose.position.z = 0.0
        msg.info.pose.orientation.w = 1.0

        names = ['occupancy', 'uncertainty', 'lane_mask', 'dir_x', 'dir_y']
        arrays = [self.occupancy, self.uncertainty, self.lane_mask, self.dir_x, self.dir_y]
        msg.layers = names
        msg.basic_layers = ['occupancy', 'uncertainty']

        n_x = self.cols   # cells along X
        n_y = self.rows   # cells along Y

        for layer_data in arrays:
            arr = Float32MultiArray()
            # grid_map Eigen convention:
            #   dim[0] = column_index = Eigen cols = n_y
            #   dim[1] = row_index   = Eigen rows = n_x
            d0 = MultiArrayDimension()
            d0.label = 'column_index'
            d0.size = n_y
            d0.stride = n_x * n_y
            d1 = MultiArrayDimension()
            d1.label = 'row_index'
            d1.size = n_x
            d1.stride = n_x
            arr.layout.dim = [d0, d1]
            arr.layout.data_offset = 0
            # Transform: numpy (n_y, n_x) → Eigen (n_x, n_y)
            eigen_data = layer_data[::-1, ::-1].T.astype(np.float32)
            arr.data = eigen_data.flatten(order='F').tolist()
            msg.data.append(arr)

        msg.outer_start_index = 0
        msg.inner_start_index = 0
        self.gm_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    try:
        node = MapLoaderNode()
    except Exception as e:
        print(f'MapLoaderNode failed to start: {e}')
        rclpy.shutdown()
        return
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
