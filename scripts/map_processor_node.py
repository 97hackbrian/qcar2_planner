#!/usr/bin/env python3
# =============================================================================
# map_processor_node.py — Perception Engine (Node 1)
# =============================================================================
#
# Transforms nvblox 3D mesh data into a 2D multi-layer grid map.
#
# LAYERS:
#   occupancy  — binary free/occupied (0.0 = free, 1.0 = obstacle)
#   hits       — observation counter per cell
#   uncertainty — U(x,y) = 1.0 - min(1.0, hits(x,y) / N_threshold)
#   dir_x      — x-component of the legal direction unit vector
#   dir_y      — y-component of the legal direction unit vector
#
# PIPELINE:
#   1. Subscribe to /nvblox_node/mesh_marker (visualization_msgs/Marker)
#   2. Flatten: project Marker.points[] (x,y,z) → 2D grid cells
#   3. Morphological closure: cv2.morphologyEx(MORPH_CLOSE) on occupancy
#   4. Uncertainty: U = 1 - min(1, hits / N_threshold)
#   5. Direction: record robot TF heading into dir_x, dir_y
#   6. Publish grid_map_msgs/GridMap
#   7. Debug: cv2.imshow uncertainty heatmap
#
# GUARDRAIL: This node does NOT publish any motor commands.
# =============================================================================

import math
import numpy as np
import cv2
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from visualization_msgs.msg import Marker, MarkerArray
from grid_map_msgs.msg import GridMap as GridMapMsg, GridMapInfo
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, MultiArrayLayout, Header
from geometry_msgs.msg import Pose, Point, Quaternion

import tf2_ros
from tf2_ros import Buffer, TransformListener


class MapProcessorNode(Node):
    """Processes nvblox mesh_marker into a multi-layer 2D grid map."""

    def __init__(self):
        super().__init__('map_processor_node')

        # ── Declare parameters ──────────────────────────────────────────────
        self.declare_parameter('mesh_marker_topic', '/nvblox_node/mesh_marker')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('map_resolution', 0.05)
        self.declare_parameter('map_size_x', 20.0)
        self.declare_parameter('map_size_y', 20.0)
        self.declare_parameter('map_origin_x', 0.0)
        self.declare_parameter('map_origin_y', 0.0)
        self.declare_parameter('z_min', -0.5)
        self.declare_parameter('z_max', 2.0)
        self.declare_parameter('kernel_size', 5)
        self.declare_parameter('hits_threshold', 20)
        self.declare_parameter('publish_rate', 2.0)
        self.declare_parameter('show_debug_window', True)

        # ── Read parameters ─────────────────────────────────────────────────
        self.mesh_topic = self.get_parameter('mesh_marker_topic').value
        self.map_frame = self.get_parameter('map_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.resolution = self.get_parameter('map_resolution').value
        self.size_x = self.get_parameter('map_size_x').value
        self.size_y = self.get_parameter('map_size_y').value
        self.origin_x = self.get_parameter('map_origin_x').value
        self.origin_y = self.get_parameter('map_origin_y').value
        self.z_min = self.get_parameter('z_min').value
        self.z_max = self.get_parameter('z_max').value
        self.kernel_size = self.get_parameter('kernel_size').value
        self.hits_threshold = self.get_parameter('hits_threshold').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.show_debug = self.get_parameter('show_debug_window').value

        # ── Grid dimensions ─────────────────────────────────────────────────
        self.cols = int(self.size_x / self.resolution)   # x-axis cells
        self.rows = int(self.size_y / self.resolution)   # y-axis cells

        # ── Allocate numpy layers ───────────────────────────────────────────
        #   occupancy:   0.0 = unknown/free, 1.0 = obstacle
        #   hits:        observation counter
        #   uncertainty: 1.0 = fully uncertain, 0.0 = fully known
        #   dir_x/dir_y: legal direction unit vector
        self.occupancy = np.ones((self.rows, self.cols), dtype=np.float32)
        self.hits = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.uncertainty = np.ones((self.rows, self.cols), dtype=np.float32)
        self.dir_x = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.dir_y = np.zeros((self.rows, self.cols), dtype=np.float32)

        self.lock = threading.Lock()

        # ── TF2 ─────────────────────────────────────────────────────────────
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── Subscribers ─────────────────────────────────────────────────────
        qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )
        self.mesh_sub = self.create_subscription(
            MarkerArray, self.mesh_topic, self.mesh_callback, qos
        )

        # ── Publishers ──────────────────────────────────────────────────────
        self.gridmap_pub = self.create_publisher(GridMapMsg, '/grid_map', 10)

        # ── Timers ──────────────────────────────────────────────────────────
        period = 1.0 / self.publish_rate
        self.publish_timer = self.create_timer(period, self.publish_gridmap)
        self.direction_timer = self.create_timer(0.1, self.record_direction)

        if self.show_debug:
            self.debug_timer = self.create_timer(0.25, self.show_debug_window)

        self.get_logger().info(
            f'MapProcessorNode initialized: grid={self.cols}x{self.rows}, '
            f'res={self.resolution}m, topic={self.mesh_topic}'
        )

    # =====================================================================
    # World ↔ Grid coordinate conversions
    # =====================================================================
    def world_to_grid(self, wx: float, wy: float):
        """Convert world coordinates (meters) to grid indices (col, row)."""
        col = int((wx - self.origin_x + self.size_x / 2.0) / self.resolution)
        row = int((wy - self.origin_y + self.size_y / 2.0) / self.resolution)
        return col, row

    def in_bounds(self, col: int, row: int) -> bool:
        return 0 <= col < self.cols and 0 <= row < self.rows

    # =====================================================================
    # Mesh callback — Flattening (3D → 2D projection)
    # =====================================================================
    def mesh_callback(self, msg: MarkerArray):
        """
        Flatten the TRIANGLE_LIST mesh into the 2D occupancy grid.

        Iterates over every Marker in the MarkerArray. For each
        TRIANGLE_LIST or POINTS marker, projects vertices to (x, y) grid
        cells. Vertices within [z_min, z_max] mark their cells as FREE.
        This is the core 3D → 2D projection step.
        """
        count = 0
        with self.lock:
            for marker in msg.markers:
                # Only process mesh-type markers
                if marker.type != Marker.TRIANGLE_LIST and marker.type != Marker.POINTS:
                    continue

                for pt in marker.points:
                    # Filter by z-range for the flattening projection
                    if pt.z < self.z_min or pt.z > self.z_max:
                        continue

                    col, row = self.world_to_grid(pt.x, pt.y)
                    if not self.in_bounds(col, row):
                        continue

                    # Mark cell as FREE (observed road surface)
                    self.occupancy[row, col] = 0.0
                    self.hits[row, col] += 1.0
                    count += 1

            # ── Recompute uncertainty ───────────────────────────────────
            # U(x,y) = 1.0 - min(1.0, hits(x,y) / N_threshold)
            # A cell observed N_threshold times has U = 0 (fully certain).
            ratio = np.minimum(1.0, self.hits / float(self.hits_threshold))
            self.uncertainty = 1.0 - ratio

            # ── Morphological closure on occupancy ──────────────────────
            # Closure = Dilation ⊕ Erosion  →  (A ⊕ B) ⊖ B
            # Fills small gaps/holes in the detected road surface.
            k = self.kernel_size
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            # Invert: morphologyEx CLOSE fills dark holes in light regions.
            # Our occupancy: 0=free, 1=obstacle. We want to close gaps in
            # the FREE area, so we invert, close, and invert back.
            inv = 1.0 - self.occupancy
            closed = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel)
            self.occupancy = 1.0 - closed

        if count > 0:
            self.get_logger().info(
                f'Mesh callback: projected {count} vertices from '
                f'{len(msg.markers)} markers',
                throttle_duration_sec=2.0
            )

    # =====================================================================
    # Direction recording — stores robot heading in dir_x / dir_y layers
    # =====================================================================
    def record_direction(self):
        """
        Record the robot's current heading (unit vector) into the direction
        layers at the robot's current grid cell.

        This captures the LEGAL driving direction during the initial mapping
        phase. Cells accumulate direction samples; the stored vector is the
        most recent heading (instantaneous, during exploration).
        """
        try:
            t = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time()
            )
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return

        # Extract yaw from quaternion
        q = t.transform.rotation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        # Unit direction vector from yaw
        dx = math.cos(yaw)
        dy = math.sin(yaw)

        # Robot position → grid cell
        wx = t.transform.translation.x
        wy = t.transform.translation.y
        col, row = self.world_to_grid(wx, wy)

        if not self.in_bounds(col, row):
            return

        with self.lock:
            # Store the direction vector for this cell.
            # During mapping the robot drives in the LEGAL direction,
            # so we record it as the lane's legal heading.
            self.dir_x[row, col] = dx
            self.dir_y[row, col] = dy

            # Propagate direction to nearby cells in a small radius
            # to fill gaps between sparse TF updates.
            radius = 3  # cells
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    nr, nc = row + dr, col + dc
                    if self.in_bounds(nc, nr) and self.occupancy[nr, nc] < 0.5:
                        # Only write to free cells that have no direction yet
                        if self.dir_x[nr, nc] == 0.0 and self.dir_y[nr, nc] == 0.0:
                            self.dir_x[nr, nc] = dx
                            self.dir_y[nr, nc] = dy

    # =====================================================================
    # Publish grid_map_msgs/GridMap
    # =====================================================================
    def publish_gridmap(self):
        """
        Construct and publish a grid_map_msgs/GridMap message with all layers.

        Follows the grid_map_ros column-major convention:
          dim[0] = "column_index", size = cols, stride = rows * cols
          dim[1] = "row_index",    size = rows, stride = rows
          data   = flattened in Fortran (column-major) order
        This ensures compatibility with RViz grid_map plugin.
        """
        msg = GridMapMsg()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame

        # GridMapInfo
        msg.info.resolution = self.resolution
        msg.info.length_x = self.size_x
        msg.info.length_y = self.size_y
        msg.info.pose.position.x = self.origin_x
        msg.info.pose.position.y = self.origin_y
        msg.info.pose.position.z = 0.0
        msg.info.pose.orientation.w = 1.0

        # Layer names
        layer_names = ['occupancy', 'hits', 'uncertainty', 'dir_x', 'dir_y']
        msg.layers = layer_names
        msg.basic_layers = ['occupancy', 'uncertainty']

        with self.lock:
            layers_data = {
                'occupancy': self.occupancy.copy(),
                'hits': self.hits.copy(),
                'uncertainty': self.uncertainty.copy(),
                'dir_x': self.dir_x.copy(),
                'dir_y': self.dir_y.copy(),
            }

        # Serialize each layer following grid_map_ros column-major convention
        for name in layer_names:
            arr = Float32MultiArray()
            data = layers_data[name]

            # dim[0]: column_index — outer dimension
            dim0 = MultiArrayDimension()
            dim0.label = 'column_index'
            dim0.size = self.cols
            dim0.stride = self.rows * self.cols

            # dim[1]: row_index — inner dimension
            dim1 = MultiArrayDimension()
            dim1.label = 'row_index'
            dim1.size = self.rows
            dim1.stride = self.rows

            arr.layout.dim = [dim0, dim1]
            arr.layout.data_offset = 0
            # Column-major (Fortran) order matches Eigen's default storage
            arr.data = data.flatten(order='F').tolist()

            msg.data.append(arr)

        msg.outer_start_index = 0
        msg.inner_start_index = 0

        self.gridmap_pub.publish(msg)
        self.get_logger().debug('Published GridMap with all layers')

    # =====================================================================
    # Debug — OpenCV uncertainty heatmap
    # =====================================================================
    def show_debug_window(self):
        """
        Display the uncertainty layer as a color heatmap in an OpenCV window.
        Blue = certain (U ≈ 0), Red = uncertain (U ≈ 1).
        """
        with self.lock:
            u = self.uncertainty.copy()

        # Scale to 0-255 for colormap
        u8 = (u * 255.0).astype(np.uint8)
        heatmap = cv2.applyColorMap(u8, cv2.COLORMAP_JET)

        # Add text overlay
        mapped_pct = np.mean(u < 0.5) * 100.0
        cv2.putText(
            heatmap,
            f'Mapped: {mapped_pct:.1f}%',
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )

        cv2.imshow('Uncertainty Heatmap', heatmap)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = MapProcessorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
