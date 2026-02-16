#!/usr/bin/env python3
# =============================================================================
# exploration_manager_node.py — State Monitor (Node 2)
# =============================================================================
#
# Monitors the mapping progress by computing the mean uncertainty over the
# working area. Manages two states:
#
#   MAPPING:  U_bar > τ  →  publish PoseStamped goals at frontier centroids
#   READY:    U_bar ≤ τ  →  enable the directional planner via SetBool
#
# FRONTIER DETECTION:
#   Frontiers are regions with a high gradient of uncertainty, indicating the
#   boundary between mapped (certain) and unmapped (uncertain) areas. The node
#   clusters these frontier cells and publishes goals at their centroids.
#
# METRIC:
#   U_bar = (1/N) * Σ U_i  (mean uncertainty over track cells in working area)
#
# GUARDRAIL: This node does NOT publish any motor commands.
# =============================================================================

import numpy as np

import rclpy
from rclpy.node import Node

from grid_map_msgs.msg import GridMap as GridMapMsg
from geometry_msgs.msg import Pose, PoseStamped, PoseArray
from visualization_msgs.msg import Marker, MarkerArray
from std_srvs.srv import SetBool


class ExplorationManagerNode(Node):
    """Manages exploration state and frontier goal generation."""

    # States
    STATE_MAPPING = 'MAPPING'
    STATE_READY = 'READY'

    def __init__(self):
        super().__init__('exploration_manager_node')

        # ── Declare parameters ──────────────────────────────────────────────
        self.declare_parameter('uncertainty_threshold', 0.15)
        self.declare_parameter('work_area_x_min', -10.0)
        self.declare_parameter('work_area_x_max', 10.0)
        self.declare_parameter('work_area_y_min', -10.0)
        self.declare_parameter('work_area_y_max', 10.0)
        self.declare_parameter('frontier_min_size', 5)
        self.declare_parameter('gradient_threshold', 0.3)
        self.declare_parameter('publish_rate', 1.0)

        # ── Read parameters ─────────────────────────────────────────────────
        self.tau = self.get_parameter('uncertainty_threshold').value
        self.wa_x_min = self.get_parameter('work_area_x_min').value
        self.wa_x_max = self.get_parameter('work_area_x_max').value
        self.wa_y_min = self.get_parameter('work_area_y_min').value
        self.wa_y_max = self.get_parameter('work_area_y_max').value
        self.frontier_min_size = self.get_parameter('frontier_min_size').value
        self.gradient_threshold = self.get_parameter('gradient_threshold').value
        self.publish_rate = self.get_parameter('publish_rate').value

        # ── State ───────────────────────────────────────────────────────────
        self.state = self.STATE_MAPPING
        self.latest_gridmap = None
        self.map_info = None
        self.planner_enabled = False

        # ── Subscribers ─────────────────────────────────────────────────────
        self.gridmap_sub = self.create_subscription(
            GridMapMsg, '/grid_map', self.gridmap_callback, 10
        )

        # ── Publishers ──────────────────────────────────────────────────────
        self.goal_pub = self.create_publisher(
            PoseStamped, '/exploration_goal', 10
        )
        self.all_goals_pub = self.create_publisher(
            PoseArray, '/exploration_goals', 10
        )
        self.marker_pub = self.create_publisher(
            MarkerArray, '/frontier_markers', 10
        )

        # ── Service client for enabling the planner (Node 3) ────────────────
        self.enable_client = self.create_client(
            SetBool, '/enable_planner'
        )

        # ── Timer ───────────────────────────────────────────────────────────
        period = 1.0 / self.publish_rate
        self.timer = self.create_timer(period, self.evaluate)

        self.get_logger().info(
            f'ExplorationManagerNode initialized: τ={self.tau}, '
            f'work_area=[{self.wa_x_min},{self.wa_x_max}]×'
            f'[{self.wa_y_min},{self.wa_y_max}]'
        )

    # =====================================================================
    # GridMap callback — deserialize layers from grid_map_msgs/GridMap
    # =====================================================================
    def gridmap_callback(self, msg: GridMapMsg):
        """
        Parse the incoming GridMap message and store the numpy layers.
        Uses column-major (Fortran) order matching grid_map_ros convention.
        """
        self.map_info = msg.info

        # Build a dict of layer_name → numpy array
        layers = {}
        rows = 0
        cols = 0

        for i, name in enumerate(msg.layers):
            arr = msg.data[i]

            # grid_map convention: dim[0] = column_index (Eigen cols = n_y)
            #                      dim[1] = row_index   (Eigen rows = n_x)
            if len(arr.layout.dim) >= 2:
                n_y = arr.layout.dim[0].size  # Eigen cols = cells along Y
                n_x = arr.layout.dim[1].size  # Eigen rows = cells along X
            else:
                # Fallback: use map info
                if self.map_info is not None:
                    n_x = int(self.map_info.length_x / self.map_info.resolution)
                    n_y = int(self.map_info.length_y / self.map_info.resolution)
                else:
                    continue

            # Deserialize column-major → Eigen shape (n_x, n_y)
            eigen_data = np.array(arr.data, dtype=np.float32).reshape(
                (n_x, n_y), order='F'
            )
            # Convert Eigen → our numpy: (n_x, n_y) → (n_y, n_x)
            # Eigen row0=maxX, col0=maxY → numpy row0=minY, col0=minX
            data = eigen_data.T[::-1, ::-1]
            layers[name] = data

        self.latest_gridmap = layers
        if layers:
            sample = next(iter(layers.values()))
            self.grid_rows, self.grid_cols = sample.shape  # (n_y, n_x)

    # =====================================================================
    # Periodic evaluation — state machine
    # =====================================================================
    def evaluate(self):
        """
        Main logic loop:
        1. Compute mean uncertainty over the working area (track cells only)
        2. If MAPPING and U_bar > τ  → find & publish frontier goals
        3. If MAPPING and U_bar ≤ τ  → switch to READY, enable planner
        """
        if self.latest_gridmap is None or 'uncertainty' not in self.latest_gridmap:
            self.get_logger().info(
                'Waiting for GridMap data...',
                throttle_duration_sec=5.0
            )
            return

        uncertainty = self.latest_gridmap['uncertainty']
        occupancy = self.latest_gridmap.get('occupancy', None)

        # ── Extract working area cells ──────────────────────────────────
        mask = self._working_area_mask(uncertainty)

        # Only consider FREE cells (occupancy == 0.0) for the metric
        # New convention: -1=unknown, 0=free, 1=wall
        if occupancy is not None:
            free_mask = occupancy == 0.0
            mask = mask & free_mask

        n_cells = np.sum(mask)
        if n_cells == 0:
            self.get_logger().warn(
                'No free cells in working area yet',
                throttle_duration_sec=5.0
            )
            return

        # ── Mean uncertainty: U_bar = (1/N) * Σ U_i ────────────────────
        u_values = uncertainty[mask]
        u_bar = float(np.mean(u_values))
        mapped_pct = float(np.mean(u_values < 0.5)) * 100.0

        self.get_logger().info(
            f'[{self.state}] Mean uncertainty: {u_bar:.3f} | '
            f'Mapped with certainty: {mapped_pct:.1f}% | '
            f'Threshold τ: {self.tau}'
        )

        # ── State logic ─────────────────────────────────────────────────
        if self.state == self.STATE_MAPPING:
            if u_bar > self.tau:
                # Still mapping — find and publish frontier goals
                self._publish_frontier_goals(uncertainty, mask)
            else:
                # Map is complete! Switch to READY
                self.state = self.STATE_READY
                self.get_logger().info(
                    '═══════════════════════════════════════════════\n'
                    '  ✅ MAP COMPLETE — Switching to READY state\n'
                    f'  Final mean uncertainty: {u_bar:.4f} ≤ {self.tau}\n'
                    '  Enabling directional planner service...\n'
                    '═══════════════════════════════════════════════'
                )
                self._enable_planner()

        elif self.state == self.STATE_READY:
            self.get_logger().debug(
                f'READY — Planner enabled. Map certainty: {mapped_pct:.1f}%'
            )

    # =====================================================================
    # Working area mask
    # =====================================================================
    def _working_area_mask(self, layer: np.ndarray) -> np.ndarray:
        """
        Create a boolean mask for cells within the working area quadrilateral
        defined by lidar range bounds.
        """
        if self.map_info is None:
            return np.ones(layer.shape, dtype=bool)

        res = self.map_info.resolution
        # Corner = centre - half-length (matches map_processor_node convention)
        corner_x = self.map_info.pose.position.x - self.map_info.length_x / 2.0
        corner_y = self.map_info.pose.position.y - self.map_info.length_y / 2.0

        rows, cols = layer.shape
        col_coords = corner_x + (np.arange(cols) + 0.5) * res
        row_coords = corner_y + (np.arange(rows) + 0.5) * res

        col_mask = (col_coords >= self.wa_x_min) & (col_coords <= self.wa_x_max)
        row_mask = (row_coords >= self.wa_y_min) & (row_coords <= self.wa_y_max)

        return np.outer(row_mask, col_mask)

    # =====================================================================
    # Frontier detection and goal publication
    # =====================================================================
    def _publish_frontier_goals(self, uncertainty: np.ndarray, free_mask: np.ndarray):
        """
        Detect frontiers: cells with HIGH gradient of uncertainty, meaning
        they are at the boundary between mapped and unmapped areas.

        Steps:
          1. Compute gradient magnitude of the uncertainty layer
          2. Threshold to get frontier cells
          3. Cluster frontier cells (connected components)
          4. Sort clusters by size (largest first)
          5. Publish the largest as primary PoseStamped on /exploration_goal
          6. Publish ALL frontier centroids as PoseArray on /exploration_goals
          7. Visualize with MarkerArray
        """
        import cv2

        # Gradient magnitude (Sobel)
        grad_x = cv2.Sobel(uncertainty, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(uncertainty, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # Frontier: high gradient AND free cells in working area
        frontier = (grad_mag > self.gradient_threshold) & free_mask
        frontier_u8 = frontier.astype(np.uint8) * 255

        # Connected components
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            frontier_u8, connectivity=8
        )

        # Collect valid frontier clusters sorted by size (largest first)
        clusters = []
        for label_id in range(1, n_labels):  # skip background (0)
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area < self.frontier_min_size:
                continue
            cx_px, cy_px = centroids[label_id]
            wx, wy = self._grid_to_world(cx_px, cy_px)
            clusters.append((area, label_id, wx, wy))

        clusters.sort(key=lambda c: c[0], reverse=True)

        if not clusters:
            return

        stamp = self.get_clock().now().to_msg()

        # Publish primary goal (largest frontier)
        best_area, _, best_wx, best_wy = clusters[0]
        goal = PoseStamped()
        goal.header.stamp = stamp
        goal.header.frame_id = 'map'
        goal.pose.position.x = best_wx
        goal.pose.position.y = best_wy
        goal.pose.position.z = 0.0
        goal.pose.orientation.w = 1.0
        self.goal_pub.publish(goal)

        self.get_logger().info(
            f'Published primary frontier goal: ({best_wx:.2f}, {best_wy:.2f}), '
            f'cluster size={best_area} cells, total frontiers={len(clusters)}'
        )

        # Publish ALL frontier centroids as PoseArray
        all_goals = PoseArray()
        all_goals.header.stamp = stamp
        all_goals.header.frame_id = 'map'
        for _, _, wx, wy in clusters:
            p = Pose()
            p.position.x = wx
            p.position.y = wy
            p.position.z = 0.0
            p.orientation.w = 1.0
            all_goals.poses.append(p)
        self.all_goals_pub.publish(all_goals)

        # Visualization markers
        marker_array = MarkerArray()
        for idx, (area, label_id, wx, wy) in enumerate(clusters):
            marker = Marker()
            marker.header.stamp = stamp
            marker.header.frame_id = 'map'
            marker.ns = 'frontiers'
            marker.id = label_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = wx
            marker.pose.position.y = wy
            marker.pose.position.z = 0.5
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.3
            # Primary frontier is green, others are orange
            if idx == 0:
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            else:
                marker.color.r = 1.0
                marker.color.g = 0.5
                marker.color.b = 0.0
            marker.color.a = 0.8
            marker.lifetime.sec = 2
            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)

    # =====================================================================
    # Grid ↔ World conversion (using map_info)
    # =====================================================================
    def _grid_to_world(self, col: float, row: float):
        """Convert grid indices to world coordinates (cell centre)."""
        if self.map_info is None:
            return 0.0, 0.0

        res = self.map_info.resolution
        # Corner = centre - half-length (matches map_processor_node convention)
        corner_x = self.map_info.pose.position.x - self.map_info.length_x / 2.0
        corner_y = self.map_info.pose.position.y - self.map_info.length_y / 2.0

        wx = corner_x + (col + 0.5) * res
        wy = corner_y + (row + 0.5) * res
        return float(wx), float(wy)

    # =====================================================================
    # Enable planner — call SetBool service on Node 3
    # =====================================================================
    def _enable_planner(self):
        """Send a SetBool(True) request to /enable_planner on Node 3."""
        if not self.enable_client.service_is_ready():
            self.get_logger().warn(
                'Service /enable_planner not available, retrying in 2s...'
            )
            # One-shot retry timer (stored to prevent accumulation)
            if hasattr(self, '_retry_timer') and self._retry_timer is not None:
                self._retry_timer.cancel()
            self._retry_timer = self.create_timer(2.0, self._enable_planner_retry)
            return

        request = SetBool.Request()
        request.data = True
        future = self.enable_client.call_async(request)
        future.add_done_callback(self._enable_callback)

    def _enable_planner_retry(self):
        """Retry enabling the planner (one-shot: cancels its own timer)."""
        if hasattr(self, '_retry_timer') and self._retry_timer is not None:
            self._retry_timer.cancel()
            self._retry_timer = None
        self._enable_planner()

    def _enable_callback(self, future):
        try:
            result = future.result()
            if result.success:
                self.planner_enabled = True
                self.get_logger().info(
                    f'Planner enabled successfully: {result.message}'
                )
            else:
                self.get_logger().error(
                    f'Failed to enable planner: {result.message}'
                )
        except Exception as e:
            self.get_logger().error(f'Enable planner service call failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = ExplorationManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
