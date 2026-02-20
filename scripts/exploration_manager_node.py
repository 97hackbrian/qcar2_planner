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

import math
import numpy as np

import rclpy
from rclpy.node import Node

from grid_map_msgs.msg import GridMap as GridMapMsg
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_srvs.srv import SetBool
from std_msgs.msg import Float32

import tf2_ros
from tf2_ros import Buffer, TransformListener


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
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'odom')
        self.declare_parameter('frontier_fov_deg', 140.0)
        self.declare_parameter('frontier_max_dist', 5.0)
        self.declare_parameter('spline_n_points', 4)
        self.declare_parameter('spline_wp_tolerance', 0.3)
        self.declare_parameter('spline_curvature', 0.33)

        # ── Read parameters ─────────────────────────────────────────────────
        self.tau = self.get_parameter('uncertainty_threshold').value
        self.wa_x_min = self.get_parameter('work_area_x_min').value
        self.wa_x_max = self.get_parameter('work_area_x_max').value
        self.wa_y_min = self.get_parameter('work_area_y_min').value
        self.wa_y_max = self.get_parameter('work_area_y_max').value
        self.frontier_min_size = self.get_parameter('frontier_min_size').value
        self.gradient_threshold = self.get_parameter('gradient_threshold').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.map_frame = self.get_parameter('map_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.frontier_fov_deg = self.get_parameter('frontier_fov_deg').value
        self.frontier_max_dist = self.get_parameter('frontier_max_dist').value
        self.spline_n_points = self.get_parameter('spline_n_points').value
        self.spline_wp_tolerance = self.get_parameter('spline_wp_tolerance').value
        self.spline_curvature = self.get_parameter('spline_curvature').value

        # ── State ───────────────────────────────────────────────────────────
        self.state = self.STATE_MAPPING
        self.latest_gridmap = None
        self.map_info = None

        # ── Spline navigation state ─────────────────────────────────────────
        self.spline_waypoints = []   # list of (x, y, yaw)
        self.current_wp_idx = 0      # index of current target waypoint
        self.planner_enabled = False

        # ── TF2 ─────────────────────────────────────────────────────────────
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── Subscribers ─────────────────────────────────────────────────────
        self.gridmap_sub = self.create_subscription(
            GridMapMsg, '/grid_map', self.gridmap_callback, 10
        )

        # ── Publishers ──────────────────────────────────────────────────────
        self.goal_pub = self.create_publisher(
            PoseStamped, '/exploration_goal', 10
        )
        self.marker_pub = self.create_publisher(
            MarkerArray, '/frontier_markers', 10
        )
        self.uncertainty_pub = self.create_publisher(
            Float32, '/exploration_uncertainty', 10
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

        # ── Publish uncertainty value ────────────────────────────────────
        unc_msg = Float32()
        unc_msg.data = u_bar
        self.uncertainty_pub.publish(unc_msg)

        self.get_logger().info(
            f'[{self.state}] Mean uncertainty: {u_bar:.3f} | '
            f'Mapped with certainty: {mapped_pct:.1f}% | '
            f'Threshold τ: {self.tau}'
        )

        # ── State logic ─────────────────────────────────────────────────
        if self.state == self.STATE_MAPPING:
            # If navigating along a spline, advance through waypoints
            if self.spline_waypoints:
                self._navigate_spline()

            # Always detect and publish frontier markers
            # Only generate NEW spline goals when mapped certainty ≥ τ (as %)
            # AND we have no active spline navigation
            should_publish = (mapped_pct >= self.tau * 100.0) and not self.spline_waypoints
            self._publish_frontier_goals(uncertainty, mask,
                                         publish_goal=should_publish)

            if u_bar <= self.tau:
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
    def _publish_frontier_goals(self, uncertainty: np.ndarray, free_mask: np.ndarray,
                                publish_goal: bool = False):
        """
        Detect frontiers and ALWAYS publish MarkerArray for visualization.
        Goal selection: pick the CLOSEST frontier within frontier_max_dist,
        preferring those inside the angular cone. If none in cone, fallback
        to the closest frontier overall (within max_dist).
        """
        import cv2

        # ── Get robot pose ───────────────────────────────────────────────
        robot_pose = self._get_robot_pose()
        half_fov = math.radians(self.frontier_fov_deg / 2.0)
        max_dist = self.frontier_max_dist

        if robot_pose is None:
            self.get_logger().warn(
                f'TF lookup {self.map_frame} → {self.base_frame} failed. '
                'Publishing goals without angular/distance filter.',
                throttle_duration_sec=5.0
            )

        # ── Gradient + frontier detection ───────────────────────────────
        grad_x = cv2.Sobel(uncertainty, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(uncertainty, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

        frontier = (grad_mag > self.gradient_threshold) & free_mask
        frontier_u8 = frontier.astype(np.uint8) * 255

        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            frontier_u8, connectivity=8
        )

        # ── Collect candidate frontiers with metadata ────────────────────
        candidates = []
        marker_array = MarkerArray()

        self.get_logger().warn(
            f'[DEBUG] n_labels={n_labels}, frontier_cells={int(np.sum(frontier))}, '
            f'frontier_min_size={self.frontier_min_size}, publish_goal={publish_goal}, '
            f'robot_pose={robot_pose is not None}',
            throttle_duration_sec=5.0
        )

        for label_id in range(1, n_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area < self.frontier_min_size:
                continue

            cx_px, cy_px = centroids[label_id]
            wx, wy = self._grid_to_world(cx_px, cy_px)

            # Compute distance and angle relative to robot
            if robot_pose is not None:
                rx, ry, ryaw = robot_pose
                dist = math.sqrt((wx - rx) ** 2 + (wy - ry) ** 2)
                angle_to = math.atan2(wy - ry, wx - rx)
                angle_diff = (angle_to - ryaw + math.pi) % (2.0 * math.pi) - math.pi
                in_cone = abs(angle_diff) <= half_fov
                within_dist = dist <= max_dist
            else:
                dist = 0.0
                angle_to = 0.0
                angle_diff = 0.0
                in_cone = True
                within_dist = True

            candidates.append({
                'wx': wx, 'wy': wy, 'dist': dist,
                'angle_to': angle_to,
                'angle_diff': angle_diff, 'in_cone': in_cone,
                'within_dist': within_dist, 'area': area, 'label_id': label_id
            })

            # Log EVERY candidate (no throttle)
            self.get_logger().warn(
                f'[CAND] id={label_id} pos=({wx:.2f},{wy:.2f}) '
                f'dist={dist:.2f}m angle={math.degrees(angle_diff):.1f}° '
                f'in_cone={in_cone} within_dist={within_dist} area={area}'
            )

            # ── ALWAYS add visualization marker ───────────────────────────
            marker = Marker()
            marker.header.stamp = self.get_clock().now().to_msg()
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
            # Green = in cone + within dist, Yellow = in cone but far,
            # Red = outside cone
            if in_cone and within_dist:
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            elif in_cone:
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            else:
                marker.color.r = 1.0
                marker.color.g = 0.2
                marker.color.b = 0.0
            marker.color.a = 0.8
            marker.lifetime.sec = 2
            marker_array.markers.append(marker)

        # ── Publish markers (always) ───────────────────────────────────
        if marker_array.markers:
            self.marker_pub.publish(marker_array)

        # ── Select best goal ──────────────────────────────────────────
        goal_published = False
        if publish_goal and candidates:
            # Only consider frontiers that are BOTH in cone AND within max_dist
            valid = [c for c in candidates if c['in_cone'] and c['within_dist']]

            self.get_logger().warn(
                f'[GOAL] valid={len(valid)} (in_cone+within_dist), '
                f'total={len(candidates)}'
            )

            if not valid:
                self.get_logger().warn('[GOAL] No frontier meets both angle+distance criteria')
            else:
                # Sort by distance → pick closest
                valid.sort(key=lambda c: c['dist'])
                best = valid[0]

                # Generate spline waypoints and start navigation
                rx, ry, ryaw = robot_pose
                self._generate_spline(rx, ry, ryaw,
                                      best['wx'], best['wy'], best['angle_to'])

                self.get_logger().warn(
                    f'[PUBLISHED] goal=({best["wx"]:.2f}, {best["wy"]:.2f}), '
                    f'dist={best["dist"]:.2f}m, '
                    f'angle={math.degrees(best["angle_diff"]):.1f}°, '
                    f'in_cone={best["in_cone"]}'
                )
        elif publish_goal:
            self.get_logger().warn('[GOAL] publish_goal=True but NO candidates!')
        else:
            self.get_logger().warn(
                f'[GOAL] publish_goal=False, skipping. candidates={len(candidates)}',
                throttle_duration_sec=5.0
            )

    # =====================================================================
    # Spline generation and sequential navigation
    # =====================================================================
    def _generate_spline(self, rx, ry, ryaw, gx, gy, gyaw):
        """
        Generate a cubic Bézier curve from robot to goal and store waypoints.
        Publishes the FIRST waypoint as /exploration_goal immediately.
        """
        dist = math.sqrt((gx - rx) ** 2 + (gy - ry) ** 2)
        if dist < 0.01:
            return

        d = dist * self.spline_curvature

        p0 = np.array([rx, ry])
        p1 = p0 + d * np.array([math.cos(ryaw), math.sin(ryaw)])
        p3 = np.array([gx, gy])
        p2 = p3 - d * np.array([math.cos(gyaw), math.sin(gyaw)])

        waypoints = []
        for i in range(self.spline_n_points + 1):
            t = i / float(self.spline_n_points)
            t1 = 1.0 - t

            pt = (t1**3 * p0 + 3*t1**2*t * p1 +
                  3*t1*t**2 * p2 + t**3 * p3)

            tangent = (3*t1**2 * (p1 - p0) + 6*t1*t * (p2 - p1) +
                       3*t**2 * (p3 - p2))
            yaw = math.atan2(tangent[1], tangent[0])

            waypoints.append((float(pt[0]), float(pt[1]), yaw))

        # Skip first point (robot's current position)
        self.spline_waypoints = waypoints[1:]
        self.current_wp_idx = 0

        # Publish first waypoint immediately
        self._publish_current_waypoint()

        self.get_logger().info(
            f'Spline generated: {len(self.spline_waypoints)} waypoints '
            f'to ({gx:.2f}, {gy:.2f})'
        )

    def _navigate_spline(self):
        """
        Check if robot reached current waypoint. If so, advance to next.
        Publishes spline waypoint markers every cycle.
        """
        if not self.spline_waypoints:
            return

        robot_pose = self._get_robot_pose()
        if robot_pose is None:
            return

        rx, ry, _ = robot_pose
        wx, wy, _ = self.spline_waypoints[self.current_wp_idx]

        dist_to_wp = math.sqrt((wx - rx) ** 2 + (wy - ry) ** 2)

        if dist_to_wp <= self.spline_wp_tolerance:
            # Reached current waypoint → advance
            self.current_wp_idx += 1

            if self.current_wp_idx >= len(self.spline_waypoints):
                # Completed all waypoints
                self.get_logger().info('Spline navigation complete!')
                self.spline_waypoints = []
                self.current_wp_idx = 0
                return

            self._publish_current_waypoint()
            self.get_logger().info(
                f'Waypoint {self.current_wp_idx}/{len(self.spline_waypoints)} '
                f'dist_to_prev={dist_to_wp:.2f}m'
            )

        # Publish markers for all spline waypoints
        self._publish_spline_markers()

    def _publish_current_waypoint(self):
        """Publish the current spline waypoint as /exploration_goal."""
        if self.current_wp_idx >= len(self.spline_waypoints):
            return

        wx, wy, wyaw = self.spline_waypoints[self.current_wp_idx]

        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = 'map'
        goal.pose.position.x = wx
        goal.pose.position.y = wy
        goal.pose.position.z = 0.0
        goal.pose.orientation.z = math.sin(wyaw / 2.0)
        goal.pose.orientation.w = math.cos(wyaw / 2.0)
        self.goal_pub.publish(goal)

    def _publish_spline_markers(self):
        """Publish all spline waypoints as ARROW markers in /frontier_markers."""
        marker_array = MarkerArray()

        for i, (wx, wy, wyaw) in enumerate(self.spline_waypoints):
            marker = Marker()
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.header.frame_id = 'map'
            marker.ns = 'spline'
            marker.id = 1000 + i
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.pose.position.x = wx
            marker.pose.position.y = wy
            marker.pose.position.z = 0.3
            marker.pose.orientation.z = math.sin(wyaw / 2.0)
            marker.pose.orientation.w = math.cos(wyaw / 2.0)
            marker.scale.x = 0.2   # arrow length
            marker.scale.y = 0.06  # arrow width
            marker.scale.z = 0.06  # arrow height

            if i < self.current_wp_idx:
                # Reached → dim green
                marker.color.r = 0.3
                marker.color.g = 0.7
                marker.color.b = 0.3
                marker.color.a = 0.4
            elif i == self.current_wp_idx:
                # Current target → bright cyan
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 1.0
                marker.color.a = 1.0
            else:
                # Pending → blue
                marker.color.r = 0.2
                marker.color.g = 0.4
                marker.color.b = 1.0
                marker.color.a = 0.6

            marker.lifetime.sec = 1
            marker_array.markers.append(marker)

        if marker_array.markers:
            self.marker_pub.publish(marker_array)

    # =====================================================================
    # Robot pose from TF (map → base_link)
    # =====================================================================
    def _get_robot_pose(self):
        """Get robot (x, y, yaw) in map frame via TF lookup."""
        try:
            t = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time()
            )
            x = t.transform.translation.x
            y = t.transform.translation.y
            q = t.transform.rotation
            yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y ** 2 + q.z ** 2),
            )
            return x, y, yaw
        except Exception:
            return None

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
