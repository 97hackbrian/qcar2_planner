#!/usr/bin/env python3
# =============================================================================
# exploration_manager_node.py — State Monitor (Node 2)
# =============================================================================
#
# Monitors mapping progress by computing uncertainty over the working area.
# Manages two states:
#
#   MAPPING:  U_bar > τ  →  publish PoseStamped goals at frontier centroids
#   READY:    U_bar ≤ τ  →  enable the directional planner via SetBool
#
# INTEREST-POINT DETECTION:
#   Frontiers are detected as FREE cells adjacent to UNKNOWN cells.
#   Then candidates are filtered to keep intersection-like points (multiple
#   unknown angular sectors around the candidate), improving path discovery.
#   The node publishes goals and RViz markers on these points.
#
# METRIC:
#   U_bar = (1/N) * Σ U_i  (mean uncertainty over track cells in working area)
#   Plus standardized topic /exploration_metrics with mapping progress.
#
# GUARDRAIL: This node does NOT publish any motor commands.
# =============================================================================

import numpy as np
import cv2

import rclpy
from rclpy.node import Node

from grid_map_msgs.msg import GridMap as GridMapMsg
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from std_srvs.srv import SetBool
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
        self.declare_parameter('certainty_threshold', 0.35)
        self.declare_parameter('mapping_completion_threshold', 0.95)
        self.declare_parameter('frontier_unknown_dilate_cells', 1)
        self.declare_parameter('frontier_clearance_cells', 1)
        self.declare_parameter('frontier_connectivity', 8)
        self.declare_parameter('frontier_uncertainty_threshold', 0.80)
        self.declare_parameter('frontier_fallback_to_map_unknown', False)
        self.declare_parameter('reachable_inflation_cells', 1)
        self.declare_parameter('interest_radius_cells', 10)
        self.declare_parameter('min_unknown_sectors', 2)
        self.declare_parameter('max_interest_points', 12)
        self.declare_parameter('goal_republish_distance', 0.7)
        self.declare_parameter('goal_republish_period_sec', 2.0)
        self.declare_parameter('goal_switch_margin_m', 0.6)
        self.declare_parameter('local_frontier_radius_m', 2.5)
        self.declare_parameter('center_attractor_weight', 0.35)
        self.declare_parameter('enable_penetration_points', True)
        self.declare_parameter('penetration_ray_count', 72)
        self.declare_parameter('penetration_ray_step_cells', 1)
        self.declare_parameter('penetration_min_free_run', 4)
        self.declare_parameter('penetration_boost_score', 50.0)
        self.declare_parameter('min_unknown_blob_cells', 200)
        # ── Trajectory road-mask & gap detection ────────────────────────
        self.declare_parameter('enable_trajectory_road_mask', True)
        self.declare_parameter('trajectory_topic', '/trajectory_node_list')
        self.declare_parameter('constraint_topic', '/constraint_list')
        self.declare_parameter('road_mask_width_cells', 3)
        self.declare_parameter('road_gap_min_opening_cells', 6)
        self.declare_parameter('road_gap_boost_score', 80.0)
        # ── Persistent point bank & reveal radius ───────────────────
        self.declare_parameter('reveal_radius_m', 4.0)
        self.declare_parameter('point_bank_cell_size_m', 0.15)
        self.declare_parameter('point_bank_max_size', 2000)
        # ── Hotspot (entrance) detection ────────────────────────────
        self.declare_parameter('hotspot_radius_m', 1.5)
        self.declare_parameter('hotspot_min_points', 3)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'odom')
        self.declare_parameter('publish_rate', 1.0)

        # ── Read parameters ─────────────────────────────────────────────────
        self.tau = self.get_parameter('uncertainty_threshold').value
        self.wa_x_min = self.get_parameter('work_area_x_min').value
        self.wa_x_max = self.get_parameter('work_area_x_max').value
        self.wa_y_min = self.get_parameter('work_area_y_min').value
        self.wa_y_max = self.get_parameter('work_area_y_max').value
        self.frontier_min_size = self.get_parameter('frontier_min_size').value
        self.certainty_threshold = self.get_parameter('certainty_threshold').value
        self.mapping_completion_threshold = self.get_parameter(
            'mapping_completion_threshold'
        ).value
        self.frontier_unknown_dilate_cells = self.get_parameter(
            'frontier_unknown_dilate_cells'
        ).value
        self.frontier_clearance_cells = self.get_parameter(
            'frontier_clearance_cells'
        ).value
        self.frontier_connectivity = int(self.get_parameter('frontier_connectivity').value)
        self.frontier_uncertainty_threshold = float(
            self.get_parameter('frontier_uncertainty_threshold').value
        )
        self.frontier_fallback_to_map_unknown = bool(
            self.get_parameter('frontier_fallback_to_map_unknown').value
        )
        self.reachable_inflation_cells = int(self.get_parameter('reachable_inflation_cells').value)
        self.interest_radius_cells = self.get_parameter('interest_radius_cells').value
        self.min_unknown_sectors = self.get_parameter('min_unknown_sectors').value
        self.max_interest_points = self.get_parameter('max_interest_points').value
        self.goal_republish_distance = self.get_parameter(
            'goal_republish_distance'
        ).value
        self.goal_republish_period_sec = self.get_parameter(
            'goal_republish_period_sec'
        ).value
        self.goal_switch_margin_m = self.get_parameter('goal_switch_margin_m').value
        self.local_frontier_radius_m = float(
            self.get_parameter('local_frontier_radius_m').value
        )
        self.center_attractor_weight = float(
            self.get_parameter('center_attractor_weight').value
        )
        self.enable_penetration_points = bool(
            self.get_parameter('enable_penetration_points').value
        )
        self.penetration_ray_count = int(
            self.get_parameter('penetration_ray_count').value
        )
        self.penetration_ray_step_cells = int(
            self.get_parameter('penetration_ray_step_cells').value
        )
        self.penetration_min_free_run = int(
            self.get_parameter('penetration_min_free_run').value
        )
        self.penetration_boost_score = float(
            self.get_parameter('penetration_boost_score').value
        )
        self.min_unknown_blob_cells = int(
            self.get_parameter('min_unknown_blob_cells').value
        )
        self.enable_trajectory_road_mask = bool(
            self.get_parameter('enable_trajectory_road_mask').value
        )
        self.trajectory_topic = str(
            self.get_parameter('trajectory_topic').value
        )
        self.constraint_topic = str(
            self.get_parameter('constraint_topic').value
        )
        self.road_mask_width_cells = int(
            self.get_parameter('road_mask_width_cells').value
        )
        self.road_gap_min_opening_cells = int(
            self.get_parameter('road_gap_min_opening_cells').value
        )
        self.road_gap_boost_score = float(
            self.get_parameter('road_gap_boost_score').value
        )
        self.reveal_radius_m = float(
            self.get_parameter('reveal_radius_m').value
        )
        self.point_bank_cell_size_m = float(
            self.get_parameter('point_bank_cell_size_m').value
        )
        self.point_bank_max_size = int(
            self.get_parameter('point_bank_max_size').value
        )
        self.hotspot_radius_m = float(
            self.get_parameter('hotspot_radius_m').value
        )
        self.hotspot_min_points = int(
            self.get_parameter('hotspot_min_points').value
        )
        self.map_frame = self.get_parameter('map_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.publish_rate = self.get_parameter('publish_rate').value

        # ── State ───────────────────────────────────────────────────────────
        self.state = self.STATE_MAPPING
        self.latest_gridmap = None
        self.map_info = None
        self.planner_enabled = False
        self._work_mask_cache = None
        self._work_mask_signature = None
        self._last_goal_xy = None
        self._last_goal_time_ns = 0

        # Trajectory road mask state
        self._trajectory_points = []      # list of (x,y) world coords
        self._constraint_points = []      # inter-constraint node positions
        self._road_mask = None            # uint8 grid rasterized from trajectory
        self._road_gaps = []              # detected gap centroids [(wx,wy),...]
        self._road_mask_stamp = 0         # last update time

        # Persistent point bank: spatial-key → point dict
        # Key = (int(wx / cell_size), int(wy / cell_size)) for dedup
        # Points are REMOVED when the robot passes nearby (explored).
        self._point_bank = {}             # {(kx,ky): point_dict}
        self._bank_marker_count = 0       # last published marker count (for cleanup)
        self._robot_travel = 0.0          # cumulative distance for startup guard
        self._prev_robot_xy = None        # previous pose for travel accumulation
        self._hotspot_key = None          # spatial key of current densest cluster centroid

        # ── TF2 ─────────────────────────────────────────────────────────────
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── Subscribers ─────────────────────────────────────────────────────
        self.gridmap_sub = self.create_subscription(
            GridMapMsg, '/grid_map', self.gridmap_callback, 10
        )

        # Trajectory + constraint from Cartographer
        if self.enable_trajectory_road_mask:
            self.traj_sub = self.create_subscription(
                MarkerArray, self.trajectory_topic,
                self._trajectory_callback, 10
            )
            self.constraint_sub = self.create_subscription(
                MarkerArray, self.constraint_topic,
                self._constraint_callback, 10
            )
            self.get_logger().info(
                f'Road-mask enabled: traj={self.trajectory_topic}, '
                f'constr={self.constraint_topic}'
            )

        # ── Publishers ──────────────────────────────────────────────────────
        self.goal_pub = self.create_publisher(
            PoseStamped, '/exploration_goal', 10
        )
        self.marker_pub = self.create_publisher(
            MarkerArray, '/frontier_markers', 10
        )
        self.metrics_pub = self.create_publisher(
            Float32MultiArray, '/exploration_metrics', 10
        )
        self.centroid_marker_pub = self.create_publisher(
            Marker, '/unknown_centroid_marker', 10
        )
        self.road_gap_marker_pub = self.create_publisher(
            MarkerArray, '/road_gap_markers', 10
        )
        self.reveal_ring_pub = self.create_publisher(
            Marker, '/reveal_radius_marker', 10
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
        self._invalidate_work_area_cache_if_needed()

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
    # Trajectory node list callback — Cartographer trajectory polyline
    # =====================================================================
    def _trajectory_callback(self, msg: MarkerArray):
        """
        /trajectory_node_list is a MarkerArray with LINE_STRIP markers.
        Each marker.points[] contains the trajectory nodes the robot visited.
        We collect all (x,y) points to rasterize the road boundary.
        """
        pts = []
        for marker in msg.markers:
            if marker.type != Marker.LINE_STRIP:
                continue
            for p in marker.points:
                pts.append((float(p.x), float(p.y)))
        if pts:
            self._trajectory_points = pts
            self._road_mask = None  # invalidate cached mask

    # =====================================================================
    # Constraint list callback — inter/intra constraint edges
    # =====================================================================
    def _constraint_callback(self, msg: MarkerArray):
        """
        /constraint_list is a MarkerArray with LINE_LIST markers.
        Namespaces:
          'Intra constraints' — sequential within submap
          'Inter constraints, same trajectory' — loop closures
        We extract the node endpoints to densify the road boundary.
        """
        pts = []
        for marker in msg.markers:
            if marker.type != Marker.LINE_LIST:
                continue
            # Keep intra + inter same-trajectory (both define road)
            ns_lower = marker.ns.lower()
            if 'intra' in ns_lower or 'same trajectory' in ns_lower:
                for p in marker.points:
                    pts.append((float(p.x), float(p.y)))
        if pts:
            self._constraint_points = pts
            self._road_mask = None  # invalidate cached mask

    # =====================================================================
    # Build road boundary mask from trajectory + constraints
    # =====================================================================
    def _build_road_mask(self):
        """
        Rasterize trajectory + constraint points into a binary mask
        on the grid. The mask represents 'cells the robot has driven through'
        = the road boundary ring.
        Returns the mask or None if no data yet.
        """
        if self.map_info is None or not hasattr(self, 'grid_rows'):
            return None

        all_pts = self._trajectory_points + self._constraint_points
        if len(all_pts) < 10:
            return None

        # Deduplicate by converting to grid coords
        res = self.map_info.resolution
        corner_x = self.map_info.pose.position.x - self.map_info.length_x / 2.0
        corner_y = self.map_info.pose.position.y - self.map_info.length_y / 2.0
        rows = self.grid_rows
        cols = self.grid_cols

        mask = np.zeros((rows, cols), dtype=np.uint8)

        # Draw polyline segments on mask
        prev_cr = None
        for wx, wy in self._trajectory_points:
            c = int((wx - corner_x) / res)
            r = int((wy - corner_y) / res)
            if 0 <= c < cols and 0 <= r < rows:
                if prev_cr is not None:
                    cv2.line(mask, prev_cr, (c, r), 1, thickness=1)
                prev_cr = (c, r)
            else:
                prev_cr = None

        # Also stamp constraint points
        for wx, wy in self._constraint_points:
            c = int((wx - corner_x) / res)
            r = int((wy - corner_y) / res)
            if 0 <= c < cols and 0 <= r < rows:
                mask[r, c] = 1

        # Dilate to create road band
        w = max(1, int(self.road_mask_width_cells))
        k = 2 * w + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask, kernel)

        self._road_mask = mask
        n_road = int(np.count_nonzero(mask))
        self.get_logger().info(
            f'Road mask built: {len(self._trajectory_points)} traj pts + '
            f'{len(self._constraint_points)} constr pts → '
            f'{n_road} road cells (width={w})',
            throttle_duration_sec=5.0,
        )
        return mask

    # =====================================================================
    # Detect gaps in the road ring — openings to the interior
    # =====================================================================
    def _detect_road_gaps(self, occupancy, unknown_mask, robot_xy=None):
        """
        Finds gaps/openings in the trajectory road ring.
        A gap is a segment of the road ring boundary that borders unknown
        (interior) space without being blocked by walls.

        These gaps are where the robot can turn off the perimeter road
        to enter the unexplored center.
        """
        if self._road_mask is None:
            mask = self._build_road_mask()
            if mask is None:
                return []

        road = self._road_mask
        if road is None or occupancy is None:
            return []

        rows, cols = occupancy.shape
        if road.shape != (rows, cols):
            # Grid size mismatch — rebuild
            self._road_mask = None
            return []

        # Find the inner boundary of the road mask — cells on the road
        # that are adjacent to unknown/unexplored cells (the center).
        n4_kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.uint8)
        unknown_u8 = unknown_mask.astype(np.uint8)
        has_unknown_neighbor = cv2.filter2D(unknown_u8, cv2.CV_16U, n4_kernel) > 0

        # Road cells adjacent to unknown = inner edge of the road ring
        inner_edge = (road > 0) & has_unknown_neighbor

        # Also require that these cells are free (navigable)
        free_mask = occupancy == 0.0
        # Check adjacent free cells — the gap should have a free corridor
        free_u8 = free_mask.astype(np.uint8)
        has_free_neighbor = cv2.filter2D(free_u8, cv2.CV_16U, n4_kernel) > 0

        gap_candidates = inner_edge & has_free_neighbor

        # Remove cells near walls
        wall_u8 = (occupancy > 0.5).astype(np.uint8)
        wall_dilated = cv2.dilate(wall_u8, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (3, 3)
        )) > 0
        gap_candidates = gap_candidates & (~wall_dilated)

        gap_u8 = gap_candidates.astype(np.uint8)
        n_gap = int(np.count_nonzero(gap_u8))
        if n_gap == 0:
            return []

        # Cluster gap cells
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            gap_u8, connectivity=8
        )

        min_opening = max(2, int(self.road_gap_min_opening_cells))
        reachable_mask = self._compute_reachable_mask(occupancy, robot_xy)

        points = []
        for label_id in range(1, n_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area < min_opening:
                continue

            cx = float(centroids[label_id][0])  # col
            cy = float(centroids[label_id][1])  # row
            col = int(round(cx))
            row = int(round(cy))
            if row < 0 or row >= rows or col < 0 or col >= cols:
                continue

            wx, wy = self._grid_to_world(col, row)
            dist_robot = -1.0
            if robot_xy is not None:
                dist_robot = float(np.hypot(wx - robot_xy[0], wy - robot_xy[1]))

            reachable = True
            if reachable_mask is not None:
                # Check if any cell in the gap cluster is reachable
                cluster = labels == label_id
                reachable = bool(np.any(reachable_mask[cluster]))

            score = float(self.road_gap_boost_score) + 0.2 * area

            points.append({
                'row': row,
                'col': col,
                'wx': wx,
                'wy': wy,
                'area': area,
                'sectors': 4,
                'reachable': reachable,
                'dist_robot': dist_robot,
                'score': score,
                'penetration': True,
                'road_gap': True,
            })

        # Sort: reachable first, then nearest
        points.sort(key=lambda p: (
            0 if p['reachable'] else 1,
            p['dist_robot'] if p['dist_robot'] >= 0 else 1.0e9,
        ))

        self.get_logger().info(
            f'Road gap detection: {n_gap} edge cells → '
            f'{n_labels - 1} clusters → {len(points)} valid gaps',
            throttle_duration_sec=3.0,
        )

        # Publish gap markers for RViz
        self._publish_road_gap_markers(points)

        return points[:int(self.max_interest_points)]

    def _publish_road_gap_markers(self, gaps):
        """Publish magenta markers at detected road gaps."""
        ma = MarkerArray()
        clear = Marker()
        clear.header.stamp = self.get_clock().now().to_msg()
        clear.header.frame_id = 'map'
        clear.ns = 'road_gaps'
        clear.id = 0
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)

        for i, g in enumerate(gaps):
            m = Marker()
            m.header.stamp = self.get_clock().now().to_msg()
            m.header.frame_id = 'map'
            m.ns = 'road_gaps'
            m.id = i + 1
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = g['wx']
            m.pose.position.y = g['wy']
            m.pose.position.z = 0.6
            m.scale.x = 0.4
            m.scale.y = 0.4
            m.scale.z = 0.4
            # Magenta for road gaps
            m.color.r = 1.0
            m.color.g = 0.0
            m.color.b = 0.85
            m.color.a = 0.9
            m.lifetime.sec = 3
            ma.markers.append(m)

        self.road_gap_marker_pub.publish(ma)

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
        robot_xy = self._get_robot_pose_xy()

        # ── Extract working area cells ──────────────────────────────────
        work_mask = self._working_area_mask(uncertainty)

        # Only consider FREE cells (occupancy == 0.0) for the metric
        # New convention: -1=unknown, 0=free, 1=wall
        if occupancy is not None:
            free_mask = occupancy == 0.0
            metric_mask = work_mask & free_mask
            unknown_mask = occupancy < 0.0
        else:
            metric_mask = work_mask
            unknown_mask = np.zeros_like(work_mask, dtype=bool)

        n_cells = np.sum(metric_mask)
        if n_cells == 0:
            self.get_logger().warn(
                'No free cells in working area yet',
                throttle_duration_sec=5.0
            )
            return

        # ── Mean uncertainty: U_bar = (1/N) * Σ U_i ────────────────────
        u_values = uncertainty[metric_mask]
        u_bar = float(np.mean(u_values))
        mapped_ratio = float(np.mean(u_values <= self.certainty_threshold))
        mapped_pct = mapped_ratio * 100.0
        explored_pct = float((1.0 - u_bar) * 100.0)

        interest_points = self._detect_interest_points(
            occupancy=occupancy,
            uncertainty=uncertainty,
            work_mask=work_mask,
            free_work_mask=metric_mask,
            unknown_mask=unknown_mask,
            robot_xy=robot_xy,
        )

        # ── Track cumulative robot travel (startup guard) ─────────────
        if robot_xy is not None:
            if self._prev_robot_xy is not None:
                dx = robot_xy[0] - self._prev_robot_xy[0]
                dy = robot_xy[1] - self._prev_robot_xy[1]
                self._robot_travel += float(np.hypot(dx, dy))
            self._prev_robot_xy = robot_xy

        # ── Penetration point injection ─────────────────────────────────
        # Skip if robot hasn't moved at least 1 m (avoids spurious
        # points while the robot is still at spawn).
        if self.enable_penetration_points and self._robot_travel >= 1.0:
            pen_pts = self._detect_penetration_points(
                occupancy=occupancy,
                unknown_mask=unknown_mask,
                work_mask=work_mask,
                robot_xy=robot_xy,
            )
            if pen_pts:
                interest_points.extend(pen_pts)
                # Re-sort: penetration points are high score so they
                # surface at the top.
                interest_points.sort(
                    key=lambda p: (
                        p['dist_robot'] if p['dist_robot'] >= 0.0 else 1.0e9,
                        -p['score'],
                    )
                )
                interest_points = interest_points[:int(self.max_interest_points)]
                self.get_logger().info(
                    f'Penetration injection: {len(pen_pts)} ingress points added',
                    throttle_duration_sec=3.0,
                )

        # ── Road gap injection (from Cartographer trajectory) ───────────
        if self.enable_trajectory_road_mask:
            gap_pts = self._detect_road_gaps(
                occupancy=occupancy,
                unknown_mask=unknown_mask,
                robot_xy=robot_xy,
            )
            if gap_pts:
                interest_points.extend(gap_pts)
                interest_points.sort(
                    key=lambda p: (
                        p['dist_robot'] if p['dist_robot'] >= 0.0 else 1.0e9,
                        -p['score'],
                    )
                )
                interest_points = interest_points[:int(self.max_interest_points)]
                self.get_logger().info(
                    f'Road gap injection: {len(gap_pts)} gap points added',
                    throttle_duration_sec=3.0,
                )

        # ── Accumulate into persistent point bank ───────────────────
        self._accumulate_point_bank(interest_points)

        # ── Remove explored points near robot ───────────────────────────
        if robot_xy is not None:
            self._remove_explored_near_robot(robot_xy)
            self._publish_reveal_ring(robot_xy)

        # ── Detect hotspot (densest cluster = entrance) ──────────────
        self._hotspot_key = self._find_hotspot_key()

        # ── Publish only pending (un-visited) markers ───────────────────
        pending_pts = self._get_pending_points()
        self._publish_pending_markers()

        # Use pending (un-visited) points for goal selection
        reachable_interest = int(sum(
            1 for p in pending_pts if p.get('reachable', True)
        ))

        selected_goal_dist = -1.0
        self._publish_metrics(
            mean_uncertainty=u_bar,
            mapped_pct=mapped_pct,
            explored_pct=explored_pct,
            n_free_cells=int(n_cells),
            n_interest=len(interest_points),
            n_reachable_interest=reachable_interest,
            selected_goal_dist=selected_goal_dist,
            robot_xy=robot_xy,
        )

        self.get_logger().info(
            f'[{self.state}] Mean uncertainty: {u_bar:.3f} | '
            f'Mapped with certainty: {mapped_pct:.1f}% | '
            f'Explored estimate: {explored_pct:.1f}% | '
            f'Interest points: {len(interest_points)} | '
            f'Threshold τ: {self.tau}'
            ,
            throttle_duration_sec=1.0
        )

        # ── State logic ─────────────────────────────────────────────────
        if self.state == self.STATE_MAPPING:
            # Continue mapping while either uncertainty is high OR there are
            # intersections/frontiers with undiscovered branches.
            should_continue_mapping = (
                (u_bar > self.tau)
                or (mapped_ratio < float(self.mapping_completion_threshold))
                or (reachable_interest > 0)
            )
            if should_continue_mapping:
                selected = self._publish_best_interest_goal(pending_pts, robot_xy)
                if selected is not None:
                    selected_goal_dist = float(selected.get('dist_robot', -1.0))
                self._publish_metrics(
                    mean_uncertainty=u_bar,
                    mapped_pct=mapped_pct,
                    explored_pct=explored_pct,
                    n_free_cells=int(n_cells),
                    n_interest=len(interest_points),
                    n_reachable_interest=reachable_interest,
                    selected_goal_dist=selected_goal_dist,
                    robot_xy=robot_xy,
                )
            else:
                # Map is complete! Switch to READY
                self.state = self.STATE_READY
                self.get_logger().info(
                    '═══════════════════════════════════════════════\n'
                    '  ✅ MAP COMPLETE — Switching to READY state\n'
                    f'  Final mean uncertainty: {u_bar:.4f} ≤ {self.tau}\n'
                    f'  Mapped with certainty: {mapped_pct:.1f}% '
                    f'(target {self.mapping_completion_threshold * 100.0:.1f}%)\n'
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

        signature = (
            layer.shape,
            float(self.map_info.resolution),
            float(self.map_info.pose.position.x),
            float(self.map_info.pose.position.y),
            float(self.map_info.length_x),
            float(self.map_info.length_y),
            float(self.wa_x_min),
            float(self.wa_x_max),
            float(self.wa_y_min),
            float(self.wa_y_max),
        )
        if self._work_mask_cache is not None and signature == self._work_mask_signature:
            return self._work_mask_cache

        res = self.map_info.resolution
        # Corner = centre - half-length (matches map_processor_node convention)
        corner_x = self.map_info.pose.position.x - self.map_info.length_x / 2.0
        corner_y = self.map_info.pose.position.y - self.map_info.length_y / 2.0

        rows, cols = layer.shape
        col_coords = corner_x + (np.arange(cols) + 0.5) * res
        row_coords = corner_y + (np.arange(rows) + 0.5) * res

        col_mask = (col_coords >= self.wa_x_min) & (col_coords <= self.wa_x_max)
        row_mask = (row_coords >= self.wa_y_min) & (row_coords <= self.wa_y_max)

        out = np.outer(row_mask, col_mask)
        self._work_mask_cache = out
        self._work_mask_signature = signature
        return out

    # =====================================================================
    # Frontier/intersection detection and goal publication
    # =====================================================================
    def _detect_interest_points(
        self,
        occupancy: np.ndarray,
        uncertainty: np.ndarray,
        work_mask: np.ndarray,
        free_work_mask: np.ndarray,
        unknown_mask: np.ndarray,
        robot_xy=None,
    ):
        """
        Detects frontier points and filters them to keep intersection-like
        candidates (multiple unknown sectors around a free frontier cell).
        """
        if occupancy is None:
            return []

        free_global = occupancy == 0.0

        # Frontier from planner uncertainty: free/high-uncertainty touching
        # free/low-uncertainty. This aligns with /planner_uncertainty.
        uncertain_free = free_work_mask & (
            uncertainty >= float(self.frontier_uncertainty_threshold)
        )
        discovered_free = free_work_mask & (
            uncertainty <= float(self.certainty_threshold)
        )

        discovered_u8 = discovered_free.astype(np.uint8)
        n4_kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.uint8)
        has_discovered_neighbor = cv2.filter2D(
            discovered_u8, cv2.CV_16U, n4_kernel
        ) > 0
        frontier_cells = uncertain_free & has_discovered_neighbor

        # Optional fallback to occupancy unknown frontier (disabled by default).
        if (not np.any(frontier_cells)) and self.frontier_fallback_to_map_unknown:
            unknown_work = unknown_mask & work_mask
            free_u8 = free_global.astype(np.uint8)
            has_free_neighbor = cv2.filter2D(free_u8, cv2.CV_16U, n4_kernel) > 0
            frontier_cells = unknown_work & has_free_neighbor

        # Remove cells too close to walls to avoid center-of-road artifacts.
        clearance = int(self.frontier_clearance_cells)
        wall_near = None
        if clearance > 0:
            wall_u8 = (occupancy > 0.5).astype(np.uint8)
            k_wall = 2 * clearance + 1
            kernel_wall = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_wall, k_wall))
            wall_near = cv2.dilate(wall_u8, kernel_wall) > 0

        frontier_u8 = frontier_cells.astype(np.uint8)
        if np.count_nonzero(frontier_u8) == 0:
            return []

        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            frontier_u8, connectivity=max(4, int(self.frontier_connectivity))
        )

        # Optional reachable-set from robot pose on inflated free space.
        reachable_mask = self._compute_reachable_mask(occupancy, robot_xy)
        robot_grid = self._world_to_grid(robot_xy[0], robot_xy[1]) if robot_xy is not None else None

        free_dist = cv2.distanceTransform(free_global.astype(np.uint8), cv2.DIST_L2, 3)

        points = []
        fallback_points = []
        for label_id in range(1, n_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area < int(self.frontier_min_size):
                continue

            cluster_frontier = labels == label_id

            # Candidate cells are frontier cells (already free + uncertain).
            candidate_free = cluster_frontier & free_work_mask
            if wall_near is not None:
                candidate_free &= (~wall_near)

            if np.count_nonzero(candidate_free) == 0:
                continue

            c_rows, c_cols = np.where(candidate_free)
            if c_cols.size == 0:
                continue

            # Nearest frontier in real-time: distance to robot has top priority.
            if robot_grid is not None:
                rc, rr = robot_grid
                d2_robot = (c_cols.astype(np.float32) - rc) ** 2 + (c_rows.astype(np.float32) - rr) ** 2
                idx = int(np.argmin(d2_robot))
            else:
                # Fallback: nearest to component centroid.
                cx, cy = centroids[label_id]
                d2_centroid = (c_cols.astype(np.float32) - cx) ** 2 + (c_rows.astype(np.float32) - cy) ** 2
                idx = int(np.argmin(d2_centroid))

            col = int(c_cols[idx])
            row = int(c_rows[idx])

            py, px = np.where(labels == label_id)
            if px.size == 0:
                continue

            sectors = self._count_unknown_sectors(uncertain_free, col, row)

            wx, wy = self._grid_to_world(col, row)
            dist_robot = -1.0
            if robot_xy is not None:
                dist_robot = float(np.hypot(wx - robot_xy[0], wy - robot_xy[1]))
            reachable = True
            if reachable_mask is not None:
                reachable = bool(reachable_mask[row, col])

            # Score still available for tie-breakers; nearest is handled at publish step.
            score = (0.1 * float(area)) + (2.0 * float(sectors)) + (0.5 * float(free_dist[row, col]))
            candidate = {
                'row': row,
                'col': col,
                'wx': wx,
                'wy': wy,
                'area': area,
                'sectors': sectors,
                'reachable': reachable,
                'dist_robot': dist_robot,
                'score': score,
            }
            fallback_points.append(candidate)
            if sectors >= int(self.min_unknown_sectors):
                points.append(candidate)

        if len(points) == 0 and len(fallback_points) > 0:
            # Fallback: keep frontiers even without strong intersection geometry.
            points = fallback_points

        self.get_logger().info(
            f'Frontier debug: frontier_pixels={int(np.count_nonzero(frontier_u8))} '
            f'clusters={int(n_labels - 1)} candidates={len(points)} '
            f'unc_thr={self.frontier_uncertainty_threshold:.2f}',
            throttle_duration_sec=2.0,
        )

        # Keep deterministic ordering: nearest first, then richer frontier.
        points.sort(key=lambda p: (p['dist_robot'] if p['dist_robot'] >= 0.0 else 1.0e9, -p['score']))
        return points[:int(self.max_interest_points)]

    def _count_unknown_sectors(self, unknown_mask: np.ndarray, col: int, row: int) -> int:
        """Count separated angular sectors of unknown area around a point."""
        radius = int(self.interest_radius_cells)
        if radius <= 1:
            return 0

        r0 = max(0, row - radius)
        r1 = min(unknown_mask.shape[0], row + radius + 1)
        c0 = max(0, col - radius)
        c1 = min(unknown_mask.shape[1], col + radius + 1)

        local = unknown_mask[r0:r1, c0:c1]
        if local.size == 0 or not np.any(local):
            return 0

        ys, xs = np.where(local)
        dy = ys + r0 - row
        dx = xs + c0 - col

        d2 = dx * dx + dy * dy
        r2 = radius * radius
        inner2 = max(1, int((0.35 * radius) ** 2))
        ring = (d2 <= r2) & (d2 >= inner2)
        if not np.any(ring):
            return 0

        dx = dx[ring]
        dy = dy[ring]
        if dx.size == 0:
            return 0

        angles = np.arctan2(dy.astype(np.float32), dx.astype(np.float32))
        bins = 24
        idx = ((angles + np.pi) * (bins / (2.0 * np.pi))).astype(np.int32) % bins
        occ = np.zeros(bins, dtype=np.uint8)
        occ[idx] = 1

        # Light smoothing to tolerate small gaps/noise.
        occ = np.maximum(occ, np.roll(occ, 1))
        occ = np.maximum(occ, np.roll(occ, -1))

        transitions = np.sum((np.roll(occ, 1) == 0) & (occ == 1))
        return int(transitions)

    def _publish_best_interest_goal(self, interest_points, robot_xy=None):
        """Publish nearest reachable frontier goal with hysteresis against switching."""
        if not interest_points:
            return None

        valid = [p for p in interest_points if p.get('reachable', True)]
        if not valid:
            return None

        # ── Compute attractor = centroid of unknown mass ──────────────
        # This pulls goals toward the actual unexplored region rather than
        # the geometric center of the map.
        attractor_xy = self._map_center_xy()  # fallback
        if hasattr(self, 'latest_gridmap') and self.latest_gridmap is not None:
            occ = self.latest_gridmap.get('occupancy', None)
            if occ is not None:
                unk_mask = occ < 0.0
                w_mask = self._working_area_mask(occ)
                cent_info = self._unknown_mass_centroid(unk_mask, w_mask)
                if cent_info is not None:
                    cc, cr, _ = cent_info
                    ax, ay = self._grid_to_world(cc, cr)
                    attractor_xy = (ax, ay)

        if robot_xy is not None:
            for p in valid:
                p['dist_robot'] = float(np.hypot(p['wx'] - robot_xy[0], p['wy'] - robot_xy[1]))
        if attractor_xy is not None:
            for p in valid:
                p['dist_center'] = float(
                    np.hypot(p['wx'] - attractor_xy[0], p['wy'] - attractor_xy[1])
                )
        else:
            for p in valid:
                p['dist_center'] = 0.0

        # Prefer local frontiers around robot for real-time reaction.
        local_candidates = [
            p for p in valid
            if p.get('dist_robot', 1.0e9) <= float(self.local_frontier_radius_m)
        ]
        pool = local_candidates if local_candidates else valid

        # Identify the hotspot world position for ranking bonus
        _hotspot_wx = None
        _hotspot_wy = None
        if self._hotspot_key is not None and self._hotspot_key in self._point_bank:
            hp = self._point_bank[self._hotspot_key]
            _hotspot_wx = hp['wx']
            _hotspot_wy = hp['wy']

        def rank_key(p):
            d_robot = float(p.get('dist_robot', 1.0e9))
            d_center = float(p.get('dist_center', 0.0))
            is_pen = 1.0 if p.get('penetration', False) else 0.0
            is_gap = 1.0 if p.get('road_gap', False) else 0.0
            # Check if this point IS the hotspot
            is_hot = 0.0
            if _hotspot_wx is not None:
                if abs(p['wx'] - _hotspot_wx) < 0.05 and abs(p['wy'] - _hotspot_wy) < 0.05:
                    is_hot = 1.0
            # Penetration points get a bonus; road gaps even more; hotspot highest
            pen_bonus = -2.0 * is_pen - 3.0 * is_gap - 5.0 * is_hot
            cost = d_robot + float(self.center_attractor_weight) * d_center + pen_bonus
            return (cost, d_robot, -p.get('score', 0.0))

        pool.sort(key=rank_key)
        best = pool[0]

        # Histéresis de selección: evitar cambiar de objetivo por mejoras pequeñas.
        selected = best
        if self._last_goal_xy is not None:
            keep = min(
                valid,
                key=lambda p: (p['wx'] - self._last_goal_xy[0]) ** 2 + (p['wy'] - self._last_goal_xy[1]) ** 2,
            )
            d_keep_to_last = float(np.hypot(keep['wx'] - self._last_goal_xy[0], keep['wy'] - self._last_goal_xy[1]))
            best_dist = float(best.get('dist_robot', 1.0e9))
            keep_dist = float(keep.get('dist_robot', 1.0e9))
            if d_keep_to_last <= float(self.goal_republish_distance) * 2.0 and keep_dist <= (best_dist + float(self.goal_switch_margin_m)):
                selected = keep

        wx = float(selected['wx'])
        wy = float(selected['wy'])
        now_ns = self.get_clock().now().nanoseconds

        should_publish = True
        if self._last_goal_xy is not None:
            dx = wx - self._last_goal_xy[0]
            dy = wy - self._last_goal_xy[1]
            dist = float(np.hypot(dx, dy))
            dt = (now_ns - self._last_goal_time_ns) * 1e-9
            if dist < float(self.goal_republish_distance) and dt < float(self.goal_republish_period_sec):
                should_publish = False

        if not should_publish:
            return selected

        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = 'map'
        goal.pose.position.x = wx
        goal.pose.position.y = wy
        goal.pose.position.z = 0.0
        goal.pose.orientation.w = 1.0
        self.goal_pub.publish(goal)

        self._last_goal_xy = (wx, wy)
        self._last_goal_time_ns = now_ns
        pen_tag = ''
        if _hotspot_wx is not None and abs(wx - _hotspot_wx) < 0.05 and abs(wy - _hotspot_wy) < 0.05:
            pen_tag = ' [★ HOTSPOT]'
        elif selected.get('road_gap', False):
            pen_tag = ' [ROAD_GAP]'
        elif selected.get('penetration', False):
            pen_tag = ' [PENETRATION]'
        self.get_logger().info(
            f'Published exploration goal: ({wx:.2f}, {wy:.2f}){pen_tag} | '
            f'sectors={selected["sectors"]}, cluster={selected["area"]} | '
            f'd_robot={selected.get("dist_robot", -1.0):.2f}m | '
            f'd_attractor={selected.get("dist_center", -1.0):.2f}m'
        )
        return selected

    # =================================================================
    # Penetration-point detection — finds ingress corridors into
    # large unexplored regions
    # =================================================================
    def _unknown_mass_centroid(self, unknown_mask: np.ndarray, work_mask: np.ndarray):
        """
        Return (centroid_col, centroid_row, blob_size) of the largest
        connected unknown blob in the work area, or None.
        """
        unk_work = (unknown_mask & work_mask).astype(np.uint8)
        n_unk = int(np.count_nonzero(unk_work))
        if n_unk < int(self.min_unknown_blob_cells):
            return None

        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            unk_work, connectivity=8
        )
        if n_labels <= 1:
            return None

        # Largest blob (skip label 0 = background)
        areas = stats[1:, cv2.CC_STAT_AREA]
        best_label = int(np.argmax(areas)) + 1
        best_area = int(areas[best_label - 1])
        if best_area < int(self.min_unknown_blob_cells):
            return None

        cx = float(centroids[best_label][0])  # col
        cy = float(centroids[best_label][1])  # row
        return (cx, cy, best_area)

    def _detect_penetration_points(
        self,
        occupancy: np.ndarray,
        unknown_mask: np.ndarray,
        work_mask: np.ndarray,
        robot_xy=None,
    ):
        """
        Raycast from the centroid of the largest unknown blob outward.
        Where a ray crosses through free space (a corridor/opening),
        mark the free-side entry cell as a penetration point.

        These points represent *ingress corridors* the robot can use
        to reach the unexplored interior.
        """
        if occupancy is None:
            return []

        centroid_info = self._unknown_mass_centroid(unknown_mask, work_mask)
        if centroid_info is None:
            return []

        c_col, c_row, blob_size = centroid_info

        # Publish centroid marker for RViz debugging
        self._publish_centroid_marker(c_col, c_row)

        rows, cols = occupancy.shape
        free_mask = occupancy == 0.0

        # Reachable mask from robot
        reachable_mask = self._compute_reachable_mask(occupancy, robot_xy)

        n_rays = max(8, int(self.penetration_ray_count))
        step = max(1, int(self.penetration_ray_step_cells))
        min_free = max(2, int(self.penetration_min_free_run))
        max_dist = int(max(rows, cols))

        angles = np.linspace(0.0, 2.0 * np.pi, n_rays, endpoint=False)
        points = []
        seen_cells = set()

        for angle in angles:
            dx = np.cos(angle)
            dy = np.sin(angle)

            # Walk outward from centroid
            # State machine: UNKNOWN -> crossing -> FREE run
            in_unknown = True
            free_run = 0
            entry_col = -1
            entry_row = -1

            for t in range(1, max_dist, step):
                cc = int(round(c_col + dx * t))
                rr = int(round(c_row + dy * t))
                if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                    break

                is_free = bool(free_mask[rr, cc])
                is_wall = bool(occupancy[rr, cc] > 0.5)
                is_unknown = bool(unknown_mask[rr, cc])

                if in_unknown:
                    if is_free:
                        # Transition: entered free space
                        in_unknown = False
                        free_run = 1
                        entry_col = cc
                        entry_row = rr
                    elif is_wall:
                        break  # Wall blocks this ray
                else:
                    # Already in free space
                    if is_free:
                        free_run += 1
                    elif is_wall or is_unknown:
                        # End of free corridor
                        break

            # If we found a free corridor of sufficient length from 
            # the unknown boundary, record entry point
            if not in_unknown and free_run >= min_free and entry_col >= 0:
                cell_key = (entry_col // 3, entry_row // 3)  # spatial dedup
                if cell_key in seen_cells:
                    continue
                seen_cells.add(cell_key)

                wx, wy = self._grid_to_world(entry_col, entry_row)
                dist_robot = -1.0
                if robot_xy is not None:
                    dist_robot = float(
                        np.hypot(wx - robot_xy[0], wy - robot_xy[1])
                    )

                reachable = True
                if reachable_mask is not None:
                    reachable = bool(reachable_mask[entry_row, entry_col])

                # High score so these surface above regular frontiers
                score = float(self.penetration_boost_score) + 0.1 * free_run

                points.append({
                    'row': entry_row,
                    'col': entry_col,
                    'wx': wx,
                    'wy': wy,
                    'area': free_run,
                    'sectors': 4,  # virtual: corridor implies directionality
                    'reachable': reachable,
                    'dist_robot': dist_robot,
                    'score': score,
                    'penetration': True,
                })

        # Sort by reachable first, then nearest to robot
        points.sort(key=lambda p: (
            0 if p['reachable'] else 1,
            p['dist_robot'] if p['dist_robot'] >= 0 else 1.0e9,
        ))

        self.get_logger().info(
            f'Penetration rays: {n_rays} | blob={blob_size} cells | '
            f'centroid=({c_col:.0f},{c_row:.0f}) | ingress={len(points)}',
            throttle_duration_sec=3.0,
        )
        return points[:int(self.max_interest_points)]

    def _publish_centroid_marker(self, col: float, row: float):
        """Publish a large marker at the unknown-mass centroid for debugging."""
        wx, wy = self._grid_to_world(col, row)
        m = Marker()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = 'map'
        m.ns = 'unknown_centroid'
        m.id = 0
        m.type = Marker.CYLINDER
        m.action = Marker.ADD
        m.pose.position.x = wx
        m.pose.position.y = wy
        m.pose.position.z = 0.5
        m.scale.x = 0.5
        m.scale.y = 0.5
        m.scale.z = 0.8
        m.color.r = 1.0
        m.color.g = 0.65
        m.color.b = 0.0
        m.color.a = 0.85
        m.lifetime.sec = 3
        self.centroid_marker_pub.publish(m)

    def _map_center_xy(self):
        """Return map geometric center in map frame."""
        if self.map_info is None:
            return None
        return (
            float(self.map_info.pose.position.x),
            float(self.map_info.pose.position.y),
        )

    # =================================================================
    # Persistent Point Bank — delete-on-visit
    # =================================================================

    def _accumulate_point_bank(self, interest_points):
        """Add newly detected interest points into the persistent bank.

        Each point is stored under a spatial-cell key so that nearby
        duplicates are deduplicated automatically.  If the bank exceeds
        ``point_bank_max_size`` the oldest entries are dropped.
        """
        cs = self.point_bank_cell_size_m
        for p in interest_points:
            kx = int(p['wx'] / cs) if p['wx'] >= 0 else int(p['wx'] / cs) - 1
            ky = int(p['wy'] / cs) if p['wy'] >= 0 else int(p['wy'] / cs) - 1
            key = (kx, ky)
            if key not in self._point_bank:
                self._point_bank[key] = dict(p)  # store a copy

        # Evict oldest entries when over capacity
        if len(self._point_bank) > self.point_bank_max_size:
            excess = len(self._point_bank) - self.point_bank_max_size
            keys_to_remove = list(self._point_bank.keys())[:excess]
            for k in keys_to_remove:
                del self._point_bank[k]

    def _remove_explored_near_robot(self, robot_xy):
        """DELETE bank entries within ``reveal_radius_m`` of the robot.

        The robot has already explored those locations, so they are no
        longer pending and should disappear from the map.
        """
        rx, ry = robot_xy
        r2 = self.reveal_radius_m ** 2
        to_delete = [
            key for key, p in self._point_bank.items()
            if (p['wx'] - rx) ** 2 + (p['wy'] - ry) ** 2 <= r2
        ]
        for key in to_delete:
            del self._point_bank[key]

    def _get_pending_points(self):
        """Return list of point dicts still in the bank (pending exploration)."""
        return list(self._point_bank.values())

    def _publish_pending_markers(self):
        """Publish RViz markers for pending (un-visited) bank points only.

        Uses DELETEALL first so that markers from deleted bank entries
        disappear cleanly in RViz.
        """
        marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        # Always clear stale markers from the previous cycle
        clear = Marker()
        clear.header.stamp = stamp
        clear.header.frame_id = 'map'
        clear.ns = 'point_bank'
        clear.id = 0
        clear.action = Marker.DELETEALL
        marker_array.markers.append(clear)

        for idx, (key, p) in enumerate(self._point_bank.items()):
            is_hotspot = (key == self._hotspot_key)

            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = 'map'
            m.ns = 'point_bank'
            m.id = idx + 1
            m.action = Marker.ADD
            m.pose.position.x = float(p['wx'])
            m.pose.position.y = float(p['wy'])
            m.pose.position.z = 0.45 if not is_hotspot else 0.65

            # Base scale
            sx = 0.28
            sy = 0.28
            sz = 0.28

            if is_hotspot:
                # ★ Hotspot: bright gold diamond — entrance to new path
                m.type = Marker.CUBE
                m.color.r = 1.0
                m.color.g = 0.85
                m.color.b = 0.0
                sx = sy = sz = 0.52
            # Colour / shape by type
            elif p.get('road_gap', False):
                m.type = Marker.CUBE
                m.color.r = 1.0
                m.color.g = 0.0
                m.color.b = 0.85
                sx = sy = sz = 0.38
            elif p.get('penetration', False):
                m.type = Marker.SPHERE
                m.color.r = 0.0
                m.color.g = 0.85
                m.color.b = 1.0
                sx = sy = sz = 0.35
            elif p.get('reachable', True):
                m.type = Marker.SPHERE
                m.color.r = 0.15
                m.color.g = 0.9
                m.color.b = 0.2
            else:
                m.type = Marker.SPHERE
                m.color.r = 0.9
                m.color.g = 0.2
                m.color.b = 0.2

            m.scale.x = sx
            m.scale.y = sy
            m.scale.z = sz
            m.color.a = 1.0 if is_hotspot else 0.85
            marker_array.markers.append(m)

        self.marker_pub.publish(marker_array)

    def _find_hotspot_key(self):
        """Find the bank key whose neighbourhood has the highest point density.

        For every point in the bank, count how many other points fall within
        ``hotspot_radius_m``.  The point with the highest neighbour count
        (>= ``hotspot_min_points``) is marked as the *hotspot* — the most
        likely entrance to an unexplored path.

        Returns the spatial key of that point, or None.
        """
        pts = list(self._point_bank.items())
        if len(pts) < self.hotspot_min_points:
            return None

        r2 = self.hotspot_radius_m ** 2
        best_key = None
        best_count = 0

        # Build coordinate arrays for vectorised distance
        keys = [k for k, _ in pts]
        coords = np.array([[p['wx'], p['wy']] for _, p in pts], dtype=np.float64)
        n = len(coords)

        for i in range(n):
            dx = coords[:, 0] - coords[i, 0]
            dy = coords[:, 1] - coords[i, 1]
            count = int(np.sum(dx * dx + dy * dy <= r2))  # includes self
            if count > best_count:
                best_count = count
                best_key = keys[i]

        if best_count >= self.hotspot_min_points:
            return best_key
        return None

    def _publish_reveal_ring(self, robot_xy):
        """Publish a translucent cylinder ring around the robot showing the clear radius."""
        m = Marker()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = 'map'
        m.ns = 'reveal_ring'
        m.id = 0
        m.type = Marker.CYLINDER
        m.action = Marker.ADD
        m.pose.position.x = float(robot_xy[0])
        m.pose.position.y = float(robot_xy[1])
        m.pose.position.z = 0.05  # just above ground plane
        m.pose.orientation.w = 1.0
        diameter = self.reveal_radius_m * 2.0
        m.scale.x = diameter
        m.scale.y = diameter
        m.scale.z = 0.02  # very thin disk
        m.color.r = 0.3
        m.color.g = 0.8
        m.color.b = 1.0
        m.color.a = 0.18
        m.lifetime.sec = 3
        self.reveal_ring_pub.publish(m)

    # =================================================================
    # Legacy interest markers (unused — kept for reference)
    # =================================================================

    def _publish_interest_markers(self, interest_points):
        """Publish RViz markers for interest points (intersections/frontiers)."""
        marker_array = MarkerArray()

        clear = Marker()
        clear.header.stamp = self.get_clock().now().to_msg()
        clear.header.frame_id = 'map'
        clear.ns = 'frontiers'
        clear.id = 0
        clear.action = Marker.DELETEALL
        marker_array.markers.append(clear)

        for i, p in enumerate(interest_points):
            marker = Marker()
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.header.frame_id = 'map'
            marker.ns = 'frontiers'
            marker.id = i + 1
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = p['wx']
            marker.pose.position.y = p['wy']
            marker.pose.position.z = 0.45
            marker.scale.x = 0.28
            marker.scale.y = 0.28
            marker.scale.z = 0.28

            # Road-gap=magenta, penetration=cyan, reachable=green, unreachable=red.
            if p.get('road_gap', False):
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.85
                marker.type = Marker.CUBE
                marker.scale.x = 0.38
                marker.scale.y = 0.38
                marker.scale.z = 0.38
            elif p.get('penetration', False):
                marker.color.r = 0.0
                marker.color.g = 0.85
                marker.color.b = 1.0
                marker.scale.x = 0.35
                marker.scale.y = 0.35
                marker.scale.z = 0.35
            elif p.get('reachable', True):
                marker.color.r = 0.15
                marker.color.g = 0.9
                marker.color.b = 0.2
            else:
                marker.color.r = 0.9
                marker.color.g = 0.2
                marker.color.b = 0.2
            marker.color.a = 0.85
            marker.lifetime.sec = 2
            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)

    def _publish_metrics(
        self,
        mean_uncertainty: float,
        mapped_pct: float,
        explored_pct: float,
        n_free_cells: int,
        n_interest: int,
        n_reachable_interest: int,
        selected_goal_dist: float,
        robot_xy,
    ):
        """
        Standardized metrics v2 in fixed order (Float32MultiArray):
          [0] mapped_pct
          [1] explored_pct
          [2] mean_uncertainty
          [3] free_cells
          [4] interest_points
          [5] reachable_interest_points
          [6] selected_goal_dist_m
          [7] robot_x
          [8] robot_y
          [9] state (0=MAPPING, 1=READY)
        """
        msg = Float32MultiArray()
        dim = MultiArrayDimension()
        dim.label = 'exploration_metrics_v2'
        dim.size = 10
        dim.stride = 10
        msg.layout.dim = [dim]
        msg.layout.data_offset = 0
        rx = float(robot_xy[0]) if robot_xy is not None else float('nan')
        ry = float(robot_xy[1]) if robot_xy is not None else float('nan')
        msg.data = [
            float(mapped_pct),
            float(explored_pct),
            float(mean_uncertainty),
            float(n_free_cells),
            float(n_interest),
            float(n_reachable_interest),
            float(selected_goal_dist),
            rx,
            ry,
            0.0 if self.state == self.STATE_MAPPING else 1.0,
        ]
        self.metrics_pub.publish(msg)

    def _get_robot_pose_xy(self):
        """Get robot pose in map frame from TF. Returns tuple (x,y) or None."""
        try:
            t = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time()
            )
            return (
                float(t.transform.translation.x),
                float(t.transform.translation.y),
            )
        except Exception:
            return None

    def _world_to_grid(self, wx: float, wy: float):
        """Convert world coordinates to integer grid indices."""
        if self.map_info is None:
            return None
        res = self.map_info.resolution
        corner_x = self.map_info.pose.position.x - self.map_info.length_x / 2.0
        corner_y = self.map_info.pose.position.y - self.map_info.length_y / 2.0
        col = int((wx - corner_x) / res)
        row = int((wy - corner_y) / res)
        if row < 0 or row >= self.grid_rows or col < 0 or col >= self.grid_cols:
            return None
        return col, row

    def _compute_reachable_mask(self, occupancy: np.ndarray, robot_xy):
        """Compute reachable free cells from robot pose with optional inflation."""
        if occupancy is None or robot_xy is None:
            return None

        rc = self._world_to_grid(robot_xy[0], robot_xy[1])
        if rc is None:
            return None

        col, row = rc
        obstacle = (occupancy != 0.0).astype(np.uint8)
        inf = max(0, int(self.reachable_inflation_cells))
        if inf > 0:
            k = 2 * inf + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            obstacle = cv2.dilate(obstacle, kernel)

        navigable = (occupancy == 0.0) & (obstacle == 0)
        if not navigable[row, col]:
            # Find nearest navigable cell around current pose.
            found = None
            for rad in range(1, 8):
                r0 = max(0, row - rad)
                r1 = min(self.grid_rows, row + rad + 1)
                c0 = max(0, col - rad)
                c1 = min(self.grid_cols, col + rad + 1)
                local = navigable[r0:r1, c0:c1]
                if np.any(local):
                    ys, xs = np.where(local)
                    ys = ys + r0
                    xs = xs + c0
                    idx = int(np.argmin((xs - col) ** 2 + (ys - row) ** 2))
                    found = (int(xs[idx]), int(ys[idx]))
                    break
            if found is None:
                return None
            col, row = found

        img = (navigable.astype(np.uint8) * 255)
        flood = img.copy()
        mask = np.zeros((self.grid_rows + 2, self.grid_cols + 2), dtype=np.uint8)
        cv2.floodFill(flood, mask, (col, row), 127)
        return flood == 127

    def _invalidate_work_area_cache_if_needed(self):
        """Reset cached work-area mask when map geometry may change."""
        self._work_mask_cache = None
        self._work_mask_signature = None

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
