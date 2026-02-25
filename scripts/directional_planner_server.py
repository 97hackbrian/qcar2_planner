#!/usr/bin/env python3
# =============================================================================
# directional_planner_server.py — Oriented A* Planner (Node 3)
# =============================================================================
#
# Provides a service `/get_directional_path` that computes a path from the
# robot's current position to a goal using an oriented A* algorithm.
#
# ─────────────────────────────────────────────────────────────────────────────
# COST FUNCTION — Directional Dot-Product Penalty
# ─────────────────────────────────────────────────────────────────────────────
#
# When expanding from cell A to neighbor B, the cost is:
#
#     Cost(A, B) = dist(A, B) + P
#
# Where the penalty P enforces LEGAL driving direction:
#
#     V_AB = normalize(B - A)            # movement vector from A to B
#     L    = (dir_x[A], dir_y[A])        # legal direction stored in map
#
#     dot  = V_AB · L                    # dot product
#
#     if dot >= 0:  P = 0                # ✓ Correct direction (aligned)
#     if dot <  0:  P = direction_penalty # ✗ Wrong way (≈ ∞, blocks path)
#
# MATHEMATICAL INTUITION:
#   The dot product V · L = |V||L|cos(θ), where θ is the angle between the
#   movement direction and the legal lane direction.
#   - cos(θ) ≥ 0  when θ ∈ [-90°, +90°]  → driving WITH traffic
#   - cos(θ) < 0  when θ ∈ (90°, 270°)   → driving AGAINST traffic
#   By applying P ≈ ∞ when cos(θ) < 0, we effectively BLOCK all wrong-way
#   traversals, forcing A* to find paths that follow the legal lane direction.
#
# ─────────────────────────────────────────────────────────────────────────────
#
# GUARDRAIL: This node does NOT publish any motor commands.
#            Output is exclusively nav_msgs/Path via the service response.
# =============================================================================

import heapq
import math
import numpy as np

import rclpy
from rclpy.node import Node

from grid_map_msgs.msg import GridMap as GridMapMsg
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray
from std_srvs.srv import SetBool
from std_msgs.msg import ColorRGBA

import tf2_ros
from tf2_ros import Buffer, TransformListener

# For semi-goal sequential navigation
from rclpy.action import ActionClient
try:
    from nav2_msgs.action import NavigateToPose
    HAS_NAV2 = True
except ImportError:
    HAS_NAV2 = False


class DirectionalPlannerServer(Node):
    """
    Oriented A* path planner that respects legal lane direction.
    Provides /get_directional_path service and /enable_planner guard.
    """

    def __init__(self):
        super().__init__('directional_planner_server')

        # ── Declare parameters ──────────────────────────────────────────────
        self.declare_parameter('direction_penalty', 1.0e6)
        self.declare_parameter('occupancy_threshold', 0.5)
        self.declare_parameter('safety_margin_cells', 2)
        self.declare_parameter('publish_path_markers', True)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        # New: semi-goal and goal input parameters
        self.declare_parameter('goal_input_topic', '/goal_pose')
        self.declare_parameter('mission_goals_topic', '/mission_goals')
        self.declare_parameter('semi_goals_markers_topic', '/semi_goals_markers')
        self.declare_parameter('semi_goal_spacing_m', 1.0)
        self.declare_parameter('semi_goal_tolerance', 0.3)
        self.declare_parameter('auto_enable', False)
        # Ackermann rules
        self.declare_parameter('max_turn_angle_deg', 120.0)
        self.declare_parameter('right_bias_penalty', 0.5)
        self.declare_parameter('forward_lock_cells', 20)

        # ── Read parameters ─────────────────────────────────────────────────
        self.direction_penalty = self.get_parameter('direction_penalty').value
        self.occ_threshold = self.get_parameter('occupancy_threshold').value
        self.safety_margin = self.get_parameter('safety_margin_cells').value
        self.pub_markers = self.get_parameter('publish_path_markers').value
        self.map_frame = self.get_parameter('map_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        goal_input_topic = str(self.get_parameter('goal_input_topic').value)
        mission_goals_topic = str(self.get_parameter('mission_goals_topic').value)
        semi_markers_topic = str(self.get_parameter('semi_goals_markers_topic').value)
        self.semi_goal_spacing = float(self.get_parameter('semi_goal_spacing_m').value)
        self.semi_goal_tolerance = float(self.get_parameter('semi_goal_tolerance').value)
        auto_enable = bool(self.get_parameter('auto_enable').value)
        max_turn_deg = float(self.get_parameter('max_turn_angle_deg').value)
        self.max_turn_cos = math.cos(math.radians(max_turn_deg))  # dot threshold
        self.right_bias_penalty = float(self.get_parameter('right_bias_penalty').value)
        self.forward_lock_cells = int(self.get_parameter('forward_lock_cells').value)

        self.get_logger().info(
            f'Ackermann: max_turn={max_turn_deg}° (cos={self.max_turn_cos:.3f}), '
            f'right_bias={self.right_bias_penalty}, '
            f'forward_lock={self.forward_lock_cells} cells'
        )

        # ── State ───────────────────────────────────────────────────────────
        self.enabled = auto_enable  # auto-enable when running in map-loader mode
        self.latest_layers = None   # dict of layer_name → numpy array
        self.map_info = None
        self.grid_rows = 0
        self.grid_cols = 0

        # ── Semi-goal navigation state ──────────────────────────────────────
        self.semi_goals = []        # list of (x, y, yaw)
        self.current_sg_idx = 0     # index of current semi-goal
        self.main_goal = None       # (x, y) of the main goal
        self.navigating = False     # True when semi-goal navigation is active

        # ── TF2 ─────────────────────────────────────────────────────────────
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── Subscribers ─────────────────────────────────────────────────────
        self.gridmap_sub = self.create_subscription(
            GridMapMsg, '/grid_map', self.gridmap_callback, 10
        )
        # Goal input subscriber
        self.goal_sub = self.create_subscription(
            PoseStamped, goal_input_topic, self._goal_callback, 10
        )

        # ── Service: /enable_planner (SetBool) ──────────────────────────────
        self.enable_srv = self.create_service(
            SetBool, '/enable_planner', self.enable_callback
        )

        # ── Service: /get_directional_path (custom srv) ─────────────────────
        from qcar2_planner.srv import GetDirectionalPath
        self.plan_srv = self.create_service(
            GetDirectionalPath, '/get_directional_path', self.plan_callback
        )

        # ── Publishers ──────────────────────────────────────────────────────
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)
        self.arrow_pub = self.create_publisher(
            MarkerArray, '/path_arrows', 10
        )
        # New: semi-goal publishers
        self.mission_pub = self.create_publisher(
            PoseStamped, mission_goals_topic, 10
        )
        self.sg_marker_pub = self.create_publisher(
            MarkerArray, semi_markers_topic, 10
        )

        # ── Semi-goal tracking timer ────────────────────────────────────────
        self.sg_timer = self.create_timer(0.1, self._semi_goal_tick)

        self.get_logger().info(
            'DirectionalPlannerServer initialized. '
            f'Penalty={self.direction_penalty}, '
            f'occ_thresh={self.occ_threshold}, '
            f'auto_enable={auto_enable}. '
            f'Goal input: {goal_input_topic}, '
            f'Mission output: {mission_goals_topic}'
        )

    # =====================================================================
    # GridMap callback
    # =====================================================================
    def gridmap_callback(self, msg: GridMapMsg):
        """Parse and store latest GridMap layers (Eigen → numpy convention)."""
        self.map_info = msg.info
        layers = {}
        for i, name in enumerate(msg.layers):
            arr = msg.data[i]
            # dim[0] = column_index (Eigen cols = n_y)
            # dim[1] = row_index   (Eigen rows = n_x)
            if len(arr.layout.dim) >= 2:
                n_y = arr.layout.dim[0].size
                n_x = arr.layout.dim[1].size
            else:
                continue
            # Deserialize column-major → Eigen (n_x, n_y)
            eigen_data = np.array(arr.data, dtype=np.float32).reshape(
                (n_x, n_y), order='F'
            )
            # Convert Eigen → numpy: row0=maxX,col0=maxY → row0=minY,col0=minX
            data = eigen_data.T[::-1, ::-1]
            layers[name] = data

        self.latest_layers = layers
        if layers:
            sample = next(iter(layers.values()))
            self.grid_rows, self.grid_cols = sample.shape

    # =====================================================================
    # Enable/disable service
    # =====================================================================
    def enable_callback(self, request, response):
        """Handle SetBool to enable/disable the planner."""
        self.enabled = request.data
        response.success = True
        state_str = 'ENABLED' if self.enabled else 'DISABLED'
        response.message = f'Directional planner {state_str}'
        self.get_logger().info(f'Planner {state_str} via /enable_planner')
        return response

    # =====================================================================
    # Planning service — main entry point
    # =====================================================================
    def plan_callback(self, request, response):
        """
        Handle /get_directional_path service request.
        Runs oriented A* from robot's current pose to the requested goal.
        """
        # ── Guard: reject if not enabled ────────────────────────────────
        if not self.enabled:
            response.success = False
            response.message = (
                'Planner is DISABLED. Map has not been declared READY by '
                'exploration_manager_node. Wait until mapping is complete.'
            )
            self.get_logger().warn(response.message)
            return response

        # ── Guard: need map data ────────────────────────────────────────
        if self.latest_layers is None or 'occupancy' not in self.latest_layers:
            response.success = False
            response.message = 'No grid map data available yet.'
            self.get_logger().warn(response.message)
            return response

        # ── Get robot's current position from TF ────────────────────────
        try:
            t = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time()
            )
            start_x = t.transform.translation.x
            start_y = t.transform.translation.y
        except Exception as e:
            response.success = False
            response.message = f'Cannot get robot pose from TF: {e}'
            self.get_logger().error(response.message)
            return response

        # ── Goal position ───────────────────────────────────────────────
        goal_x = request.goal.pose.position.x
        goal_y = request.goal.pose.position.y

        self.get_logger().info(
            f'Planning path: ({start_x:.2f}, {start_y:.2f}) → '
            f'({goal_x:.2f}, {goal_y:.2f})'
        )

        # ── Convert to grid coordinates ─────────────────────────────────
        start_col, start_row = self._world_to_grid(start_x, start_y)
        goal_col, goal_row = self._world_to_grid(goal_x, goal_y)

        if not self._in_bounds(start_col, start_row):
            response.success = False
            response.message = f'Start position ({start_x:.2f}, {start_y:.2f}) is outside map bounds.'
            return response

        if not self._in_bounds(goal_col, goal_row):
            response.success = False
            response.message = f'Goal position ({goal_x:.2f}, {goal_y:.2f}) is outside map bounds.'
            return response

        # ── Run oriented A* ─────────────────────────────────────────────
        # Get robot yaw for Ackermann constraints
        robot_pose = self._get_robot_pose()
        start_yaw = robot_pose[2] if robot_pose else 0.0
        path_cells = self._astar_directional(
            start_col, start_row, goal_col, goal_row, start_yaw=start_yaw
        )

        if path_cells is None:
            response.success = False
            response.message = (
                'No valid path found. The goal may be unreachable while '
                'respecting the legal driving direction.'
            )
            self.get_logger().warn(response.message)
            return response

        # ── Convert path to nav_msgs/Path ───────────────────────────────
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'

        for col, row in path_cells:
            wx, wy = self._grid_to_world(col, row)
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position.x = wx
            ps.pose.position.y = wy
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            path_msg.poses.append(ps)

        # ── Set orientation along path direction ────────────────────────
        for i in range(len(path_msg.poses) - 1):
            p1 = path_msg.poses[i].pose.position
            p2 = path_msg.poses[i + 1].pose.position
            yaw = math.atan2(p2.y - p1.y, p2.x - p1.x)
            # Quaternion from yaw
            path_msg.poses[i].pose.orientation.z = math.sin(yaw / 2.0)
            path_msg.poses[i].pose.orientation.w = math.cos(yaw / 2.0)

        # Last pose gets same orientation as previous
        if len(path_msg.poses) >= 2:
            path_msg.poses[-1].pose.orientation = path_msg.poses[-2].pose.orientation

        response.path = path_msg
        response.success = True
        response.message = f'Path found with {len(path_cells)} waypoints.'

        self.get_logger().info(response.message)

        # ── Publish for visualization ───────────────────────────────────
        self.path_pub.publish(path_msg)
        if self.pub_markers:
            self._publish_path_arrows(path_msg)

        return response

    # =====================================================================
    # Oriented A* Algorithm
    # =====================================================================
    def _astar_directional(self, sc, sr, gc, gr, start_yaw=0.0):
        """
        A* search with Ackermann constraints:
          1. No U-turns: penalizes movements backwards relative to incoming dir
          2. Right-side bias: penalizes left turns to hug the right edge
          3. Traffic direction penalty (from dir_x/dir_y layers if available)

        Args:
            sc, sr: start column, row
            gc, gr: goal column, row
            start_yaw: robot heading in radians (used for initial direction)

        Returns:
            List of (col, row) tuples, or None if no path found.
        """
        occupancy = self.latest_layers['occupancy']
        dir_x = self.latest_layers.get('dir_x', np.zeros_like(occupancy))
        dir_y = self.latest_layers.get('dir_y', np.zeros_like(occupancy))

        # ── Distance-to-wall map (for right-edge attraction) ────────────
        import cv2
        free_mask_bin = (np.abs(occupancy) < 0.1).astype(np.uint8)
        # Distance from each free cell to the nearest wall
        dist_to_wall = cv2.distanceTransform(free_mask_bin, cv2.DIST_L2, 5)
        dist_to_wall = dist_to_wall.astype(np.float32)

        # ── Inflate obstacles for safety ────────────────────────────────
        obstacle_mask = (np.abs(occupancy) > 0.1).astype(np.uint8)

        self.get_logger().info(
            f'[A*] Grid {self.grid_cols}x{self.grid_rows}, '
            f'free={int(np.sum(obstacle_mask == 0))}, '
            f'blocked={int(np.sum(obstacle_mask > 0))}, '
            f'start_yaw={math.degrees(start_yaw):.1f}°',
        )

        if self.safety_margin > 0:
            k = 2 * self.safety_margin + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            obstacle_mask = cv2.dilate(obstacle_mask, kernel)

        # ── Start/goal blocked check ────────────────────────────────────
        if obstacle_mask[sr, sc] > 0:
            self.get_logger().warn('[A*] Start blocked! Clearing 3x3.')
            obstacle_mask[max(0,sr-1):min(self.grid_rows,sr+2),
                          max(0,sc-1):min(self.grid_cols,sc+2)] = 0
        if obstacle_mask[gr, gc] > 0:
            self.get_logger().warn('[A*] Goal blocked! Clearing 3x3.')
            obstacle_mask[max(0,gr-1):min(self.grid_rows,gr+2),
                          max(0,gc-1):min(self.grid_cols,gc+2)] = 0

        # ── A* data structures ──────────────────────────────────────────
        INF = float('inf')
        g_score = np.full((self.grid_rows, self.grid_cols), INF, dtype=np.float64)
        g_score[sr, sc] = 0.0

        f_start = self._heuristic(sc, sr, gc, gr)
        open_set = [(f_start, sc, sr)]
        came_from = {}
        closed = set()

        # Robot's original heading
        fwd_dx = math.cos(start_yaw)
        fwd_dy = math.sin(start_yaw)

        # Hybrid: strict heading near start, incoming-dir beyond lock radius
        # incoming_dir tracks per-cell direction for cells far from start
        incoming_dir = {}
        incoming_dir[(sc, sr)] = (fwd_dx, fwd_dy)
        lock_radius_sq = self.forward_lock_cells ** 2

        # 8-connected neighbors: (dc, dr, distance)
        SQRT2 = math.sqrt(2)
        neighbors_8 = [
            (-1, -1, SQRT2), (0, -1, 1.0), (1, -1, SQRT2),
            (-1,  0, 1.0),                  (1,  0, 1.0),
            (-1,  1, SQRT2), (0,  1, 1.0), (1,  1, SQRT2),
        ]

        iterations = 0
        max_iterations = self.grid_rows * self.grid_cols * 2

        while open_set and iterations < max_iterations:
            iterations += 1
            f, cc, cr = heapq.heappop(open_set)

            if cc == gc and cr == gr:
                # ── Reconstruct path ────────────────────────────────────
                path = [(gc, gr)]
                current = (gc, gr)
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()

                self.get_logger().info(
                    f'A* completed in {iterations} iterations, '
                    f'path length: {len(path)} cells'
                )
                return path

            if (cc, cr) in closed:
                continue
            closed.add((cc, cr))

            # Determine which direction to compare against:
            # Near start → strict robot heading (no going backwards at all)
            # Far from start → incoming direction (follow road curves)
            dist_from_start_sq = (cc - sc)**2 + (cr - sr)**2
            near_start = dist_from_start_sq <= lock_radius_sq

            if near_start:
                # STRICT: use original robot heading
                ref_dx, ref_dy = fwd_dx, fwd_dy
                # Near start: hard 90° limit (never go backwards)
                turn_threshold = 0.0  # cos(90°) = 0
            else:
                # FAR: use incoming direction, allow larger turns for road curves
                ref_dx, ref_dy = incoming_dir.get(
                    (cc, cr), (fwd_dx, fwd_dy)
                )
                turn_threshold = self.max_turn_cos

            for dc, dr, dist in neighbors_8:
                nc, nr = cc + dc, cr + dr

                if not self._in_bounds(nc, nr):
                    continue
                if obstacle_mask[nr, nc] > 0:
                    continue
                if (nc, nr) in closed:
                    continue

                # ── Movement direction (normalized) ─────────────────────
                vx = float(dc)
                vy = float(dr)
                v_norm = math.sqrt(vx * vx + vy * vy)
                if v_norm > 0:
                    vx /= v_norm
                    vy /= v_norm

                # ── Rule 1: HARD-BLOCK excessive turns ──────────────────
                dot_forward = vx * ref_dx + vy * ref_dy
                if dot_forward < turn_threshold:
                    continue  # BLOCKED

                penalty = 0.0

                # ── Rule 2: Right-edge attraction ───────────────────────
                # 2a. Penalize left turns
                cross = ref_dx * vy - ref_dy * vx
                if cross > 0.05:
                    penalty += self.right_bias_penalty

                # 2b. Attract toward walls (prefer being close to wall)
                d = dist_to_wall[nr, nc]
                penalty += d * 0.02

                # ── Traffic direction penalty (from map layers) ─────────
                lx = float(dir_x[cr, cc])
                ly = float(dir_y[cr, cc])
                l_norm = math.sqrt(lx * lx + ly * ly)
                if l_norm > 0.01:
                    dot = vx * lx + vy * ly
                    if dot < 0:
                        penalty += self.direction_penalty

                # ── Total edge cost ─────────────────────────────────────
                edge_cost = dist * self.map_info.resolution + penalty

                tentative_g = g_score[cr, cc] + edge_cost
                if tentative_g < g_score[nr, nc]:
                    g_score[nr, nc] = tentative_g
                    came_from[(nc, nr)] = (cc, cr)
                    incoming_dir[(nc, nr)] = (vx, vy)
                    f_score = tentative_g + self._heuristic(nc, nr, gc, gr)
                    heapq.heappush(open_set, (f_score, nc, nr))

        self.get_logger().warn(
            f'A* exhausted search after {iterations} iterations. No path found.'
        )
        return None

    # =====================================================================
    # A* heuristic — Euclidean distance in meters
    # =====================================================================
    def _heuristic(self, c1, r1, c2, r2):
        """Euclidean distance heuristic (admissible for A*)."""
        dx = (c2 - c1) * self.map_info.resolution
        dy = (r2 - r1) * self.map_info.resolution
        return math.sqrt(dx * dx + dy * dy)

    # =====================================================================
    # Coordinate conversions
    # =====================================================================
    def _world_to_grid(self, wx, wy):
        """Convert world → grid. Corner = centre - half-length."""
        res = self.map_info.resolution
        corner_x = self.map_info.pose.position.x - self.map_info.length_x / 2.0
        corner_y = self.map_info.pose.position.y - self.map_info.length_y / 2.0
        col = int((wx - corner_x) / res)
        row = int((wy - corner_y) / res)
        return col, row

    def _grid_to_world(self, col, row):
        """Convert grid → world (cell centre). Corner = centre - half-length."""
        res = self.map_info.resolution
        corner_x = self.map_info.pose.position.x - self.map_info.length_x / 2.0
        corner_y = self.map_info.pose.position.y - self.map_info.length_y / 2.0
        wx = corner_x + (col + 0.5) * res
        wy = corner_y + (row + 0.5) * res
        return float(wx), float(wy)

    def _in_bounds(self, col, row):
        return 0 <= col < self.grid_cols and 0 <= row < self.grid_rows

    # =====================================================================
    # Path visualization — arrow markers for RViz2
    # =====================================================================
    def _publish_path_arrows(self, path_msg: Path):
        """
        Publish the planned path as an arrow MarkerArray in RViz2.
        Each arrow shows the driving direction at that waypoint.
        """
        marker_array = MarkerArray()

        # Clear previous markers
        clear = Marker()
        clear.header = path_msg.header
        clear.ns = 'path_arrows'
        clear.id = 0
        clear.action = Marker.DELETEALL
        marker_array.markers.append(clear)

        for i, pose in enumerate(path_msg.poses):
            arrow = Marker()
            arrow.header = path_msg.header
            arrow.ns = 'path_arrows'
            arrow.id = i + 1
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD

            arrow.pose = pose.pose
            arrow.scale.x = 0.15  # shaft length
            arrow.scale.y = 0.04  # shaft width
            arrow.scale.z = 0.04  # head width

            # Green to blue gradient along path
            t = i / max(len(path_msg.poses) - 1, 1)
            arrow.color = ColorRGBA(
                r=0.0,
                g=float(1.0 - t),
                b=float(t),
                a=0.9
            )

            arrow.lifetime.sec = 30
            marker_array.markers.append(arrow)

        self.arrow_pub.publish(marker_array)
        self.get_logger().info(
            f'Published {len(path_msg.poses)} path arrow markers'
        )


    # =====================================================================
    # Goal topic callback — receive main goal, plan A*, generate semi-goals
    # =====================================================================
    def _goal_callback(self, msg: PoseStamped):
        """Handle incoming goal from /goal_pose topic."""
        if self.latest_layers is None or 'occupancy' not in self.latest_layers:
            self.get_logger().warn('No map data yet, ignoring goal.')
            return

        if not self.enabled:
            self.get_logger().warn('Planner not enabled, ignoring goal.')
            return

        goal_x = msg.pose.position.x
        goal_y = msg.pose.position.y

        # Get robot pose from TF
        robot_pose = self._get_robot_pose()
        if robot_pose is None:
            self.get_logger().error('Cannot get robot pose from TF.')
            return

        start_x, start_y = robot_pose[0], robot_pose[1]

        self.get_logger().info(
            f'Goal received: ({goal_x:.2f}, {goal_y:.2f}) from '
            f'({start_x:.2f}, {start_y:.2f})'
        )

        # Convert to grid
        start_col, start_row = self._world_to_grid(start_x, start_y)
        goal_col, goal_row = self._world_to_grid(goal_x, goal_y)

        # ── Diagnostics ─────────────────────────────────────────────────
        occupancy = self.latest_layers['occupancy']

        # Dump map_info for coordinate tracing
        if self.map_info is not None:
            cx = self.map_info.pose.position.x
            cy = self.map_info.pose.position.y
            lx = self.map_info.length_x
            ly = self.map_info.length_y
            res = self.map_info.resolution
            corner_x = cx - lx / 2.0
            corner_y = cy - ly / 2.0
            self.get_logger().info(
                f'[MAP_INFO] center=({cx:.2f},{cy:.2f}) size=({lx:.1f},{ly:.1f}) '
                f'res={res:.3f} corner=({corner_x:.2f},{corner_y:.2f})'
            )

        if self._in_bounds(start_col, start_row):
            start_occ = occupancy[start_row, start_col]
        else:
            start_occ = float('nan')
        if self._in_bounds(goal_col, goal_row):
            goal_occ = occupancy[goal_row, goal_col]
        else:
            goal_occ = float('nan')

        self.get_logger().info(
            f'[DIAG] start_grid=({start_col},{start_row}) occ={start_occ:.3f}, '
            f'goal_grid=({goal_col},{goal_row}) occ={goal_occ:.3f}, '
            f'grid_size={self.grid_cols}x{self.grid_rows}'
        )

        # Map content analysis
        n_free = int(np.sum(np.abs(occupancy) < 0.1))
        n_wall = int(np.sum(occupancy > 0.5))
        n_unk = int(np.sum(occupancy < -0.5))
        self.get_logger().info(
            f'[DIAG] occupancy stats: free(≈0)={n_free}, wall(≈1)={n_wall}, '
            f'unknown(≈-1)={n_unk}, unique={np.unique(occupancy).tolist()}'
        )

        # Find where free cells actually are (first/last free row and col)
        free_mask = np.abs(occupancy) < 0.1
        free_rows = np.any(free_mask, axis=1)
        free_cols = np.any(free_mask, axis=0)
        if np.any(free_rows):
            first_free_row = int(np.argmax(free_rows))
            last_free_row = int(len(free_rows) - 1 - np.argmax(free_rows[::-1]))
            first_free_col = int(np.argmax(free_cols))
            last_free_col = int(len(free_cols) - 1 - np.argmax(free_cols[::-1]))
            self.get_logger().info(
                f'[DIAG] Free area rows=[{first_free_row}..{last_free_row}], '
                f'cols=[{first_free_col}..{last_free_col}]'
            )
        else:
            self.get_logger().error('[DIAG] NO FREE CELLS IN MAP!')

        if not self._in_bounds(start_col, start_row):
            self.get_logger().error(
                f'Start ({start_x:.2f},{start_y:.2f}) outside map bounds '
                f'grid=({start_col},{start_row})'
            )
            return
        if not self._in_bounds(goal_col, goal_row):
            self.get_logger().error(
                f'Goal ({goal_x:.2f},{goal_y:.2f}) outside map bounds '
                f'grid=({goal_col},{goal_row})'
            )
            return

        # Run A* with robot yaw for Ackermann constraints
        start_yaw = robot_pose[2] if robot_pose is not None else 0.0
        path_cells = self._astar_directional(start_col, start_row,
                                              goal_col, goal_row,
                                              start_yaw=start_yaw)
        if path_cells is None:
            self.get_logger().error('No path found to goal.')
            return

        # Build nav_msgs/Path
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = self.map_frame

        for col, row in path_cells:
            wx, wy = self._grid_to_world(col, row)
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position.x = wx
            ps.pose.position.y = wy
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            path_msg.poses.append(ps)

        # Set orientations along path
        for i in range(len(path_msg.poses) - 1):
            p1 = path_msg.poses[i].pose.position
            p2 = path_msg.poses[i + 1].pose.position
            yaw = math.atan2(p2.y - p1.y, p2.x - p1.x)
            path_msg.poses[i].pose.orientation.z = math.sin(yaw / 2.0)
            path_msg.poses[i].pose.orientation.w = math.cos(yaw / 2.0)
        if len(path_msg.poses) >= 2:
            path_msg.poses[-1].pose.orientation = \
                path_msg.poses[-2].pose.orientation

        # Publish full path for RViz
        self.path_pub.publish(path_msg)
        if self.pub_markers:
            self._publish_path_arrows(path_msg)

        # Generate semi-goals
        self.main_goal = (goal_x, goal_y)
        self._generate_semi_goals(path_msg)

    # =====================================================================
    # Semi-goal generation from A* path
    # =====================================================================
    def _generate_semi_goals(self, path_msg: Path):
        """Split path into semi-goals every semi_goal_spacing_m meters."""
        if not path_msg.poses:
            return

        semi_goals = []
        accumulated_dist = 0.0

        # First semi-goal: the first pose (skip robot's current position)
        # Start from index 1 to skip the robot's current location
        last_x = path_msg.poses[0].pose.position.x
        last_y = path_msg.poses[0].pose.position.y

        for i in range(1, len(path_msg.poses)):
            px = path_msg.poses[i].pose.position.x
            py = path_msg.poses[i].pose.position.y
            seg_dist = math.sqrt((px - last_x)**2 + (py - last_y)**2)
            accumulated_dist += seg_dist
            last_x, last_y = px, py

            if accumulated_dist >= self.semi_goal_spacing:
                q = path_msg.poses[i].pose.orientation
                yaw = math.atan2(
                    2.0 * (q.w * q.z + q.x * q.y),
                    1.0 - 2.0 * (q.y**2 + q.z**2),
                )
                semi_goals.append((px, py, yaw))
                accumulated_dist = 0.0

        # Always add the final goal as the last semi-goal
        last_pose = path_msg.poses[-1]
        lx = last_pose.pose.position.x
        ly = last_pose.pose.position.y
        lq = last_pose.pose.orientation
        lyaw = math.atan2(
            2.0 * (lq.w * lq.z + lq.x * lq.y),
            1.0 - 2.0 * (lq.y**2 + lq.z**2),
        )
        # Don't duplicate if the last semi-goal is very close
        if not semi_goals or math.sqrt(
            (lx - semi_goals[-1][0])**2 + (ly - semi_goals[-1][1])**2
        ) > 0.1:
            semi_goals.append((lx, ly, lyaw))

        self.semi_goals = semi_goals
        self.current_sg_idx = 0
        self.navigating = True

        self.get_logger().info(
            f'Generated {len(semi_goals)} semi-goals '
            f'(spacing={self.semi_goal_spacing}m)'
        )

        # Publish debug markers for ALL semi-goals
        self._publish_sg_markers()

        # Publish the FIRST semi-goal immediately
        self._publish_current_semi_goal()

    # =====================================================================
    # Semi-goal tick — monitor robot and advance
    # =====================================================================
    def _semi_goal_tick(self):
        """Check if robot reached current semi-goal; if so, publish next."""
        if not self.navigating or not self.semi_goals:
            return

        robot_pose = self._get_robot_pose()
        if robot_pose is None:
            return

        rx, ry = robot_pose[0], robot_pose[1]
        sx, sy, _ = self.semi_goals[self.current_sg_idx]

        dist = math.sqrt((sx - rx)**2 + (sy - ry)**2)

        if dist <= self.semi_goal_tolerance:
            self.get_logger().info(
                f'Semi-goal {self.current_sg_idx + 1}/'
                f'{len(self.semi_goals)} reached (dist={dist:.3f}m)'
            )
            self.current_sg_idx += 1

            if self.current_sg_idx >= len(self.semi_goals):
                # All semi-goals completed
                self.navigating = False
                self.get_logger().info(
                    '✅ All semi-goals reached. Navigation complete!'
                )
                return

            # Publish next semi-goal
            self._publish_current_semi_goal()
            # Update markers (color changes)
            self._publish_sg_markers()

    def _publish_current_semi_goal(self):
        """Publish current semi-goal as PoseStamped on /mission_goals."""
        if self.current_sg_idx >= len(self.semi_goals):
            return

        sx, sy, syaw = self.semi_goals[self.current_sg_idx]

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        msg.pose.position.x = float(sx)
        msg.pose.position.y = float(sy)
        msg.pose.position.z = 0.0
        msg.pose.orientation.z = math.sin(syaw / 2.0)
        msg.pose.orientation.w = math.cos(syaw / 2.0)

        self.mission_pub.publish(msg)

        self.get_logger().info(
            f'→ Published semi-goal {self.current_sg_idx + 1}/'
            f'{len(self.semi_goals)}: ({sx:.2f}, {sy:.2f})'
        )

    # =====================================================================
    # Debug markers for semi-goals
    # =====================================================================
    def _publish_sg_markers(self):
        """Publish all semi-goals as colored spheres in RViz2."""
        marker_array = MarkerArray()

        # Clear previous
        clear = Marker()
        clear.header.stamp = self.get_clock().now().to_msg()
        clear.header.frame_id = self.map_frame
        clear.ns = 'semi_goals'
        clear.id = 0
        clear.action = Marker.DELETEALL
        marker_array.markers.append(clear)

        for i, (sx, sy, syaw) in enumerate(self.semi_goals):
            m = Marker()
            m.header.stamp = self.get_clock().now().to_msg()
            m.header.frame_id = self.map_frame
            m.ns = 'semi_goals'
            m.id = i + 1
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(sx)
            m.pose.position.y = float(sy)
            m.pose.position.z = 0.3
            m.scale.x = 0.25
            m.scale.y = 0.25
            m.scale.z = 0.25

            if i < self.current_sg_idx:
                # Reached → dim green
                m.color = ColorRGBA(r=0.3, g=0.7, b=0.3, a=0.4)
            elif i == self.current_sg_idx:
                # Current → bright cyan
                m.color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0)
            elif i == len(self.semi_goals) - 1:
                # Final goal → red
                m.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.9)
            else:
                # Pending → blue
                m.color = ColorRGBA(r=0.2, g=0.4, b=1.0, a=0.7)

            m.lifetime.sec = 0  # persistent
            marker_array.markers.append(m)

        self.sg_marker_pub.publish(marker_array)

    # =====================================================================
    # Robot pose from TF
    # =====================================================================
    def _get_robot_pose(self):
        """Get robot (x, y, yaw) in map frame."""
        try:
            t = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time()
            )
            x = t.transform.translation.x
            y = t.transform.translation.y
            q = t.transform.rotation
            yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y**2 + q.z**2),
            )
            return x, y, yaw
        except Exception:
            return None


def main(args=None):
    rclpy.init(args=args)
    node = DirectionalPlannerServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
