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

        # ── Read parameters ─────────────────────────────────────────────────
        self.direction_penalty = self.get_parameter('direction_penalty').value
        self.occ_threshold = self.get_parameter('occupancy_threshold').value
        self.safety_margin = self.get_parameter('safety_margin_cells').value
        self.pub_markers = self.get_parameter('publish_path_markers').value
        self.map_frame = self.get_parameter('map_frame').value
        self.base_frame = self.get_parameter('base_frame').value

        # ── State ───────────────────────────────────────────────────────────
        self.enabled = False      # Set to True when map is READY (via SetBool)
        self.latest_layers = None # dict of layer_name → numpy array
        self.map_info = None
        self.grid_rows = 0
        self.grid_cols = 0

        # ── TF2 ─────────────────────────────────────────────────────────────
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── Subscribers ─────────────────────────────────────────────────────
        self.gridmap_sub = self.create_subscription(
            GridMapMsg, '/grid_map', self.gridmap_callback, 10
        )

        # ── Service: /enable_planner (SetBool) ──────────────────────────────
        self.enable_srv = self.create_service(
            SetBool, '/enable_planner', self.enable_callback
        )

        # ── Service: /get_directional_path (custom srv) ─────────────────────
        # Import the auto-generated service type
        from qcar2_planner.srv import GetDirectionalPath
        self.plan_srv = self.create_service(
            GetDirectionalPath, '/get_directional_path', self.plan_callback
        )

        # ── Publishers ──────────────────────────────────────────────────────
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)
        self.arrow_pub = self.create_publisher(
            MarkerArray, '/path_arrows', 10
        )

        self.get_logger().info(
            'DirectionalPlannerServer initialized. '
            f'Penalty={self.direction_penalty}, '
            f'occ_thresh={self.occ_threshold}. '
            'Waiting for /enable_planner...'
        )

    # =====================================================================
    # GridMap callback
    # =====================================================================
    def gridmap_callback(self, msg: GridMapMsg):
        """Parse and store latest GridMap layers (column-major convention)."""
        self.map_info = msg.info
        layers = {}
        for i, name in enumerate(msg.layers):
            arr = msg.data[i]
            # grid_map convention: dim[0] = column_index, dim[1] = row_index
            if len(arr.layout.dim) >= 2:
                cols = arr.layout.dim[0].size  # column_index
                rows = arr.layout.dim[1].size  # row_index
            else:
                continue
            data = np.array(arr.data, dtype=np.float32).reshape(
                (rows, cols), order='F'
            )
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
        path_cells = self._astar_directional(
            start_col, start_row, goal_col, goal_row
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
    def _astar_directional(self, sc, sr, gc, gr):
        """
        A* search on the 2D grid with directional cost penalty.

        The cost to move from cell A=(ac,ar) to cell B=(bc,br) is:

            Cost(A, B) = dist(A, B) + P

        Where P is the dot-product penalty:
            V_AB = normalize(B - A)       # movement direction
            L    = (dir_x[A], dir_y[A])   # legal lane direction

            dot  = V_AB · L = Vx*Lx + Vy*Ly

            P = 0                if dot >= 0  (aligned with traffic)
            P = direction_penalty if dot <  0  (against traffic)

        This blocks wrong-way paths while allowing any forward-aligned movement.

        Args:
            sc, sr: start column, row
            gc, gr: goal column, row

        Returns:
            List of (col, row) tuples, or None if no path found.
        """
        occupancy = self.latest_layers['occupancy']
        dir_x = self.latest_layers.get('dir_x', np.zeros_like(occupancy))
        dir_y = self.latest_layers.get('dir_y', np.zeros_like(occupancy))

        # ── Inflate obstacles for safety ────────────────────────────────
        # New occupancy: -1=unknown, 0=free, 1=wall → block everything != 0
        import cv2
        obstacle_mask = (occupancy != 0.0).astype(np.uint8)
        if self.safety_margin > 0:
            k = 2 * self.safety_margin + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            obstacle_mask = cv2.dilate(obstacle_mask, kernel)

        # ── A* data structures ──────────────────────────────────────────
        # open_set: priority queue of (f_score, col, row)
        # g_score: best known cost from start to (col, row)
        # came_from: backtracking dict
        INF = float('inf')
        g_score = np.full((self.grid_rows, self.grid_cols), INF, dtype=np.float64)
        g_score[sr, sc] = 0.0

        f_start = self._heuristic(sc, sr, gc, gr)
        open_set = [(f_start, sc, sr)]
        came_from = {}
        closed = set()

        # 8-connected neighbors: (dc, dr, distance)
        neighbors_8 = [
            (-1, -1, math.sqrt(2)), (0, -1, 1.0), (1, -1, math.sqrt(2)),
            (-1,  0, 1.0),                          (1,  0, 1.0),
            (-1,  1, math.sqrt(2)), (0,  1, 1.0), (1,  1, math.sqrt(2)),
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

            for dc, dr, dist in neighbors_8:
                nc, nr = cc + dc, cr + dr

                if not self._in_bounds(nc, nr):
                    continue
                if obstacle_mask[nr, nc] > 0:
                    continue
                if (nc, nr) in closed:
                    continue

                # ── Directional penalty via dot product ─────────────────
                # Movement vector V_AB from current (cc,cr) to neighbor (nc,nr)
                vx = float(dc)
                vy = float(dr)
                v_norm = math.sqrt(vx * vx + vy * vy)
                if v_norm > 0:
                    vx /= v_norm
                    vy /= v_norm

                # Legal direction L at current cell
                lx = float(dir_x[cr, cc])
                ly = float(dir_y[cr, cc])

                # Dot product: V_AB · L = |V||L|cos(θ)
                # If dot >= 0: aligned with traffic (P = 0)
                # If dot <  0: against traffic (P = penalty ≈ ∞)
                penalty = 0.0
                l_norm = math.sqrt(lx * lx + ly * ly)
                if l_norm > 0.01:  # Only penalize where direction is known
                    dot = vx * lx + vy * ly
                    if dot < 0:
                        penalty = self.direction_penalty

                # ── Total edge cost ─────────────────────────────────────
                # Cost = Euclidean distance (in cells) + direction penalty
                edge_cost = dist * self.map_info.resolution + penalty

                tentative_g = g_score[cr, cc] + edge_cost
                if tentative_g < g_score[nr, nc]:
                    g_score[nr, nc] = tentative_g
                    came_from[(nc, nr)] = (cc, cr)
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
