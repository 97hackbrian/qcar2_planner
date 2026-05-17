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

import os
import yaml
import heapq
import math
import time
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

# ── Numba (optional JIT acceleration) ──────────────────────────────────────
# NOTE: First execution with Numba will be slower due to JIT compilation.
#       Subsequent calls use the cached compiled version and are much faster.
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False


# =========================================================================
# Pure-numerical A* core — Numba-compatible
# =========================================================================
# This function contains ONLY numeric operations on numpy arrays.
# No self, no ROS, no strings, no dicts, no yaml, no logging.
#
# Lane encoding:
#   lane_id = -2  → GLOBAL mode (single-lane, uses global dir_x/dir_y)
#   lane_id = -1  → NONE (start is outside all lanes)
#   lane_id >= 0  → index into lane_masks_3d / lane_dir_x_3d / lane_dir_y_3d
#
# successor_matrix[i, j] = 1  means lane i can transition to lane j
# =========================================================================
def _astar_core_python(
    obstacle_mask,        # uint8   [rows, cols]
    grad_col,             # float32 [rows, cols]
    grad_row,             # float32 [rows, cols]
    global_dir_x,         # float32 [rows, cols]
    global_dir_y,         # float32 [rows, cols]
    global_lane_mask,     # float32 [rows, cols] or None → zeros
    lane_masks_3d,        # float32 [N, rows, cols] — per-lane masks (empty if GLOBAL)
    lane_dir_x_3d,        # float32 [N, rows, cols]
    lane_dir_y_3d,        # float32 [N, rows, cols]
    successor_matrix,     # int8    [N, N]
    start_lane_ids,       # int32   [S]  — lane ids for start cell
    sc, sr, gc, gr,       # int32 scalars
    fwd_dx, fwd_dy,       # float64 — initial heading
    resolution,           # float64
    min_direction_dot,    # float64
    direction_penalty,    # float64
    max_turn_cos,         # float64
    right_bias_penalty,   # float64
    lock_radius_sq,       # int32
    enforce_lane_mask,    # bool
    enforce_lane_successors,  # bool
    rows, cols,           # int32
):
    """
    Pure-numerical A* core. Returns (path_cols, path_rows, iterations) or
    (empty, empty, iterations) if no path found.
    """
    INF = 1e18
    SQRT2 = 1.4142135623730951
    n_lanes = lane_masks_3d.shape[0]

    # 8-connected neighbors: (dc, dr, distance)
    nbr_dc = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.int32)
    nbr_dr = np.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=np.int32)
    nbr_dist = np.array([SQRT2, 1.0, SQRT2, 1.0, 1.0, SQRT2, 1.0, SQRT2], dtype=np.float64)

    # Total lane slots: actual lanes + 2 special (GLOBAL=-2→slot n_lanes, NONE=-1→slot n_lanes+1)
    total_slots = n_lanes + 2
    GLOBAL_SLOT = n_lanes
    NONE_SLOT = n_lanes + 1

    # g_score as 3D array [rows, cols, total_slots]
    g_score = np.full((rows, cols, total_slots), INF, dtype=np.float64)

    # came_from as 3D arrays (store parent col, row, lane_slot)
    parent_c = np.full((rows, cols, total_slots), -1, dtype=np.int32)
    parent_r = np.full((rows, cols, total_slots), -1, dtype=np.int32)
    parent_l = np.full((rows, cols, total_slots), -1, dtype=np.int32)

    # incoming direction per state
    inc_dx = np.zeros((rows, cols, total_slots), dtype=np.float64)
    inc_dy = np.zeros((rows, cols, total_slots), dtype=np.float64)

    # closed set as 3D bool
    closed = np.zeros((rows, cols, total_slots), dtype=np.bool_)

    # Map lane_id to slot
    # lane_id >= 0 → slot = lane_id
    # lane_id == -2 → GLOBAL_SLOT
    # lane_id == -1 → NONE_SLOT

    # Manual binary heap (Numba can't use heapq)
    # Heap entries: (f_score, col, row, lane_slot)
    MAX_HEAP = rows * cols * 4
    heap_f = np.full(MAX_HEAP, INF, dtype=np.float64)
    heap_c = np.zeros(MAX_HEAP, dtype=np.int32)
    heap_r = np.zeros(MAX_HEAP, dtype=np.int32)
    heap_l = np.zeros(MAX_HEAP, dtype=np.int32)
    heap_size = 0

    # Initialize start states
    for si in range(start_lane_ids.shape[0]):
        lid = start_lane_ids[si]
        if lid == -2:
            slot = GLOBAL_SLOT
        elif lid == -1:
            slot = NONE_SLOT
        else:
            slot = lid

        g_score[sr, sc, slot] = 0.0
        inc_dx[sr, sc, slot] = fwd_dx
        inc_dy[sr, sc, slot] = fwd_dy

        dx_h = float(gc - sc) * resolution
        dy_h = float(gr - sr) * resolution
        f_val = math.sqrt(dx_h * dx_h + dy_h * dy_h)

        if heap_size < MAX_HEAP:
            heap_f[heap_size] = f_val
            heap_c[heap_size] = sc
            heap_r[heap_size] = sr
            heap_l[heap_size] = slot
            # Sift up
            i = heap_size
            while i > 0:
                p = (i - 1) // 2
                if heap_f[i] < heap_f[p]:
                    heap_f[i], heap_f[p] = heap_f[p], heap_f[i]
                    heap_c[i], heap_c[p] = heap_c[p], heap_c[i]
                    heap_r[i], heap_r[p] = heap_r[p], heap_r[i]
                    heap_l[i], heap_l[p] = heap_l[p], heap_l[i]
                    i = p
                else:
                    break
            heap_size += 1

    iterations = 0
    max_iterations = rows * cols * 2

    while heap_size > 0 and iterations < max_iterations:
        iterations += 1

        # Pop minimum from heap
        f_val = heap_f[0]
        cc = heap_c[0]
        cr = heap_r[0]
        c_slot = heap_l[0]

        # Replace root with last element and sift down
        heap_size -= 1
        if heap_size > 0:
            heap_f[0] = heap_f[heap_size]
            heap_c[0] = heap_c[heap_size]
            heap_r[0] = heap_r[heap_size]
            heap_l[0] = heap_l[heap_size]
            # Sift down
            i = 0
            while True:
                left = 2 * i + 1
                right_child = 2 * i + 2
                smallest = i
                if left < heap_size and heap_f[left] < heap_f[smallest]:
                    smallest = left
                if right_child < heap_size and heap_f[right_child] < heap_f[smallest]:
                    smallest = right_child
                if smallest != i:
                    heap_f[i], heap_f[smallest] = heap_f[smallest], heap_f[i]
                    heap_c[i], heap_c[smallest] = heap_c[smallest], heap_c[i]
                    heap_r[i], heap_r[smallest] = heap_r[smallest], heap_r[i]
                    heap_l[i], heap_l[smallest] = heap_l[smallest], heap_l[i]
                    i = smallest
                else:
                    break

        # Goal check
        if cc == gc and cr == gr:
            # Reconstruct path
            path_c_buf = np.empty(rows * cols, dtype=np.int32)
            path_r_buf = np.empty(rows * cols, dtype=np.int32)
            path_len = 0
            cur_c, cur_r, cur_l = cc, cr, c_slot
            while cur_c != -1:
                path_c_buf[path_len] = cur_c
                path_r_buf[path_len] = cur_r
                path_len += 1
                pc = parent_c[cur_r, cur_c, cur_l]
                pr = parent_r[cur_r, cur_c, cur_l]
                pl = parent_l[cur_r, cur_c, cur_l]
                cur_c, cur_r, cur_l = pc, pr, pl
            # Reverse
            path_cols = np.empty(path_len, dtype=np.int32)
            path_rows = np.empty(path_len, dtype=np.int32)
            for pi in range(path_len):
                path_cols[pi] = path_c_buf[path_len - 1 - pi]
                path_rows[pi] = path_r_buf[path_len - 1 - pi]
            return path_cols, path_rows, iterations

        if closed[cr, cc, c_slot]:
            continue
        closed[cr, cc, c_slot] = True

        # Direction reference
        dist_from_start_sq = (cc - sc) * (cc - sc) + (cr - sr) * (cr - sr)
        near_start = dist_from_start_sq <= lock_radius_sq

        if near_start:
            ref_dx = fwd_dx
            ref_dy = fwd_dy
            turn_threshold = 0.0
        else:
            ref_dx = inc_dx[cr, cc, c_slot]
            ref_dy = inc_dy[cr, cc, c_slot]
            if ref_dx == 0.0 and ref_dy == 0.0:
                ref_dx = fwd_dx
                ref_dy = fwd_dy
            turn_threshold = max_turn_cos

        # Build valid next lane slots
        # We store them in a small fixed-size array
        valid_slots = np.empty(total_slots, dtype=np.int32)
        n_valid = 0

        if c_slot == GLOBAL_SLOT:
            valid_slots[0] = GLOBAL_SLOT
            n_valid = 1
        elif c_slot == NONE_SLOT:
            valid_slots[0] = NONE_SLOT
            n_valid = 1
            # Also add any lane whose mask covers current cell
            for li in range(n_lanes):
                if lane_masks_3d[li, cr, cc] > 0.5:
                    valid_slots[n_valid] = li
                    n_valid += 1
        else:
            valid_slots[0] = c_slot
            n_valid = 1
            if enforce_lane_successors:
                for li in range(n_lanes):
                    if successor_matrix[c_slot, li] > 0:
                        valid_slots[n_valid] = li
                        n_valid += 1
            else:
                for li in range(n_lanes):
                    if li != c_slot:
                        valid_slots[n_valid] = li
                        n_valid += 1

        # Expand 8 neighbors
        for ni in range(8):
            dc = nbr_dc[ni]
            dr = nbr_dr[ni]
            step_dist = nbr_dist[ni]
            nc = cc + dc
            nr = cr + dr

            if nc < 0 or nc >= cols or nr < 0 or nr >= rows:
                continue
            if obstacle_mask[nr, nc] > 0:
                continue

            # Movement direction (normalized)
            vx = float(dc)
            vy = float(dr)
            v_norm = math.sqrt(vx * vx + vy * vy)
            if v_norm > 0.0:
                vx /= v_norm
                vy /= v_norm

            # Turn constraint
            dot_forward = vx * ref_dx + vy * ref_dy
            if dot_forward < turn_threshold:
                continue

            # Right-wall attraction penalty
            right_vx = vy
            right_vy = -vx
            gx = float(grad_col[nr, nc])
            gy = float(grad_row[nr, nc])
            right_dot = gx * right_vx + gy * right_vy
            base_penalty = 0.0
            tmp = right_dot + 0.3
            if tmp > 0.0:
                base_penalty = tmp * right_bias_penalty

            # Try each valid next lane
            for vi in range(n_valid):
                n_slot = valid_slots[vi]
                if closed[nr, nc, n_slot]:
                    continue

                lx = 0.0
                ly = 0.0
                penalty = base_penalty
                skip = False

                if n_slot == GLOBAL_SLOT:
                    if enforce_lane_mask:
                        if global_lane_mask[nr, nc] < 0.5:
                            skip = True
                    if not skip:
                        lx = float(global_dir_x[cr, cc])
                        ly = float(global_dir_y[cr, cc])
                elif n_slot != NONE_SLOT:
                    # It's a real lane index
                    if lane_masks_3d[n_slot, nr, nc] < 0.5:
                        skip = True
                    if not skip:
                        lx = float(lane_dir_x_3d[n_slot, nr, nc])
                        ly = float(lane_dir_y_3d[n_slot, nr, nc])

                if skip:
                    continue

                l_norm = math.sqrt(lx * lx + ly * ly)
                if l_norm > 0.01:
                    dot_lane = vx * lx + vy * ly
                    if dot_lane < min_direction_dot:
                        continue
                    if dot_lane < 0.0:
                        penalty += direction_penalty

                edge_cost = step_dist * resolution + penalty
                tentative_g = g_score[cr, cc, c_slot] + edge_cost

                if tentative_g < g_score[nr, nc, n_slot]:
                    g_score[nr, nc, n_slot] = tentative_g
                    parent_c[nr, nc, n_slot] = cc
                    parent_r[nr, nc, n_slot] = cr
                    parent_l[nr, nc, n_slot] = c_slot
                    inc_dx[nr, nc, n_slot] = vx
                    inc_dy[nr, nc, n_slot] = vy

                    # Heuristic
                    dx_h = float(gc - nc) * resolution
                    dy_h = float(gr - nr) * resolution
                    f_new = tentative_g + math.sqrt(dx_h * dx_h + dy_h * dy_h)

                    # Push to heap
                    if heap_size < MAX_HEAP:
                        heap_f[heap_size] = f_new
                        heap_c[heap_size] = nc
                        heap_r[heap_size] = nr
                        heap_l[heap_size] = n_slot
                        # Sift up
                        i = heap_size
                        while i > 0:
                            p = (i - 1) // 2
                            if heap_f[i] < heap_f[p]:
                                heap_f[i], heap_f[p] = heap_f[p], heap_f[i]
                                heap_c[i], heap_c[p] = heap_c[p], heap_c[i]
                                heap_r[i], heap_r[p] = heap_r[p], heap_r[i]
                                heap_l[i], heap_l[p] = heap_l[p], heap_l[i]
                                i = p
                            else:
                                break
                        heap_size += 1

    # No path found
    return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32), iterations


# Attempt to JIT-compile the core function
if NUMBA_AVAILABLE:
    try:
        _astar_core_numba = njit(cache=True)(_astar_core_python)
    except Exception:
        _astar_core_numba = None
        NUMBA_AVAILABLE = False
else:
    _astar_core_numba = None


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
        self.declare_parameter('enforce_lane_mask', False)
        self.declare_parameter('min_direction_dot', 0.0)
        self.declare_parameter('max_turn_angle_deg', 120.0)
        self.declare_parameter('right_bias_penalty', 0.5)
        self.declare_parameter('forward_lock_cells', 20)
        
        # Independent Lanes
        self.declare_parameter('use_independent_lanes', True)
        self.declare_parameter('enforce_lane_successors', True)
        self.declare_parameter('connection_threshold_m', 0.30)
        self.declare_parameter('lanes_yaml_path', '')

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
        self.enforce_lane_mask = bool(self.get_parameter('enforce_lane_mask').value)
        self.min_direction_dot = float(self.get_parameter('min_direction_dot').value)
        self.right_bias_penalty = float(self.get_parameter('right_bias_penalty').value)
        self.forward_lock_cells = int(self.get_parameter('forward_lock_cells').value)
        
        self.use_independent_lanes = bool(self.get_parameter('use_independent_lanes').value)
        self.enforce_lane_successors = bool(self.get_parameter('enforce_lane_successors').value)
        self.connection_threshold_m = float(self.get_parameter('connection_threshold_m').value)
        raw_lanes_path = str(self.get_parameter('lanes_yaml_path').value)
        self.lanes_yaml_path = os.path.abspath(
            os.path.expanduser(
                raw_lanes_path.strip().strip('"').strip("'")
            )
        )
        self.get_logger().info(f"[LANES] lanes_yaml_path={repr(self.lanes_yaml_path)}")

        self.get_logger().info(
            f'Lane Enforcement: mask={self.enforce_lane_mask}, min_dot={self.min_direction_dot}\n'
            f'Independent Lanes: {self.use_independent_lanes}, enforce successors: {self.enforce_lane_successors}\n'
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
        
        self.lane_grids = {} # internal_id -> {mask, dir_x, dir_y, successors, points, name}
        self.lanes_loaded = False

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
            
            if self.use_independent_lanes and not self.lanes_loaded:
                self._load_independent_lanes()

    # =====================================================================
    # Load independent lanes
    # =====================================================================
    def _load_independent_lanes(self):
        import cv2
        
        path = os.path.abspath(
            os.path.expanduser(
                str(self.lanes_yaml_path).strip().strip('"').strip("'")
            )
        )

        self.get_logger().info(f"[LANES] Trying to load lanes_yaml_path={repr(path)}")
        self.get_logger().info(f"[LANES] exists={os.path.exists(path)}, isfile={os.path.isfile(path)}")
        self.get_logger().info(f"[LANES] cwd={os.getcwd()}")

        if not path:
            self.get_logger().warn("[LANES] lanes_yaml_path parameter is empty.")
            return

        if not os.path.isfile(path):
            fallback = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "config", "lanes.yaml")
            )
            self.get_logger().warn(f"[LANES] Primary path not found: {repr(path)}")
            self.get_logger().warn(f"[LANES] Trying fallback path: {repr(fallback)}")
            self.get_logger().warn(f"[LANES] fallback exists={os.path.exists(fallback)}, isfile={os.path.isfile(fallback)}")

            if os.path.isfile(fallback):
                path = fallback
            else:
                self.get_logger().warn(f"[LANES] No lanes.yaml found. Primary={repr(path)} fallback={repr(fallback)}")
                return

        try:
            with open(path, "r") as f:
                lanes_data = yaml.safe_load(f) or {}
        except Exception as e:
            self.get_logger().error(f"[LANES] Failed to read/parse lanes.yaml: {e}")
            return

        if "lanes" not in lanes_data or not isinstance(lanes_data["lanes"], list):
            self.get_logger().warn("[LANES] lanes.yaml loaded but has no valid 'lanes' list.")
            return

        lanes = lanes_data["lanes"]
        self.get_logger().info(f"[LANES] Loaded {len(lanes)} lanes from YAML.")
        
        res = self.map_info.resolution
        origin_x = self.map_info.pose.position.x - self.map_info.length_x / 2.0
        origin_y = self.map_info.pose.position.y - self.map_info.length_y / 2.0
        
        name_to_id = {}
        parsed_lanes = []
        
        # 1) Parse lanes and build grids
        for idx, lane in enumerate(lanes):
            internal_id = f"lane_{idx + 1:03d}"
            original_name = lane.get('name', '')
            name = original_name if original_name else internal_id
            
            width = float(lane.get('width', 0.0))
            points = lane.get('points', [])
            manual_successors = lane.get('successors', [])
            
            if not points or width <= 0.0:
                continue
                
            name_to_id[name] = internal_id
            if original_name:
                name_to_id[original_name] = internal_id
                
            width_px = width / res
            
            mask = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)
            dir_x = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)
            dir_y = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)
            
            for i in range(len(points) - 1):
                p1 = points[i]
                p2 = points[i+1]
                
                c1 = int((p1[0] - origin_x) / res - 0.5)
                r1 = int((p1[1] - origin_y) / res - 0.5)
                c2 = int((p2[0] - origin_x) / res - 0.5)
                r2 = int((p2[1] - origin_y) / res - 0.5)
                
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                mag = math.sqrt(dx*dx + dy*dy)
                if mag < 1e-6:
                    continue
                ux = dx / mag
                uy = dy / mag
                
                thickness = max(1, int(width_px))
                cv2.line(mask, (c1, r1), (c2, r2), 1.0, thickness)
                cv2.line(dir_x, (c1, r1), (c2, r2), float(ux), thickness)
                cv2.line(dir_y, (c1, r1), (c2, r2), float(uy), thickness)
            
            parsed_lanes.append({
                'id': internal_id,
                'name': name,
                'original_name': original_name,
                'points': points,
                'mask': mask,
                'dir_x': dir_x,
                'dir_y': dir_y,
                'manual_successors': manual_successors,
                'all_successors': set()
            })
            
            self.lane_grids[internal_id] = parsed_lanes[-1]
            self.get_logger().info(f'Loaded {internal_id} (name: {name})')
            
        # 2) Process successors (manual + automatic)
        manual_conn = 0
        auto_conn = 0
        
        for lane in parsed_lanes:
            # Resolve manual
            for succ_name in lane['manual_successors']:
                if succ_name in name_to_id:
                    succ_id = name_to_id[succ_name]
                    lane['all_successors'].add(succ_id)
                    manual_conn += 1
            
            # Auto connections
            if len(lane['points']) >= 2:
                last_pt = lane['points'][-1]
                prev_pt = lane['points'][-2]
                vA_x = last_pt[0] - prev_pt[0]
                vA_y = last_pt[1] - prev_pt[1]
                
                for other in parsed_lanes:
                    if other['id'] == lane['id']: continue
                    if len(other['points']) < 2: continue
                    
                    first_pt = other['points'][0]
                    next_pt = other['points'][1]
                    vB_x = next_pt[0] - first_pt[0]
                    vB_y = next_pt[1] - first_pt[1]
                    
                    dist = math.sqrt((last_pt[0] - first_pt[0])**2 + (last_pt[1] - first_pt[1])**2)
                    if dist <= self.connection_threshold_m:
                        # Check direction compatibility
                        dot = vA_x * vB_x + vA_y * vB_y
                        if dot >= 0: # angle < 90 deg
                            lane['all_successors'].add(other['id'])
                            auto_conn += 1
                        else:
                            self.get_logger().info(f"Auto-connect {lane['id']} -> {other['id']} rejected: direction contradicts")
                    
            if not lane['all_successors']:
                self.get_logger().warn(f"Lane {lane['id']} ({lane['name']}) has NO successors!")
            else:
                succ_list = list(lane['all_successors'])
                self.get_logger().info(f"{lane['id']} successors: {succ_list}")
                
        self.get_logger().info(f'Connections: {manual_conn} manual, {auto_conn} automatic.')
        self.lanes_loaded = True

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
        A* search with Ackermann constraints and optional independent lanes tracking.
        Uses Numba JIT-compiled core when available, with safe fallback to pure Python.
        """
        t_total_start = time.time()

        occupancy = self.latest_layers['occupancy']
        dir_x = self.latest_layers.get('dir_x', np.zeros_like(occupancy))
        dir_y = self.latest_layers.get('dir_y', np.zeros_like(occupancy))

        # ── Distance-to-wall + gradient (for right-edge attraction) ─────
        import cv2
        free_mask_bin = (np.abs(occupancy) < 0.1).astype(np.uint8)
        dist_to_wall = cv2.distanceTransform(free_mask_bin, cv2.DIST_L2, 5)
        dist_to_wall = dist_to_wall.astype(np.float32)
        grad_row, grad_col = np.gradient(dist_to_wall)
        grad_col = cv2.GaussianBlur(grad_col, (5, 5), 1.0)
        grad_row = cv2.GaussianBlur(grad_row, (5, 5), 1.0)
        grad_mag = np.sqrt(grad_col**2 + grad_row**2) + 1e-6
        grad_col /= grad_mag
        grad_row /= grad_mag

        # ── Inflate obstacles for safety ────────────────────────────────
        obstacle_mask = (np.abs(occupancy) > 0.1).astype(np.uint8)

        self.get_logger().info(
            f'[A*] Grid {self.grid_cols}x{self.grid_rows}, '
            f'free={int(np.sum(obstacle_mask == 0))}, '
            f'blocked={int(np.sum(obstacle_mask > 0))}, '
            f'start_yaw={math.degrees(start_yaw):.1f}°, '
            f'numba={NUMBA_AVAILABLE}',
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

        # ── Prepare lane data as numeric arrays ─────────────────────────
        # Build ordered list of lane internal IDs for numeric indexing
        lane_id_list = []  # index → internal_id string
        lane_id_map = {}   # internal_id string → numeric index
        if self.use_independent_lanes and self.lanes_loaded:
            for lid in self.lane_grids:
                idx = len(lane_id_list)
                lane_id_list.append(lid)
                lane_id_map[lid] = idx

        n_lanes = len(lane_id_list)
        rows = self.grid_rows
        cols = self.grid_cols

        # 3D lane arrays [N, rows, cols]
        if n_lanes > 0:
            lane_masks_3d = np.zeros((n_lanes, rows, cols), dtype=np.float32)
            lane_dir_x_3d = np.zeros((n_lanes, rows, cols), dtype=np.float32)
            lane_dir_y_3d = np.zeros((n_lanes, rows, cols), dtype=np.float32)
            for lid, idx in lane_id_map.items():
                lane_masks_3d[idx] = self.lane_grids[lid]['mask']
                lane_dir_x_3d[idx] = self.lane_grids[lid]['dir_x']
                lane_dir_y_3d[idx] = self.lane_grids[lid]['dir_y']
        else:
            lane_masks_3d = np.zeros((0, rows, cols), dtype=np.float32)
            lane_dir_x_3d = np.zeros((0, rows, cols), dtype=np.float32)
            lane_dir_y_3d = np.zeros((0, rows, cols), dtype=np.float32)

        # Successor matrix [N, N]
        successor_matrix = np.zeros((max(n_lanes, 1), max(n_lanes, 1)), dtype=np.int8)
        if n_lanes > 0:
            successor_matrix = np.zeros((n_lanes, n_lanes), dtype=np.int8)
            for lid, idx in lane_id_map.items():
                for succ_id in self.lane_grids[lid].get('all_successors', set()):
                    if succ_id in lane_id_map:
                        successor_matrix[idx, lane_id_map[succ_id]] = 1

        # Global lane mask
        global_lane_mask = self.latest_layers.get('lane_mask', np.zeros_like(occupancy))
        if global_lane_mask is None:
            global_lane_mask = np.zeros_like(occupancy)
        global_lane_mask = global_lane_mask.astype(np.float32)

        # ── Start lanes detection → numeric IDs ─────────────────────────
        # -2 = GLOBAL, -1 = NONE, >=0 = lane index
        start_lane_ids_list = []
        if self.use_independent_lanes and self.lanes_loaded and n_lanes > 0:
            for lid, idx in lane_id_map.items():
                if self.lane_grids[lid]['mask'][sr, sc] > 0.5:
                    start_lane_ids_list.append(idx)
            if not start_lane_ids_list:
                self.get_logger().warn('Start not in any lane! Using NONE.')
                start_lane_ids_list = [-1]
            else:
                names = [lane_id_list[i] for i in start_lane_ids_list]
                self.get_logger().info(f'Start lanes detected: {names}')
        else:
            start_lane_ids_list = [-2]  # GLOBAL

        start_lane_ids = np.array(start_lane_ids_list, dtype=np.int32)

        fwd_dx = math.cos(start_yaw)
        fwd_dy = math.sin(start_yaw)

        # ── Call Numba core or Python fallback ──────────────────────────
        used_numba = False
        path_cells = None
        iterations = 0

        t_astar_start = time.time()

        if NUMBA_AVAILABLE and _astar_core_numba is not None:
            try:
                path_cols, path_rows, iterations = _astar_core_numba(
                    obstacle_mask,
                    grad_col.astype(np.float32), grad_row.astype(np.float32),
                    dir_x.astype(np.float32), dir_y.astype(np.float32),
                    global_lane_mask,
                    lane_masks_3d, lane_dir_x_3d, lane_dir_y_3d,
                    successor_matrix,
                    start_lane_ids,
                    np.int32(sc), np.int32(sr), np.int32(gc), np.int32(gr),
                    np.float64(fwd_dx), np.float64(fwd_dy),
                    np.float64(self.map_info.resolution),
                    np.float64(self.min_direction_dot),
                    np.float64(self.direction_penalty),
                    np.float64(self.max_turn_cos),
                    np.float64(self.right_bias_penalty),
                    np.int32(self.forward_lock_cells ** 2),
                    self.enforce_lane_mask,
                    self.enforce_lane_successors,
                    np.int32(rows), np.int32(cols),
                )
                used_numba = True
                if len(path_cols) > 0:
                    path_cells = [(int(path_cols[i]), int(path_rows[i])) for i in range(len(path_cols))]
                else:
                    path_cells = None
            except Exception as e:
                self.get_logger().warn(f'[A*] Numba core failed: {e}. Falling back to Python.')
                used_numba = False

        if not used_numba:
            # ── Python fallback (original logic) ────────────────────────
            try:
                path_cols, path_rows, iterations = _astar_core_python(
                    obstacle_mask,
                    grad_col.astype(np.float32), grad_row.astype(np.float32),
                    dir_x.astype(np.float32), dir_y.astype(np.float32),
                    global_lane_mask,
                    lane_masks_3d, lane_dir_x_3d, lane_dir_y_3d,
                    successor_matrix,
                    start_lane_ids,
                    np.int32(sc), np.int32(sr), np.int32(gc), np.int32(gr),
                    np.float64(fwd_dx), np.float64(fwd_dy),
                    np.float64(self.map_info.resolution),
                    np.float64(self.min_direction_dot),
                    np.float64(self.direction_penalty),
                    np.float64(self.max_turn_cos),
                    np.float64(self.right_bias_penalty),
                    np.int32(self.forward_lock_cells ** 2),
                    self.enforce_lane_mask,
                    self.enforce_lane_successors,
                    np.int32(rows), np.int32(cols),
                )
                if len(path_cols) > 0:
                    path_cells = [(int(path_cols[i]), int(path_rows[i])) for i in range(len(path_cols))]
                else:
                    path_cells = None
            except Exception as e:
                self.get_logger().error(f'[A*] Python core also failed: {e}')
                path_cells = None
                iterations = 0

        t_astar_end = time.time()
        t_total_end = time.time()

        astar_ms = (t_astar_end - t_astar_start) * 1000.0
        total_ms = (t_total_end - t_total_start) * 1000.0
        backend = 'NUMBA' if used_numba else 'PYTHON'
        path_len = len(path_cells) if path_cells else 0

        self.get_logger().info(
            f'[A* PERF] backend={backend}, '
            f'iterations={iterations}, '
            f'path_length={path_len}, '
            f'astar_runtime={astar_ms:.1f}ms, '
            f'total_planning={total_ms:.1f}ms'
        )

        if path_cells is None:
            self.get_logger().warn(f'A* exhausted search after {iterations} iterations. No path found.')

        return path_cells

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

        # Generate semi-goals (pass original goal msg for last waypoint)
        self.main_goal = (goal_x, goal_y)
        self._generate_semi_goals(path_msg, msg)

    # =====================================================================
    # Semi-goal generation from A* path
    # =====================================================================
    def _generate_semi_goals(self, path_msg: Path, original_goal: PoseStamped = None):
        """Split path into semi-goals every semi_goal_spacing_m meters.
        Orientations point toward the NEXT semi-goal.
        Last semi-goal is a copy of the original /bt/goal."""
        if not path_msg.poses:
            return

        semi_goals = []
        accumulated_dist = 0.0

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
                semi_goals.append((px, py, 0.0))  # yaw will be recalculated
                accumulated_dist = 0.0

        # ── Last semi-goal = copy of /bt/goal ────────────────────────────
        if original_goal is not None:
            gx = original_goal.pose.position.x
            gy = original_goal.pose.position.y
            gq = original_goal.pose.orientation
            gyaw = math.atan2(
                2.0 * (gq.w * gq.z + gq.x * gq.y),
                1.0 - 2.0 * (gq.y**2 + gq.z**2),
            )
            # Don't duplicate if very close to last semi-goal
            if not semi_goals or math.sqrt(
                (gx - semi_goals[-1][0])**2 + (gy - semi_goals[-1][1])**2
            ) > 0.1:
                semi_goals.append((gx, gy, gyaw))
            else:
                # Replace last with exact goal position + orientation
                semi_goals[-1] = (gx, gy, gyaw)
        else:
            # Fallback: use last path pose
            last_pose = path_msg.poses[-1]
            lx = last_pose.pose.position.x
            ly = last_pose.pose.position.y
            lq = last_pose.pose.orientation
            lyaw = math.atan2(
                2.0 * (lq.w * lq.z + lq.x * lq.y),
                1.0 - 2.0 * (lq.y**2 + lq.z**2),
            )
            if not semi_goals or math.sqrt(
                (lx - semi_goals[-1][0])**2 + (ly - semi_goals[-1][1])**2
            ) > 0.1:
                semi_goals.append((lx, ly, lyaw))

        # ── Recalculate orientations: each points toward the NEXT ────────
        for i in range(len(semi_goals) - 1):
            sx, sy, _ = semi_goals[i]
            nx, ny, _ = semi_goals[i + 1]
            yaw = math.atan2(ny - sy, nx - sx)
            semi_goals[i] = (sx, sy, yaw)
        # Last semi-goal keeps its original orientation (from /bt/goal)

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
