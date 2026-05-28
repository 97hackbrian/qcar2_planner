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
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

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
    edge_clearance_3d,    # float32 [N, rows, cols] — per-lane edge clearance cost
    curve_inner_3d,       # float32 [N, rows, cols] — per-lane curve inner cost
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
    use_edge_clearance,   # bool
    edge_clearance_weight,# float64
    use_curve_clearance,  # bool
    curve_clearance_weight,# float64
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

                # ── Edge clearance penalty (optional) ───────────────
                if use_edge_clearance and n_slot < n_lanes:
                    ec_val = float(edge_clearance_3d[n_slot, nr, nc])
                    penalty += edge_clearance_weight * ec_val

                # ── Curve inner clearance penalty (optional) ────────
                if use_curve_clearance and n_slot < n_lanes:
                    cc_val = float(curve_inner_3d[n_slot, nr, nc])
                    penalty += curve_clearance_weight * cc_val

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

        # Curve Safe Semigoal Spacing
        self.declare_parameter('use_curve_safe_semigoal_spacing', True)
        self.declare_parameter('straight_semigoal_spacing_m', 0.30)
        self.declare_parameter('curve_semigoal_spacing_m', 0.18)
        self.declare_parameter('max_curve_segment_length_m', 0.22)
        self.declare_parameter('curve_segment_check_step_m', 0.02)

        # Ackermann Curve Yaw Offset
        self.declare_parameter('use_ackermann_curve_yaw_offset', False)
        self.declare_parameter('curve_detection_window', 5)
        self.declare_parameter('curve_min_consecutive_angle_changes', 3)
        self.declare_parameter('curve_delta_angle_min_deg', 3.0)
        self.declare_parameter('curve_delta_angle_std_max_deg', 8.0)
        self.declare_parameter('curve_total_angle_min_deg', 12.0)
        self.declare_parameter('curve_yaw_offset_max_deg', 5.0)
        self.declare_parameter('curve_yaw_offset_min_deg', 0.5)
        self.declare_parameter('curve_yaw_ramp_in_ratio', 0.50)
        self.declare_parameter('curve_yaw_ramp_out_ratio', 0.50)
        self.declare_parameter('curve_yaw_rate_limit_deg', 2.0)

        self.declare_parameter('use_right_boundary_adaptive_opening', True)
        self.declare_parameter('right_boundary_check_distance_m', 0.35)
        self.declare_parameter('right_boundary_near_m', 0.12)
        self.declare_parameter('right_boundary_far_m', 0.35)
        self.declare_parameter('right_boundary_check_step_m', 0.02)
        self.declare_parameter('minimum_opening_if_no_right_boundary_deg', 0.0)

        self.declare_parameter('use_no_right_boundary_curve_tightening', True)
        self.declare_parameter('no_right_boundary_tightening_gain', 0.55)
        self.declare_parameter('no_right_boundary_near_m', 0.12)
        self.declare_parameter('no_right_boundary_far_m', 0.30)
        self.declare_parameter('no_right_boundary_check_distance_m', 0.45)
        self.declare_parameter('no_right_boundary_check_step_m', 0.02)
        self.declare_parameter('max_no_boundary_opening_reduction_ratio', 0.65)

        self.declare_parameter('use_post_curve_straight_counteroffset', False)
        self.declare_parameter('post_curve_counteroffset_m', 0.05)
        self.declare_parameter('post_curve_counteroffset_max_m', 0.10)
        self.declare_parameter('post_curve_counteroffset_distance_m', 0.70)
        self.declare_parameter('post_curve_counteroffset_ramp_in_m', 0.15)
        self.declare_parameter('post_curve_counteroffset_ramp_out_m', 0.25)
        self.declare_parameter('post_curve_straight_angle_threshold_deg', 6.0)
        self.declare_parameter('post_curve_min_total_angle_deg', 12.0)
        self.declare_parameter('post_curve_right_boundary_trigger_m', 0.22)
        self.declare_parameter('post_curve_counteroffset_check_step_m', 0.02)
        self.declare_parameter('post_curve_counteroffset_direction', 'left')

        self.declare_parameter('use_right_curve_protection', False)
        self.declare_parameter('right_curve_protection_offset_m', 0.05)
        self.declare_parameter('right_curve_protection_max_offset_m', 0.10)
        self.declare_parameter('right_curve_protection_distance_m', 0.80)
        self.declare_parameter('right_curve_protection_ramp_in_m', 0.20)
        self.declare_parameter('right_curve_protection_ramp_out_m', 0.25)
        self.declare_parameter('right_curve_angle_threshold_deg', 8.0)
        self.declare_parameter('right_curve_min_total_angle_deg', 12.0)
        self.declare_parameter('right_curve_boundary_trigger_m', 0.25)
        self.declare_parameter('right_curve_check_step_m', 0.02)
        self.declare_parameter('right_curve_apply_to_positions', True)
        self.declare_parameter('right_curve_apply_to_yaw', False)
        self.declare_parameter('right_curve_yaw_offset_deg', 3.0)

        # Edge Clearance Cost (optional soft penalty near lane edges)
        self.declare_parameter('use_edge_clearance_cost', False)
        self.declare_parameter('edge_clearance_weight', 0.08)
        self.declare_parameter('min_clearance_to_lane_edge_m', 0.12)

        # Curve Inner Clearance Cost (optional penalty near inner edges of curves)
        self.declare_parameter('use_curve_clearance_cost', False)
        self.declare_parameter('curve_clearance_weight', 0.10)
        self.declare_parameter('curve_angle_threshold_deg', 20.0)
        self.declare_parameter('curve_inner_clearance_m', 0.15)

        # Path Postprocessing (optional path cleanup before semi-goal generation)
        self.declare_parameter('use_path_postprocessing', False)
        self.declare_parameter('path_smoothing_window', 5)
        self.declare_parameter('path_prune_collinear_tolerance_m', 0.03)
        self.declare_parameter('max_smoothing_shift_m', 0.20)

        # Semi-goal zigzag cleanup (optional local correction of zigzag semi-goals)
        self.declare_parameter('use_semigoal_zigzag_cleanup', False)
        self.declare_parameter('prefer_zigzag_correction_over_deletion', True)
        self.declare_parameter('max_semigoal_deletions_per_path', 2)
        self.declare_parameter('min_semigoals_to_keep', 5)
        self.declare_parameter('max_direct_segment_after_deletion_m', 0.35)
        self.declare_parameter('zigzag_angle_threshold_deg', 30.0)
        self.declare_parameter('zigzag_lateral_threshold_m', 0.05)
        self.declare_parameter('max_zigzag_fix_shift_m', 0.12)
        self.declare_parameter('semigoal_segment_check_step_m', 0.02)

        # Semi-Goal Yaw Validation
        self.declare_parameter('use_semigoal_yaw_validation', True)
        self.declare_parameter('min_yaw_segment_distance_m', 0.08)
        self.declare_parameter('max_yaw_jump_deg', 45.0)
        self.declare_parameter('yaw_smoothing_window', 3)
        self.declare_parameter('preserve_final_goal_yaw', True)
        self.declare_parameter('repair_bad_yaw_using_neighbors', True)

        # Semi-Goal Sanitizer
        self.declare_parameter('use_semigoal_sanitizer', True)
        self.declare_parameter('min_semigoal_distance_m', 0.22)
        self.declare_parameter('min_semigoal_distance_curve_m', 0.18)
        self.declare_parameter('duplicate_semigoal_epsilon_m', 0.04)
        self.declare_parameter('max_semigoal_filter_iterations', 3)
        self.declare_parameter('max_semigoals_removed_per_path', 5)
        self.declare_parameter('preserve_first_semigoal', True)
        self.declare_parameter('preserve_final_goal', True)
        self.declare_parameter('semigoal_marker_scale', 0.12)

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

        # Curve Safe Semigoal Spacing
        self.use_curve_safe_semigoal_spacing = bool(self.get_parameter('use_curve_safe_semigoal_spacing').value)
        self.straight_semigoal_spacing_m = float(self.get_parameter('straight_semigoal_spacing_m').value)
        self.curve_semigoal_spacing_m = float(self.get_parameter('curve_semigoal_spacing_m').value)
        self.max_curve_segment_length_m = float(self.get_parameter('max_curve_segment_length_m').value)
        self.curve_segment_check_step_m = float(self.get_parameter('curve_segment_check_step_m').value)

        # Ackermann Curve Yaw Offset
        self.use_ackermann_curve_yaw_offset = bool(self.get_parameter('use_ackermann_curve_yaw_offset').value)
        self.curve_detection_window = int(self.get_parameter('curve_detection_window').value)
        self.curve_min_consecutive_angle_changes = int(self.get_parameter('curve_min_consecutive_angle_changes').value)
        self.curve_delta_angle_min_deg = float(self.get_parameter('curve_delta_angle_min_deg').value)
        self.curve_delta_angle_std_max_deg = float(self.get_parameter('curve_delta_angle_std_max_deg').value)
        self.curve_total_angle_min_deg = float(self.get_parameter('curve_total_angle_min_deg').value)
        
        self.curve_yaw_offset_max_deg = float(self.get_parameter('curve_yaw_offset_max_deg').value)
        self.curve_yaw_offset_min_deg = float(self.get_parameter('curve_yaw_offset_min_deg').value)
        self.curve_yaw_ramp_in_ratio = float(self.get_parameter('curve_yaw_ramp_in_ratio').value)
        self.curve_yaw_ramp_out_ratio = float(self.get_parameter('curve_yaw_ramp_out_ratio').value)
        self.curve_yaw_rate_limit_deg = float(self.get_parameter('curve_yaw_rate_limit_deg').value)
        
        self.use_right_boundary_adaptive_opening = bool(self.get_parameter('use_right_boundary_adaptive_opening').value)
        self.right_boundary_check_distance_m = float(self.get_parameter('right_boundary_check_distance_m').value)
        self.right_boundary_near_m = float(self.get_parameter('right_boundary_near_m').value)
        self.right_boundary_far_m = float(self.get_parameter('right_boundary_far_m').value)
        self.right_boundary_check_step_m = float(self.get_parameter('right_boundary_check_step_m').value)
        self.minimum_opening_if_no_right_boundary_deg = float(self.get_parameter('minimum_opening_if_no_right_boundary_deg').value)

        self.use_no_right_boundary_curve_tightening = bool(self.get_parameter('use_no_right_boundary_curve_tightening').value)
        self.no_right_boundary_tightening_gain = float(self.get_parameter('no_right_boundary_tightening_gain').value)
        self.no_right_boundary_near_m = float(self.get_parameter('no_right_boundary_near_m').value)
        self.no_right_boundary_far_m = float(self.get_parameter('no_right_boundary_far_m').value)
        self.no_right_boundary_check_distance_m = float(self.get_parameter('no_right_boundary_check_distance_m').value)
        self.no_right_boundary_check_step_m = float(self.get_parameter('no_right_boundary_check_step_m').value)
        self.max_no_boundary_opening_reduction_ratio = float(self.get_parameter('max_no_boundary_opening_reduction_ratio').value)

        self.use_post_curve_straight_counteroffset = bool(self.get_parameter('use_post_curve_straight_counteroffset').value)
        self.post_curve_counteroffset_m = float(self.get_parameter('post_curve_counteroffset_m').value)
        self.post_curve_counteroffset_max_m = float(self.get_parameter('post_curve_counteroffset_max_m').value)
        self.post_curve_counteroffset_distance_m = float(self.get_parameter('post_curve_counteroffset_distance_m').value)
        self.post_curve_counteroffset_ramp_in_m = float(self.get_parameter('post_curve_counteroffset_ramp_in_m').value)
        self.post_curve_counteroffset_ramp_out_m = float(self.get_parameter('post_curve_counteroffset_ramp_out_m').value)
        self.post_curve_straight_angle_threshold_deg = float(self.get_parameter('post_curve_straight_angle_threshold_deg').value)
        self.post_curve_min_total_angle_deg = float(self.get_parameter('post_curve_min_total_angle_deg').value)
        self.post_curve_right_boundary_trigger_m = float(self.get_parameter('post_curve_right_boundary_trigger_m').value)
        self.post_curve_counteroffset_check_step_m = float(self.get_parameter('post_curve_counteroffset_check_step_m').value)
        self.post_curve_counteroffset_direction = str(self.get_parameter('post_curve_counteroffset_direction').value)

        self.use_right_curve_protection = bool(self.get_parameter('use_right_curve_protection').value)
        self.right_curve_protection_offset_m = float(self.get_parameter('right_curve_protection_offset_m').value)
        self.right_curve_protection_max_offset_m = float(self.get_parameter('right_curve_protection_max_offset_m').value)
        self.right_curve_protection_distance_m = float(self.get_parameter('right_curve_protection_distance_m').value)
        self.right_curve_protection_ramp_in_m = float(self.get_parameter('right_curve_protection_ramp_in_m').value)
        self.right_curve_protection_ramp_out_m = float(self.get_parameter('right_curve_protection_ramp_out_m').value)
        self.right_curve_angle_threshold_deg = float(self.get_parameter('right_curve_angle_threshold_deg').value)
        self.right_curve_min_total_angle_deg = float(self.get_parameter('right_curve_min_total_angle_deg').value)
        self.right_curve_boundary_trigger_m = float(self.get_parameter('right_curve_boundary_trigger_m').value)
        self.right_curve_check_step_m = float(self.get_parameter('right_curve_check_step_m').value)
        self.right_curve_apply_to_positions = bool(self.get_parameter('right_curve_apply_to_positions').value)
        self.right_curve_apply_to_yaw = bool(self.get_parameter('right_curve_apply_to_yaw').value)
        self.right_curve_yaw_offset_deg = float(self.get_parameter('right_curve_yaw_offset_deg').value)

        rcp_state = 'ACTIVE' if self.use_right_curve_protection else 'INACTIVE'
        self.get_logger().info(
            f'[RIGHT_CURVE_PROT] {rcp_state}: '
            f'offset={self.right_curve_protection_offset_m}m, '
            f'max={self.right_curve_protection_max_offset_m}m, '
            f'trigger={self.right_curve_boundary_trigger_m}m, '
            f'apply_pos={self.right_curve_apply_to_positions}, '
            f'apply_yaw={self.right_curve_apply_to_yaw}'
        )

        # Edge Clearance Cost
        self.use_edge_clearance_cost = bool(self.get_parameter('use_edge_clearance_cost').value)
        self.edge_clearance_weight = float(self.get_parameter('edge_clearance_weight').value)
        self.min_clearance_to_lane_edge_m = float(self.get_parameter('min_clearance_to_lane_edge_m').value)

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

        # Edge clearance log
        ec_state = 'ACTIVE' if self.use_edge_clearance_cost else 'INACTIVE'
        self.get_logger().info(
            f'[EDGE_CLEARANCE] {ec_state}: '
            f'weight={self.edge_clearance_weight}, '
            f'min_clearance={self.min_clearance_to_lane_edge_m}m'
        )

        # Curve clearance parameters
        self.use_curve_clearance_cost = bool(self.get_parameter('use_curve_clearance_cost').value)
        self.curve_clearance_weight = float(self.get_parameter('curve_clearance_weight').value)
        self.curve_angle_threshold_deg = float(self.get_parameter('curve_angle_threshold_deg').value)
        self.curve_inner_clearance_m = float(self.get_parameter('curve_inner_clearance_m').value)

        cc_state = 'ACTIVE' if self.use_curve_clearance_cost else 'INACTIVE'
        self.get_logger().info(
            f'[CURVE_CLEARANCE] {cc_state}: '
            f'weight={self.curve_clearance_weight}, '
            f'angle_thresh={self.curve_angle_threshold_deg}°, '
            f'inner_clearance={self.curve_inner_clearance_m}m'
        )

        # Path Postprocessing
        self.use_path_postprocessing = bool(self.get_parameter('use_path_postprocessing').value)
        self.path_smoothing_window = int(self.get_parameter('path_smoothing_window').value)
        self.path_prune_collinear_tolerance_m = float(self.get_parameter('path_prune_collinear_tolerance_m').value)
        self.max_smoothing_shift_m = float(self.get_parameter('max_smoothing_shift_m').value)

        pp_state = 'ACTIVE' if self.use_path_postprocessing else 'INACTIVE'
        self.get_logger().info(
            f'[PATH_POSTPROCESS] {pp_state}: '
            f'smoothing_window={self.path_smoothing_window}, '
            f'collinear_tol={self.path_prune_collinear_tolerance_m}m, '
            f'max_shift={self.max_smoothing_shift_m}m'
        )

        # Semi-goal zigzag cleanup
        self.use_semigoal_zigzag_cleanup = bool(self.get_parameter('use_semigoal_zigzag_cleanup').value)
        self.prefer_zigzag_correction_over_deletion = bool(self.get_parameter('prefer_zigzag_correction_over_deletion').value)
        self.max_semigoal_deletions_per_path = int(self.get_parameter('max_semigoal_deletions_per_path').value)
        self.min_semigoals_to_keep = int(self.get_parameter('min_semigoals_to_keep').value)
        self.max_direct_segment_after_deletion_m = float(self.get_parameter('max_direct_segment_after_deletion_m').value)
        self.zigzag_angle_threshold_deg = float(self.get_parameter('zigzag_angle_threshold_deg').value)
        self.zigzag_lateral_threshold_m = float(self.get_parameter('zigzag_lateral_threshold_m').value)
        self.max_zigzag_fix_shift_m = float(self.get_parameter('max_zigzag_fix_shift_m').value)
        self.semigoal_segment_check_step_m = float(self.get_parameter('semigoal_segment_check_step_m').value)

        zz_state = 'ACTIVE' if self.use_semigoal_zigzag_cleanup else 'INACTIVE'
        self.get_logger().info(
            f'[ZIGZAG_CLEANUP] {zz_state}: '
            f'prefer_corr={self.prefer_zigzag_correction_over_deletion}, '
            f'max_del={self.max_semigoal_deletions_per_path}, '
            f'min_keep={self.min_semigoals_to_keep}, '
            f'angle_thresh={self.zigzag_angle_threshold_deg}°, '
            f'lateral_thresh={self.zigzag_lateral_threshold_m}m, '
            f'max_shift={self.max_zigzag_fix_shift_m}m, '
            f'check_step={self.semigoal_segment_check_step_m}m'
        )

        # Semi-Goal Yaw Validation
        self.use_semigoal_yaw_validation = bool(self.get_parameter('use_semigoal_yaw_validation').value)
        self.min_yaw_segment_distance_m = float(self.get_parameter('min_yaw_segment_distance_m').value)
        self.max_yaw_jump_deg = float(self.get_parameter('max_yaw_jump_deg').value)
        self.yaw_smoothing_window = int(self.get_parameter('yaw_smoothing_window').value)
        self.preserve_final_goal_yaw = bool(self.get_parameter('preserve_final_goal_yaw').value)
        self.repair_bad_yaw_using_neighbors = bool(self.get_parameter('repair_bad_yaw_using_neighbors').value)

        yv_state = 'ACTIVE' if self.use_semigoal_yaw_validation else 'INACTIVE'
        self.get_logger().info(
            f'[YAW_VALIDATION] {yv_state}: '
            f'min_seg_dist={self.min_yaw_segment_distance_m}m, '
            f'max_jump={self.max_yaw_jump_deg}°, '
            f'smooth_win={self.yaw_smoothing_window}, '
            f'preserve_final={self.preserve_final_goal_yaw}, '
            f'repair={self.repair_bad_yaw_using_neighbors}'
        )

        # Semi-Goal Sanitizer
        self.use_semigoal_sanitizer = bool(self.get_parameter('use_semigoal_sanitizer').value)
        self.min_semigoal_distance_m = float(self.get_parameter('min_semigoal_distance_m').value)
        self.min_semigoal_distance_curve_m = float(self.get_parameter('min_semigoal_distance_curve_m').value)
        self.duplicate_semigoal_epsilon_m = float(self.get_parameter('duplicate_semigoal_epsilon_m').value)
        self.max_semigoal_filter_iterations = int(self.get_parameter('max_semigoal_filter_iterations').value)
        self.max_semigoals_removed_per_path = int(self.get_parameter('max_semigoals_removed_per_path').value)
        self.preserve_first_semigoal = bool(self.get_parameter('preserve_first_semigoal').value)
        self.preserve_final_goal = bool(self.get_parameter('preserve_final_goal').value)
        self.semigoal_marker_scale = float(self.get_parameter('semigoal_marker_scale').value)

        san_state = 'ACTIVE' if self.use_semigoal_sanitizer else 'INACTIVE'
        self.get_logger().info(
            f'[SANITIZER] {san_state}: '
            f'min_dist={self.min_semigoal_distance_m}m, '
            f'min_dist_curve={self.min_semigoal_distance_curve_m}m, '
            f'dup_eps={self.duplicate_semigoal_epsilon_m}m, '
            f'max_iter={self.max_semigoal_filter_iterations}, '
            f'max_removed={self.max_semigoals_removed_per_path}, '
            f'marker_scale={self.semigoal_marker_scale}m'
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
        qos_latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.gridmap_sub = self.create_subscription(
            GridMapMsg, '/grid_map', self.gridmap_callback, qos_latched
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

        # 3) Compute edge clearance cost for each lane (optional)
        if self.use_edge_clearance_cost:
            self._compute_edge_clearance_cost(parsed_lanes, res)
        else:
            self.get_logger().info('[EDGE_CLEARANCE] Disabled — skipping edge clearance computation.')

        # 4) Compute curve inner clearance cost for each lane (optional)
        if self.use_curve_clearance_cost:
            self._compute_curve_inner_cost(parsed_lanes, res)
        else:
            self.get_logger().info('[CURVE_CLEARANCE] Disabled — skipping curve clearance computation.')

        self.lanes_loaded = True

    # =====================================================================
    # Compute edge clearance cost per independent lane
    # =====================================================================
    def _compute_edge_clearance_cost(self, parsed_lanes, resolution):
        """
        For each lane, compute a soft penalty grid based on distance to the
        lane mask edge.  Cells close to the edge get high cost (up to 1.0),
        cells far from the edge get 0 cost.

        Uses cv2.distanceTransform on the binary lane mask.
        Cost = max(0, 1 - dist_m / min_clearance_to_lane_edge_m)
        """
        import cv2

        min_clearance_m = self.min_clearance_to_lane_edge_m
        if min_clearance_m <= 0.0:
            self.get_logger().warn(
                '[EDGE_CLEARANCE] min_clearance_to_lane_edge_m <= 0 — '
                'disabling edge clearance cost.'
            )
            return

        self.get_logger().info(
            f'[EDGE_CLEARANCE] Computing edge clearance cost for '
            f'{len(parsed_lanes)} lanes (min_clearance={min_clearance_m}m, '
            f'resolution={resolution}m/px)...'
        )

        for lane in parsed_lanes:
            mask = lane['mask']
            # Binary mask: 1 where lane is valid, 0 elsewhere
            binary_mask = (mask > 0.5).astype(np.uint8)

            # Distance transform: distance from each foreground pixel to
            # nearest background pixel (= lane edge)
            dist_px = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
            dist_m = dist_px.astype(np.float32) * resolution

            # Cost: 1.0 at edge, linearly decays to 0.0 at min_clearance_m
            # cost = max(0, 1 - dist_m / min_clearance_m)
            edge_cost = np.clip(1.0 - dist_m / min_clearance_m, 0.0, 1.0)

            # Zero out cost outside the lane mask (safety — should already be
            # blocked by mask, but be defensive)
            edge_cost[binary_mask == 0] = 0.0

            lane['edge_clearance_cost'] = edge_cost.astype(np.float32)

            # Store in lane_grids too
            lid = lane['id']
            if lid in self.lane_grids:
                self.lane_grids[lid]['edge_clearance_cost'] = edge_cost.astype(np.float32)

            # Log stats
            valid_costs = edge_cost[binary_mask > 0]
            if valid_costs.size > 0:
                self.get_logger().info(
                    f'[EDGE_CLEARANCE] {lid} ({lane["name"]}): '
                    f'min_cost={float(valid_costs.min()):.4f}, '
                    f'max_cost={float(valid_costs.max()):.4f}, '
                    f'mean_cost={float(valid_costs.mean()):.4f}, '
                    f'penalized_cells={int(np.sum(valid_costs > 0))}'
                )
            else:
                self.get_logger().warn(
                    f'[EDGE_CLEARANCE] {lid} ({lane["name"]}): '
                    f'no valid cells in mask!'
                )

        self.get_logger().info('[EDGE_CLEARANCE] Edge clearance cost computation complete.')

    # =====================================================================
    # Compute curve inner clearance cost per independent lane
    # =====================================================================
    def _compute_curve_inner_cost(self, parsed_lanes, resolution):
        """
        For each lane, detect curves (angle > threshold) and compute a soft penalty
        grid based on distance to the inner edge of the curve.
        """
        import cv2

        angle_thresh_rad = math.radians(self.curve_angle_threshold_deg)
        inner_clearance_m = self.curve_inner_clearance_m

        if inner_clearance_m <= 0.0:
            self.get_logger().warn(
                '[CURVE_CLEARANCE] curve_inner_clearance_m <= 0 — disabling curve cost.'
            )
            return

        self.get_logger().info(
            f'[CURVE_CLEARANCE] Computing curve inner cost for '
            f'{len(parsed_lanes)} lanes (inner_clearance={inner_clearance_m}m, '
            f'angle_thresh={self.curve_angle_threshold_deg}°)...'
        )

        origin_x = self.map_info.pose.position.x - self.map_info.length_x / 2.0
        origin_y = self.map_info.pose.position.y - self.map_info.length_y / 2.0

        for lane in parsed_lanes:
            pts = lane['points']
            width = float(lane.get('width', 0.0))
            if len(pts) < 3 or width <= 0.0:
                curve_cost = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)
                lane['curve_inner_cost'] = curve_cost
                if lane['id'] in self.lane_grids:
                    self.lane_grids[lane['id']]['curve_inner_cost'] = curve_cost
                continue

            inner_edge_mask = np.zeros((self.grid_rows, self.grid_cols), dtype=np.uint8)
            curve_segments_count = 0

            for i in range(1, len(pts) - 1):
                p1 = pts[i - 1]
                p2 = pts[i]
                p3 = pts[i + 1]

                v1x = p2[0] - p1[0]
                v1y = p2[1] - p1[1]
                v2x = p3[0] - p2[0]
                v2y = p3[1] - p2[1]

                mag1 = math.sqrt(v1x * v1x + v1y * v1y)
                mag2 = math.sqrt(v2x * v2x + v2y * v2y)

                if mag1 < 1e-6 or mag2 < 1e-6:
                    continue

                v1x /= mag1; v1y /= mag1
                v2x /= mag2; v2y /= mag2

                dot = v1x * v2x + v1y * v2y
                dot = max(-1.0, min(1.0, dot))
                angle = math.acos(dot)

                if angle > angle_thresh_rad:
                    # Curve detected. Find inner side via cross product
                    cross = v1x * v2y - v1y * v2x
                    # If cross > 0 (left turn), inner is on the left
                    if cross > 0:
                        n1x, n1y = -v1y, v1x
                        n2x, n2y = -v2y, v2x
                    else:
                        n1x, n1y = v1y, -v1x
                        n2x, n2y = v2y, -v2x

                    # Inner edge points for segment 1 (p1 -> p2)
                    ie1_p1 = (p1[0] + n1x * (width / 2.0), p1[1] + n1y * (width / 2.0))
                    ie1_p2 = (p2[0] + n1x * (width / 2.0), p2[1] + n1y * (width / 2.0))

                    # Inner edge points for segment 2 (p2 -> p3)
                    ie2_p1 = (p2[0] + n2x * (width / 2.0), p2[1] + n2y * (width / 2.0))
                    ie2_p2 = (p3[0] + n2x * (width / 2.0), p3[1] + n2y * (width / 2.0))

                    def w2g(p):
                        c = int((p[0] - origin_x) / resolution - 0.5)
                        r = int((p[1] - origin_y) / resolution - 0.5)
                        return (c, r)

                    c1_1, r1_1 = w2g(ie1_p1)
                    c1_2, r1_2 = w2g(ie1_p2)
                    c2_1, r2_1 = w2g(ie2_p1)
                    c2_2, r2_2 = w2g(ie2_p2)

                    cv2.line(inner_edge_mask, (c1_1, r1_1), (c1_2, r1_2), 1, 1)
                    cv2.line(inner_edge_mask, (c2_1, r2_1), (c2_2, r2_2), 1, 1)
                    curve_segments_count += 1

            if curve_segments_count > 0:
                dist_px = cv2.distanceTransform(1 - inner_edge_mask, cv2.DIST_L2, 5)
                dist_m = dist_px.astype(np.float32) * resolution
                # Cost: 1.0 at edge, linearly decays to 0.0 at inner_clearance_m
                curve_cost = np.clip(1.0 - dist_m / inner_clearance_m, 0.0, 1.0)
                # Apply lane mask
                binary_lane_mask = (lane['mask'] > 0.5).astype(np.float32)
                curve_cost = curve_cost * binary_lane_mask
            else:
                curve_cost = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)

            lane['curve_inner_cost'] = curve_cost
            if lane['id'] in self.lane_grids:
                self.lane_grids[lane['id']]['curve_inner_cost'] = curve_cost

            # Log stats
            valid_costs = curve_cost[curve_cost > 0]
            if curve_segments_count > 0:
                self.get_logger().info(
                    f'[CURVE_CLEARANCE] {lane["id"]} ({lane["name"]}): '
                    f'{curve_segments_count} curve segments, '
                    f'penalized_cells={int(np.sum(curve_cost > 0))}'
                )

        self.get_logger().info('[CURVE_CLEARANCE] Curve inner clearance cost computation complete.')

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
            edge_clearance_3d = np.zeros((n_lanes, rows, cols), dtype=np.float32)
            curve_inner_3d = np.zeros((n_lanes, rows, cols), dtype=np.float32)
            for lid, idx in lane_id_map.items():
                lane_masks_3d[idx] = self.lane_grids[lid]['mask']
                lane_dir_x_3d[idx] = self.lane_grids[lid]['dir_x']
                lane_dir_y_3d[idx] = self.lane_grids[lid]['dir_y']
                # Fallback-safe: only copy if precomputed
                if 'edge_clearance_cost' in self.lane_grids[lid]:
                    edge_clearance_3d[idx] = self.lane_grids[lid]['edge_clearance_cost']
                if 'curve_inner_cost' in self.lane_grids[lid]:
                    curve_inner_3d[idx] = self.lane_grids[lid]['curve_inner_cost']
        else:
            lane_masks_3d = np.zeros((0, rows, cols), dtype=np.float32)
            lane_dir_x_3d = np.zeros((0, rows, cols), dtype=np.float32)
            lane_dir_y_3d = np.zeros((0, rows, cols), dtype=np.float32)
            edge_clearance_3d = np.zeros((0, rows, cols), dtype=np.float32)
            curve_inner_3d = np.zeros((0, rows, cols), dtype=np.float32)

        # Resolve effective edge clearance state for this A* call
        use_ec = self.use_edge_clearance_cost and n_lanes > 0
        ec_weight = self.edge_clearance_weight if use_ec else 0.0
        if use_ec:
            self.get_logger().info(
                f'[A*] Edge clearance cost ACTIVE (weight={ec_weight}, '
                f'min_clearance={self.min_clearance_to_lane_edge_m}m)'
            )
        else:
            self.get_logger().info('[A*] Edge clearance cost INACTIVE — no extra penalty applied.')

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
                    edge_clearance_3d, curve_inner_3d,
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
                    use_ec,
                    np.float64(ec_weight),
                    self.use_curve_clearance_cost and n_lanes > 0,
                    np.float64(self.curve_clearance_weight if self.use_curve_clearance_cost else 0.0),
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
                    edge_clearance_3d, curve_inner_3d,
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
                    use_ec,
                    np.float64(ec_weight),
                    self.use_curve_clearance_cost and n_lanes > 0,
                    np.float64(self.curve_clearance_weight if self.use_curve_clearance_cost else 0.0),
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

        # ── Optional path postprocessing ─────────────────────────────────
        path_msg_original = path_msg  # keep reference to raw A* path
        if self.use_path_postprocessing:
            path_msg = self._postprocess_path(path_msg)

            # ── Fallback: if postprocessing left < 2 poses, revert ───────
            if len(path_msg.poses) < 2:
                self.get_logger().warn(
                    '[PATH_POSTPROCESS] Postprocessed path has < 2 poses! '
                    'Falling back to original A* path.'
                )
                path_msg = path_msg_original

            # ── Resample path to ensure dense-enough points ──────────────
            resample_spacing = min(self.semi_goal_spacing / 2.0, 0.10)
            path_msg = self._resample_path_by_spacing(path_msg, resample_spacing)
        else:
            self.get_logger().info('[PATH_POSTPROCESS] Disabled — using raw A* path.')

        # Generate semi-goals (pass original goal msg for last waypoint)
        self.main_goal = (goal_x, goal_y)
        self._generate_semi_goals(path_msg, msg, path_msg_original=path_msg_original)

    # =====================================================================
    # Path postprocessing — clean up A* path before semi-goal generation
    # =====================================================================
    def _postprocess_path(self, path_msg: Path):
        """
        Clean up the raw A* path to remove zigzags, near-duplicates, and
        collinear noise.  Then apply moving-average smoothing clamped to
        max_smoothing_shift_m.  Smoothed points that fall outside the lane
        mask are reverted to original.

        Returns a NEW Path message (does not mutate the input).
        """
        if len(path_msg.poses) < 3:
            self.get_logger().info('[PATH_POSTPROCESS] Path too short to postprocess.')
            return path_msg

        # ── Extract world points ─────────────────────────────────────────
        raw_pts = [
            (p.pose.position.x, p.pose.position.y)
            for p in path_msg.poses
        ]
        n_original = len(raw_pts)

        # ── Step 1: Remove duplicate / near-duplicate points ─────────────
        dedup_pts = [raw_pts[0]]
        dedup_tol = 1e-4  # ~0.1mm — essentially exact duplicates
        for i in range(1, len(raw_pts)):
            dx = raw_pts[i][0] - dedup_pts[-1][0]
            dy = raw_pts[i][1] - dedup_pts[-1][1]
            if math.sqrt(dx*dx + dy*dy) > dedup_tol:
                dedup_pts.append(raw_pts[i])
        n_after_dedup = len(dedup_pts)

        # ── Step 2: Remove micro-zigzags ─────────────────────────────────
        # A zigzag is where the direction suddenly reverses (dot < 0)
        if len(dedup_pts) >= 3:
            clean_pts = [dedup_pts[0], dedup_pts[1]]
            for i in range(2, len(dedup_pts)):
                # Direction from prev-prev to prev
                ax = clean_pts[-1][0] - clean_pts[-2][0]
                ay = clean_pts[-1][1] - clean_pts[-2][1]
                # Direction from prev to current
                bx = dedup_pts[i][0] - clean_pts[-1][0]
                by = dedup_pts[i][1] - clean_pts[-1][1]
                a_norm = math.sqrt(ax*ax + ay*ay)
                b_norm = math.sqrt(bx*bx + by*by)
                if a_norm > 1e-6 and b_norm > 1e-6:
                    dot = (ax*bx + ay*by) / (a_norm * b_norm)
                    if dot < -0.5:
                        # Severe reversal — skip the previous point (it's the zigzag tip)
                        clean_pts[-1] = dedup_pts[i]
                        continue
                clean_pts.append(dedup_pts[i])
        else:
            clean_pts = list(dedup_pts)
        n_after_zigzag = len(clean_pts)

        # ── Step 3: Remove collinear points ──────────────────────────────
        tol = self.path_prune_collinear_tolerance_m
        if len(clean_pts) >= 3:
            pruned = [clean_pts[0]]
            for i in range(1, len(clean_pts) - 1):
                # Perpendicular distance from point i to line (pruned[-1] → clean_pts[i+1])
                ax = clean_pts[i+1][0] - pruned[-1][0]
                ay = clean_pts[i+1][1] - pruned[-1][1]
                bx = clean_pts[i][0] - pruned[-1][0]
                by = clean_pts[i][1] - pruned[-1][1]
                line_len = math.sqrt(ax*ax + ay*ay)
                if line_len < 1e-6:
                    continue
                # Cross product magnitude / line length = perpendicular distance
                perp_dist = abs(ax*by - ay*bx) / line_len
                if perp_dist > tol:
                    pruned.append(clean_pts[i])
            pruned.append(clean_pts[-1])  # always keep last
        else:
            pruned = list(clean_pts)
        n_after_prune = len(pruned)

        # ── Step 4: Moving-average smoothing ─────────────────────────────
        window = self.path_smoothing_window
        max_shift = self.max_smoothing_shift_m
        half_w = window // 2
        smoothed = list(pruned)  # copy
        max_actual_shift = 0.0
        n_reverted_shift = 0

        if len(pruned) > window and window >= 3:
            for i in range(1, len(pruned) - 1):  # never move first/last
                lo = max(0, i - half_w)
                hi = min(len(pruned), i + half_w + 1)
                avg_x = sum(p[0] for p in pruned[lo:hi]) / (hi - lo)
                avg_y = sum(p[1] for p in pruned[lo:hi]) / (hi - lo)

                shift = math.sqrt(
                    (avg_x - pruned[i][0])**2 +
                    (avg_y - pruned[i][1])**2
                )

                if shift > max_shift:
                    # Clamp: move only max_shift in the direction of avg
                    ratio = max_shift / shift
                    avg_x = pruned[i][0] + (avg_x - pruned[i][0]) * ratio
                    avg_y = pruned[i][1] + (avg_y - pruned[i][1]) * ratio
                    shift = max_shift

                max_actual_shift = max(max_actual_shift, shift)
                smoothed[i] = (avg_x, avg_y)

        # ── Step 5: Validate smoothed points against lane mask ───────────
        n_rejected_lane = 0
        if self.use_independent_lanes and self.lanes_loaded and self.lane_grids:
            for i in range(1, len(smoothed) - 1):
                sx, sy = smoothed[i]
                col, row = self._world_to_grid(sx, sy)
                if not self._in_bounds(col, row):
                    # Out of map — revert
                    smoothed[i] = pruned[i]
                    n_rejected_lane += 1
                    continue
                # Check if it's inside any lane mask
                in_lane = False
                for lid, ldata in self.lane_grids.items():
                    if ldata['mask'][row, col] > 0.5:
                        in_lane = True
                        break
                if not in_lane:
                    smoothed[i] = pruned[i]
                    n_rejected_lane += 1
        elif self.enforce_lane_mask and self.latest_layers is not None:
            # Global lane mask fallback
            global_mask = self.latest_layers.get('lane_mask', None)
            if global_mask is not None:
                for i in range(1, len(smoothed) - 1):
                    sx, sy = smoothed[i]
                    col, row = self._world_to_grid(sx, sy)
                    if not self._in_bounds(col, row) or global_mask[row, col] < 0.5:
                        smoothed[i] = pruned[i]
                        n_rejected_lane += 1

        # ── Build new Path message ───────────────────────────────────────
        new_path = Path()
        new_path.header = path_msg.header

        for (wx, wy) in smoothed:
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position.x = float(wx)
            ps.pose.position.y = float(wy)
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            new_path.poses.append(ps)

        # Recompute orientations
        for i in range(len(new_path.poses) - 1):
            p1 = new_path.poses[i].pose.position
            p2 = new_path.poses[i + 1].pose.position
            yaw = math.atan2(p2.y - p1.y, p2.x - p1.x)
            new_path.poses[i].pose.orientation.z = math.sin(yaw / 2.0)
            new_path.poses[i].pose.orientation.w = math.cos(yaw / 2.0)
        if len(new_path.poses) >= 2:
            new_path.poses[-1].pose.orientation = new_path.poses[-2].pose.orientation

        # ── Log summary ──────────────────────────────────────────────────
        self.get_logger().info(
            f'[PATH_POSTPROCESS] '
            f'original={n_original}, '
            f'after_dedup={n_after_dedup}, '
            f'after_zigzag_removal={n_after_zigzag}, '
            f'after_collinear_prune={n_after_prune}, '
            f'final_smoothed={len(smoothed)}, '
            f'max_shift={max_actual_shift:.4f}m, '
            f'rejected_out_of_lane={n_rejected_lane}'
        )

        return new_path

    # =====================================================================
    # Resample path — ensure points every spacing_m along the path
    # =====================================================================
    def _resample_path_by_spacing(self, path_msg: Path, spacing_m: float):
        """
        Take a path (possibly with few points after pruning) and return a
        new path with points interpolated every spacing_m metres along the
        original polyline.  First and last points are always preserved.
        """
        if len(path_msg.poses) < 2 or spacing_m <= 0.0:
            return path_msg

        # Extract (x, y) list
        pts = [
            (p.pose.position.x, p.pose.position.y)
            for p in path_msg.poses
        ]

        resampled = [pts[0]]  # always keep first
        residual = 0.0  # distance accumulated since last emitted point

        for i in range(1, len(pts)):
            seg_dx = pts[i][0] - pts[i - 1][0]
            seg_dy = pts[i][1] - pts[i - 1][1]
            seg_len = math.sqrt(seg_dx * seg_dx + seg_dy * seg_dy)
            if seg_len < 1e-9:
                continue

            # Unit direction along this segment
            ux = seg_dx / seg_len
            uy = seg_dy / seg_len

            consumed = 0.0  # distance consumed along this segment

            # First potential point: fill the residual from previous segment
            first_gap = spacing_m - residual
            if first_gap <= seg_len:
                consumed = first_gap
                resampled.append((
                    pts[i - 1][0] + ux * consumed,
                    pts[i - 1][1] + uy * consumed,
                ))
                # Continue placing points at full spacing intervals
                while consumed + spacing_m <= seg_len:
                    consumed += spacing_m
                    resampled.append((
                        pts[i - 1][0] + ux * consumed,
                        pts[i - 1][1] + uy * consumed,
                    ))
                residual = seg_len - consumed
            else:
                # Entire segment fits inside the residual gap
                residual += seg_len

        # Always keep last point (exact goal position)
        last = pts[-1]
        if resampled:
            dx = last[0] - resampled[-1][0]
            dy = last[1] - resampled[-1][1]
            if math.sqrt(dx * dx + dy * dy) > 1e-6:
                resampled.append(last)
            else:
                resampled[-1] = last  # snap to exact
        else:
            resampled.append(last)

        # Build new Path message
        new_path = Path()
        new_path.header = path_msg.header

        for (wx, wy) in resampled:
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position.x = float(wx)
            ps.pose.position.y = float(wy)
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            new_path.poses.append(ps)

        # Recompute orientations
        for i in range(len(new_path.poses) - 1):
            p1 = new_path.poses[i].pose.position
            p2 = new_path.poses[i + 1].pose.position
            yaw = math.atan2(p2.y - p1.y, p2.x - p1.x)
            new_path.poses[i].pose.orientation.z = math.sin(yaw / 2.0)
            new_path.poses[i].pose.orientation.w = math.cos(yaw / 2.0)
        if len(new_path.poses) >= 2:
            new_path.poses[-1].pose.orientation = new_path.poses[-2].pose.orientation

        self.get_logger().info(
            f'[RESAMPLE] input_poses={len(path_msg.poses)}, '
            f'output_poses={len(new_path.poses)}, '
            f'spacing={spacing_m:.4f}m'
        )

        return new_path

    # =====================================================================
    # Semi-goal generation from A* path — segment interpolation
    # =====================================================================
    def _generate_semi_goals(self, path_msg: Path,
                             original_goal: PoseStamped = None,
                             path_msg_original: Path = None):
        """Generate semi-goals every semi_goal_spacing_m metres by
        interpolating along each segment of the path polyline.

        This is robust to sparse / postprocessed paths because it does NOT
        depend on existing path points — it walks along the polyline and
        places semi-goals at exact spacing intervals.

        Orientations point toward the NEXT semi-goal.
        Last semi-goal is a copy of the original /bt/goal."""

        # ── Logging: path stats received ─────────────────────────────────
        n_poses = len(path_msg.poses)
        self.get_logger().info(
            f'[SEMI_GOALS] Received path with {n_poses} poses, '
            f'semi_goal_spacing_m={self.semi_goal_spacing:.3f}m'
        )

        # ── Guard: need at least 2 poses ─────────────────────────────────
        if n_poses < 2:
            if path_msg_original is not None and len(path_msg_original.poses) >= 2:
                self.get_logger().warn(
                    f'[SEMI_GOALS] Path has {n_poses} poses (< 2). '
                    'Falling back to original A* path '
                    f'({len(path_msg_original.poses)} poses).'
                )
                path_msg = path_msg_original
                n_poses = len(path_msg.poses)
            else:
                self.get_logger().warn(
                    f'[SEMI_GOALS] Path has {n_poses} poses (< 2) '
                    'and no original fallback available — cannot generate semi-goals.'
                )
                return

        if n_poses < 2:
            return

        # ── Compute total path length ────────────────────────────────────
        total_length = 0.0
        for i in range(1, n_poses):
            dx = path_msg.poses[i].pose.position.x - path_msg.poses[i - 1].pose.position.x
            dy = path_msg.poses[i].pose.position.y - path_msg.poses[i - 1].pose.position.y
            total_length += math.sqrt(dx * dx + dy * dy)

        self.get_logger().info(
            f'[SEMI_GOALS] Total path length={total_length:.3f}m'
        )

        # ── Precompute Curve Safe Spacing per segment ──────────────────────
        segment_spacings = [self.semi_goal_spacing] * (n_poses - 1)
        if self.use_curve_safe_semigoal_spacing:
            is_curve = [False] * (n_poses - 1)
            angle_thresh = math.radians(15.0)
            for i in range(1, n_poses - 1):
                p1 = path_msg.poses[i - 1].pose.position
                p2 = path_msg.poses[i].pose.position
                p3 = path_msg.poses[i + 1].pose.position

                v1x, v1y = p2.x - p1.x, p2.y - p1.y
                v2x, v2y = p3.x - p2.x, p3.y - p2.y

                mag1 = math.sqrt(v1x * v1x + v1y * v1y)
                mag2 = math.sqrt(v2x * v2x + v2y * v2y)

                if mag1 > 1e-6 and mag2 > 1e-6:
                    dot = (v1x * v2x + v1y * v2y) / (mag1 * mag2)
                    dot = max(-1.0, min(1.0, dot))
                    angle = math.acos(dot)
                    if angle > angle_thresh:
                        is_curve[i - 1] = True
                        is_curve[i] = True
                        if i > 1: is_curve[i - 2] = True # buffer before
                        if i < n_poses - 2: is_curve[i + 1] = True # buffer after

            straight_sp = self.straight_semigoal_spacing_m
            curve_sp = self.curve_semigoal_spacing_m
            segment_spacings = [curve_sp if c else straight_sp for c in is_curve]
            
            curves_count = sum(is_curve)
            self.get_logger().info(
                f'[SEMI_GOALS] Curve Safe spacing: {curves_count} curve segments '
                f'(spacing={curve_sp}m), {len(is_curve) - curves_count} straight '
                f'(spacing={straight_sp}m)'
            )

        # ── Walk along the polyline, interpolating semi-goals ────────────
        semi_goals = []
        accumulated_dist = 0.0  # distance since last emitted semi-goal
        used_interpolation = False

        for i in range(1, n_poses):
            ax = path_msg.poses[i - 1].pose.position.x
            ay = path_msg.poses[i - 1].pose.position.y
            bx = path_msg.poses[i].pose.position.x
            by = path_msg.poses[i].pose.position.y

            seg_dx = bx - ax
            seg_dy = by - ay
            seg_len = math.sqrt(seg_dx * seg_dx + seg_dy * seg_dy)
            if seg_len < 1e-9:
                continue
                
            spacing = segment_spacings[i - 1] if self.use_curve_safe_semigoal_spacing else self.semi_goal_spacing

            ux = seg_dx / seg_len
            uy = seg_dy / seg_len

            # Position along segment (offset from point A)
            consumed = 0.0

            # How far until next semi-goal?
            remaining_to_next = spacing - accumulated_dist
            if self.use_curve_safe_semigoal_spacing and remaining_to_next <= 0:
                remaining_to_next = 1e-3

            while consumed + remaining_to_next <= seg_len:
                consumed += remaining_to_next
                px = ax + ux * consumed
                py = ay + uy * consumed
                semi_goals.append((px, py, 0.0))
                accumulated_dist = 0.0
                remaining_to_next = spacing
                # If the interpolated point doesn't coincide with a path
                # vertex, flag that interpolation was used
                dx_to_b = bx - px
                dy_to_b = by - py
                if math.sqrt(dx_to_b * dx_to_b + dy_to_b * dy_to_b) > 1e-4:
                    used_interpolation = True

            # Leftover distance on this segment (not enough for a semi-goal)
            accumulated_dist += (seg_len - consumed)

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

        # ── Fallback: 0 semi-goals → use original path ──────────────────
        if len(semi_goals) == 0:
            self.get_logger().warn(
                '[SEMI_GOALS] Generated 0 semi-goals! '
                'Falling back to original A* path.'
            )
            if path_msg_original is not None and len(path_msg_original.poses) >= 2:
                return self._generate_semi_goals(
                    path_msg_original, original_goal, path_msg_original=None
                )
            else:
                self.get_logger().error(
                    '[SEMI_GOALS] No fallback path available — navigation aborted.'
                )
                return

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
            f'[SEMI_GOALS] Generated {len(semi_goals)} semi-goals '
            f'(spacing={self.semi_goal_spacing}m, '
            f'path_poses={n_poses}, '
            f'interpolation_used={used_interpolation})'
        )

        # ── Optional zigzag cleanup ──────────────────────────────────────
        if self.use_semigoal_zigzag_cleanup:
            self._cleanup_semigoal_zigzags()

        # ── Safe Curve Spacing ───────────────────────────────────────────
        self._ensure_curve_safe_spacing(path_msg)

        # ── Ackermann Curve Yaw Offset ───────────────────────────────────
        if getattr(self, 'use_ackermann_curve_yaw_offset', False):
            self._apply_ackermann_curve_yaw_offset()

        # ── Post-Curve Straight Counteroffset ───────────────────────────
        if getattr(self, 'use_post_curve_straight_counteroffset', False):
            self._apply_post_curve_straight_counteroffset()

        # ── Right Curve Protection ──────────────────────────────────────
        if getattr(self, 'use_right_curve_protection', False):
            self._apply_right_curve_protection()

        # ── Yaw Validation & Repair (final stage) ───────────────────────
        if getattr(self, 'use_semigoal_yaw_validation', False):
            self._validate_and_repair_semigoal_yaws()

        # ── Sanitizer (distance filter + robust yaw recalc) ─────────────
        if getattr(self, 'use_semigoal_sanitizer', False):
            self._sanitize_semigoals_before_publish()

        # Publish debug markers for ALL semi-goals
        self._publish_sg_markers()

        # Publish the FIRST semi-goal immediately
        self._publish_current_semi_goal()

    # =====================================================================
    # Normalize Angle Utility
    # =====================================================================
    def _normalize_angle(self, angle):
        """Normalize an angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    # =====================================================================
    # Ackermann Curve Yaw Offset & Safe Spacing
    # =====================================================================
    def _ensure_curve_safe_spacing(self, path_msg):
        if not self.use_curve_safe_semigoal_spacing or len(self.semi_goals) < 3:
            return

        N = len(self.semi_goals)
        headings = np.zeros(N - 1, dtype=np.float64)
        for j in range(N - 1):
            sx, sy, _ = self.semi_goals[j]
            nx, ny, _ = self.semi_goals[j + 1]
            headings[j] = math.atan2(ny - sy, nx - sx)

        deltas = np.zeros(N - 2, dtype=np.float64)
        for j in range(1, N - 1):
            deltas[j - 1] = self._normalize_angle(headings[j] - headings[j - 1])

        window_size = self.curve_detection_window
        min_consec = self.curve_min_consecutive_angle_changes
        delta_min = math.radians(self.curve_delta_angle_min_deg)
        std_max = math.radians(self.curve_delta_angle_std_max_deg)
        total_min = math.radians(self.curve_total_angle_min_deg)

        is_curve_sg = [False] * N
        for i in range(1, N - 1):
            center_idx = i - 1
            half_w = window_size // 2
            start_idx = max(0, center_idx - half_w)
            end_idx = min(len(deltas), center_idx + half_w + 1)
            wd = deltas[start_idx:end_idx]
            if len(wd) == 0: continue

            longest = []
            curr = []
            for d in wd:
                if abs(d) >= delta_min:
                    if not curr or np.sign(d) == np.sign(curr[0]):
                        curr.append(d)
                    else:
                        if len(curr) > len(longest): longest = curr
                        curr = [d]
                else:
                    if len(curr) > len(longest): longest = curr
                    curr = []
            if len(curr) > len(longest): longest = curr

            if len(longest) >= min_consec:
                if float(np.std(longest)) <= std_max and float(np.sum(np.abs(longest))) >= total_min:
                    is_curve_sg[i] = True

        new_semi_goals = [self.semi_goals[0]]
        
        def get_closest_path_idx(x, y, start_search=0):
            best_idx = start_search
            min_dist = float('inf')
            for idx in range(start_search, len(path_msg.poses)):
                px = path_msg.poses[idx].pose.position.x
                py = path_msg.poses[idx].pose.position.y
                dist = (px - x)**2 + (py - y)**2
                if dist < min_dist:
                    min_dist = dist
                    best_idx = idx
                elif dist > min_dist + 1.0: # Early exit since path is ordered
                    break
            return best_idx

        last_path_idx = 0
        inserted_points = 0
        unsafe_segments = 0

        for i in range(len(self.semi_goals) - 1):
            P1 = self.semi_goals[i]
            P2 = self.semi_goals[i+1]
            
            in_curve = is_curve_sg[i] or is_curve_sg[i+1]
            dist = math.sqrt((P2[0]-P1[0])**2 + (P2[1]-P1[1])**2)
            
            needs_interpolation = False
            
            if in_curve and dist > self.max_curve_segment_length_m:
                needs_interpolation = True
            
            if not needs_interpolation:
                if not self._is_segment_lane_safe(P1[0], P1[1], P2[0], P2[1], step=self.curve_segment_check_step_m):
                    needs_interpolation = True
                    unsafe_segments += 1
            
            if needs_interpolation:
                idx1 = get_closest_path_idx(P1[0], P1[1], last_path_idx)
                idx2 = get_closest_path_idx(P2[0], P2[1], idx1)
                last_path_idx = idx2
                
                subpath = path_msg.poses[idx1:idx2+1]
                if len(subpath) >= 2:
                    spacing = self.curve_semigoal_spacing_m
                    consumed = 0.0
                    accum = 0.0
                    for k in range(1, len(subpath)):
                        ax = subpath[k-1].pose.position.x
                        ay = subpath[k-1].pose.position.y
                        bx = subpath[k].pose.position.x
                        by = subpath[k].pose.position.y
                        seg_dx = bx - ax
                        seg_dy = by - ay
                        seg_len = math.sqrt(seg_dx*seg_dx + seg_dy*seg_dy)
                        
                        if seg_len < 1e-9: continue
                        ux, uy = seg_dx/seg_len, seg_dy/seg_len
                        
                        rem = spacing - accum
                        cons = 0.0
                        while cons + rem <= seg_len:
                            cons += rem
                            px = ax + ux * cons
                            py = ay + uy * cons
                            new_semi_goals.append((px, py, 0.0))
                            inserted_points += 1
                            accum = 0.0
                            rem = spacing
                        accum += (seg_len - cons)
                        
            new_semi_goals.append(P2)
            
        self.semi_goals = new_semi_goals
        if inserted_points > 0:
            self.get_logger().info(f'[CURVE_SAFE_SPACING] Inserted {inserted_points} points. Repaired {unsafe_segments} unsafe segments.')

    def _get_right_clearance(self, x, y, yaw, max_dist=None, step=None):
        dx = math.sin(yaw)
        dy = -math.cos(yaw)
        dist = 0.0
        if max_dist is None:
            max_dist = self.right_boundary_check_distance_m
        if step is None:
            step = self.right_boundary_check_step_m
        while dist <= max_dist:
            px = x + dx * dist
            py = y + dy * dist
            col, row = self._world_to_grid(px, py)
            if not self._in_bounds(col, row):
                return dist
            
            if self.latest_layers is not None and 'occupancy' in self.latest_layers:
                if self.latest_layers['occupancy'][row, col] > getattr(self, 'occupancy_threshold', 50):
                    return dist
                
            if self.use_independent_lanes and self.lanes_loaded and self.lane_grids:
                in_lane = False
                for ldata in self.lane_grids.values():
                    if ldata['mask'][row, col] > 0.5:
                        in_lane = True
                        break
                if not in_lane:
                    return dist
            elif getattr(self, 'enforce_lane_mask', False) and self.latest_layers is not None:
                global_mask = self.latest_layers.get('lane_mask', None)
                if global_mask is not None and global_mask[row, col] < 0.5:
                    return dist
                    
            dist += step
        return max_dist

    def _apply_ackermann_curve_yaw_offset(self):
        N = len(self.semi_goals)
        if N < 4: return

        headings = np.zeros(N - 1, dtype=np.float64)
        for j in range(N - 1):
            sx, sy, _ = self.semi_goals[j]
            nx, ny, _ = self.semi_goals[j + 1]
            headings[j] = math.atan2(ny - sy, nx - sx)

        deltas = np.zeros(N - 2, dtype=np.float64)
        for j in range(1, N - 1):
            deltas[j - 1] = self._normalize_angle(headings[j] - headings[j - 1])

        window_size = self.curve_detection_window
        min_consec = self.curve_min_consecutive_angle_changes
        delta_min = math.radians(self.curve_delta_angle_min_deg)
        std_max = math.radians(self.curve_delta_angle_std_max_deg)
        total_min = math.radians(self.curve_total_angle_min_deg)

        curve_types = np.zeros(N, dtype=np.int8)
        blocks = []

        i = 1
        while i < N - 1:
            center_idx = i - 1
            half_w = window_size // 2
            start_idx = max(0, center_idx - half_w)
            end_idx = min(len(deltas), center_idx + half_w + 1)
            wd = deltas[start_idx:end_idx]
            if len(wd) == 0:
                i += 1
                continue

            longest = []
            curr = []
            for d in wd:
                if abs(d) >= delta_min:
                    if not curr or np.sign(d) == np.sign(curr[0]):
                        curr.append(d)
                    else:
                        if len(curr) > len(longest): longest = curr
                        curr = [d]
                else:
                    if len(curr) > len(longest): longest = curr
                    curr = []
            if len(curr) > len(longest): longest = curr

            if len(longest) >= min_consec:
                if float(np.std(longest)) <= std_max and float(np.sum(np.abs(longest))) >= total_min:
                    avg_sign = np.sign(longest[0])
                    curve_types[i] = 1 if avg_sign > 0 else -1
            i += 1

        in_block = False
        start_b = 0
        for i in range(N):
            if curve_types[i] != 0 and not in_block:
                in_block = True
                start_b = i
            elif curve_types[i] == 0 and in_block:
                in_block = False
                if i - start_b >= 2:
                    blocks.append((start_b, i - 1, curve_types[start_b]))
            elif curve_types[i] != 0 and in_block and curve_types[i] != curve_types[i-1]:
                if i - start_b >= 2:
                    blocks.append((start_b, i - 1, curve_types[start_b]))
                start_b = i
        if in_block and N - start_b >= 2:
            blocks.append((start_b, N - 1, curve_types[start_b]))

        yaw_offsets = np.zeros(N, dtype=np.float64)
        ramp_in = self.curve_yaw_ramp_in_ratio
        ramp_out = self.curve_yaw_ramp_out_ratio
        max_deg = self.curve_yaw_offset_max_deg
        min_deg = self.curve_yaw_offset_min_deg

        def smoothstep(edge0, edge1, x):
            x = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
            return x * x * (3 - 2 * x)

        modified_count = 0
        
        for start_idx, end_idx, ctype in blocks:
            length = end_idx - start_idx
            for k in range(start_idx, end_idx + 1):
                phase = (k - start_idx) / float(max(length, 1))
                gain = 1.0
                if phase < ramp_in and ramp_in > 0:
                    gain = smoothstep(0.0, 1.0, phase / ramp_in)
                elif phase > 1.0 - ramp_out and ramp_out > 0:
                    gain = smoothstep(0.0, 1.0, (1.0 - phase) / ramp_out)

                offset_mag = min_deg + gain * (max_deg - min_deg)
                
                if self.use_right_boundary_adaptive_opening:
                    x, y, yaw_base = self.semi_goals[k]
                    clearance = self._get_right_clearance(x, y, yaw_base)
                    near_m = self.right_boundary_near_m
                    far_m = self.right_boundary_far_m
                    
                    if clearance <= near_m:
                        b_gain = 1.0
                    elif clearance >= far_m:
                        b_gain = 0.0
                    else:
                        b_gain = 1.0 - (clearance - near_m) / max(1e-6, far_m - near_m)
                        
                    if b_gain == 0.0:
                        offset_mag = self.minimum_opening_if_no_right_boundary_deg
                    else:
                        offset_mag = offset_mag * b_gain
                        
                if getattr(self, 'use_no_right_boundary_curve_tightening', False) and offset_mag > 1e-3:
                    x, y, yaw_base = self.semi_goals[k]
                    clearance2 = self._get_right_clearance(
                        x, y, yaw_base,
                        max_dist=self.no_right_boundary_check_distance_m,
                        step=self.no_right_boundary_check_step_m
                    )
                    n_near = self.no_right_boundary_near_m
                    n_far = self.no_right_boundary_far_m
                    
                    if clearance2 <= n_near:
                        tightening_gain = 0.0
                    elif clearance2 >= n_far:
                        tightening_gain = 1.0
                    else:
                        tightening_gain = (clearance2 - n_near) / max(1e-6, n_far - n_near)
                        
                    reduction = self.no_right_boundary_tightening_gain * tightening_gain
                    reduction = min(reduction, self.max_no_boundary_opening_reduction_ratio)
                    
                    orig_mag = offset_mag
                    offset_mag = offset_mag * (1.0 - reduction)
                    
                    border_found = (clearance2 < self.no_right_boundary_check_distance_m)
                    self.get_logger().debug(
                        f'[CURVE_TIGHTENING] right_boundary_distance={clearance2:.3f}, '
                        f'border_found={border_found}, '
                        f'tightening_gain={tightening_gain:.2f}, '
                        f'yaw_offset_original={orig_mag:.2f}, '
                        f'yaw_offset_final={offset_mag:.2f}, '
                        f'reduction={reduction:.2f}'
                    )
                
                yaw_offsets[k] = -offset_mag if ctype == 1 else offset_mag

        rate_limit = self.curve_yaw_rate_limit_deg
        for i in range(1, N):
            diff = yaw_offsets[i] - yaw_offsets[i-1]
            if abs(diff) > rate_limit:
                yaw_offsets[i] = yaw_offsets[i-1] + np.sign(diff) * rate_limit

        for i in range(1, N - 1):
            if abs(yaw_offsets[i]) > 1e-3:
                x, y, yaw_base = self.semi_goals[i]
                yaw_open = self._normalize_angle(yaw_base + math.radians(yaw_offsets[i]))
                self.semi_goals[i] = (x, y, yaw_open)
                modified_count += 1

        self.get_logger().info(
            f'[ACKERMANN_YAW] Analyzed {N} semi-goals. Modified {modified_count} across {len(blocks)} curves. '
            f'Params -> max_deg={max_deg}, min_deg={min_deg}, rate_limit={rate_limit}. '
            f'use_no_right_boundary_curve_tightening={getattr(self, "use_no_right_boundary_curve_tightening", False)}'
        )

    # =====================================================================
    # Post-Curve Straight Counteroffset
    # =====================================================================
    def _apply_post_curve_straight_counteroffset(self):
        """
        Applies a lateral counteroffset to semi-goals located in straight
        segments immediately following a curve, to avoid hugging the right boundary.
        """
        N = len(self.semi_goals)
        if N < 5:
            return

        headings = np.zeros(N - 1, dtype=np.float64)
        for j in range(N - 1):
            sx, sy, _ = self.semi_goals[j]
            nx, ny, _ = self.semi_goals[j + 1]
            headings[j] = math.atan2(ny - sy, nx - sx)

        deltas = np.zeros(N - 2, dtype=np.float64)
        for j in range(N - 2):
            diff = headings[j + 1] - headings[j]
            deltas[j] = self._normalize_angle(diff)

        # Detect curves and straight zones
        curve_types = np.zeros(N, dtype=np.int32)
        window = getattr(self, 'curve_detection_window', 5)
        min_consec = getattr(self, 'curve_min_consecutive_angle_changes', 3)
        total_min = math.radians(self.post_curve_min_total_angle_deg)
        straight_thresh = math.radians(self.post_curve_straight_angle_threshold_deg)

        i = 0
        while i < len(deltas) - window:
            window_slice = deltas[i:i+window]
            abs_slice = np.abs(window_slice)
            
            consecutive = 0
            longest = []
            current = []
            for d in window_slice:
                if abs(d) > 0.01:
                    current.append(d)
                    consecutive += 1
                else:
                    if consecutive > len(longest):
                        longest = current
                    consecutive = 0
                    current = []
            if consecutive > len(longest):
                longest = current
                
            if len(longest) >= min_consec and float(np.sum(np.abs(longest))) >= total_min:
                curve_types[i:i+window] = 1
            i += 1

        # Fill curve gaps and identify ends of curves
        in_curve = False
        curve_ends = []
        for j in range(N):
            if curve_types[j] == 1:
                in_curve = True
            elif in_curve:
                # Check if it's really a straight now
                is_straight = True
                check_len = min(3, N - 2 - j)
                if check_len > 0:
                    for k in range(j, j + check_len):
                        if abs(deltas[k]) >= straight_thresh:
                            is_straight = False
                            break
                if is_straight:
                    in_curve = False
                    curve_ends.append(j)
                else:
                    curve_types[j] = 1 # extend curve

        if not curve_ends:
            return

        modified_count = 0
        rejected_lane_safe = 0
        rejected_no_boundary = 0
        max_dist = self.post_curve_counteroffset_distance_m
        offset_target = self.post_curve_counteroffset_m
        offset_max = self.post_curve_counteroffset_max_m
        ramp_in = self.post_curve_counteroffset_ramp_in_m
        ramp_out = self.post_curve_counteroffset_ramp_out_m
        boundary_trigger = self.post_curve_right_boundary_trigger_m
        
        def _dist(a, b):
            return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

        for end_idx in curve_ends:
            straight_dist = 0.0
            for j in range(end_idx, N - 1):
                if j == N - 1:
                    break # Skip exact final goal
                if curve_types[j] == 1:
                    break # Hit another curve
                    
                pt = self.semi_goals[j]
                if j > end_idx:
                    straight_dist += _dist(self.semi_goals[j-1], pt)
                    
                if straight_dist > max_dist:
                    break
                    
                # Calculate ramp
                ramp_factor = 1.0
                if straight_dist < ramp_in and ramp_in > 0:
                    ramp_factor = straight_dist / ramp_in
                elif straight_dist > (max_dist - ramp_out) and ramp_out > 0:
                    ramp_factor = (max_dist - straight_dist) / ramp_out
                    
                ramp_factor = max(0.0, min(1.0, ramp_factor))
                if ramp_factor <= 0.01:
                    continue

                mag = min(offset_target * ramp_factor, offset_max)
                
                # Check right boundary
                yaw_base = headings[j] if j < len(headings) else pt[2]
                clearance = self._get_right_clearance(
                    pt[0], pt[1], yaw_base,
                    max_dist=boundary_trigger + 0.1,
                    step=self.post_curve_counteroffset_check_step_m
                )
                
                if clearance > boundary_trigger:
                    rejected_no_boundary += 1
                    continue
                    
                # Apply offset
                if self.post_curve_counteroffset_direction == "left":
                    ox = -math.sin(yaw_base) * mag
                    oy = math.cos(yaw_base) * mag
                else:
                    ox = -math.sin(yaw_base) * mag
                    oy = math.cos(yaw_base) * mag
                    
                new_x = pt[0] + ox
                new_y = pt[1] + oy
                
                # Verify lane-safe
                safe = True
                if j > 0:
                    prev = self.semi_goals[j-1]
                    safe = safe and self._is_segment_lane_safe(prev[0], prev[1], new_x, new_y, step=0.02)
                if j < N - 1:
                    nxt = self.semi_goals[j+1]
                    safe = safe and self._is_segment_lane_safe(new_x, new_y, nxt[0], nxt[1], step=0.02)
                    
                if safe:
                    self.semi_goals[j] = (new_x, new_y, pt[2])
                    modified_count += 1
                else:
                    rejected_lane_safe += 1

        self.get_logger().info(
            f'[POST_CURVE_OFFSET] curves_detected={len(curve_ends)}, '
            f'modified_points={modified_count}, '
            f'rejected_safe={rejected_lane_safe}, '
            f'rejected_no_boundary={rejected_no_boundary}'
        )

    # =====================================================================
    # Right Curve Protection
    # =====================================================================
    def _apply_right_curve_protection(self):
        """
        Applies a lateral leftward offset ONLY to semi-goals inside
        right-hand curves, to keep the QCar from hugging the inner
        boundary/sidewalk.  Left curves are completely untouched.
        """
        N = len(self.semi_goals)
        if N < 5:
            return

        # ── Compute headings and deltas ──────────────────────────────────
        headings = [0.0] * (N - 1)
        for j in range(N - 1):
            sx, sy, _ = self.semi_goals[j]
            nx, ny, _ = self.semi_goals[j + 1]
            headings[j] = math.atan2(ny - sy, nx - sx)

        deltas = [0.0] * (N - 2)
        for j in range(N - 2):
            deltas[j] = self._normalize_angle(headings[j + 1] - headings[j])

        # ── Detect right-curve blocks (negative delta = turning right) ───
        angle_thresh_rad = math.radians(self.right_curve_angle_threshold_deg)
        total_min_rad = math.radians(self.right_curve_min_total_angle_deg)
        window = getattr(self, 'curve_detection_window', 5)
        min_consec = getattr(self, 'curve_min_consecutive_angle_changes', 3)

        right_curve_mask = [False] * N
        right_blocks = []  # list of (start, end) index pairs

        i = 0
        while i <= len(deltas) - window:
            window_slice = deltas[i:i + window]

            # Count consistent negative deltas (right turn)
            consec_neg = 0
            longest_neg = []
            current_neg = []
            for d in window_slice:
                if d < -0.01:  # negative = turning right
                    current_neg.append(d)
                    consec_neg += 1
                else:
                    if len(current_neg) > len(longest_neg):
                        longest_neg = current_neg
                    consec_neg = 0
                    current_neg = []
            if len(current_neg) > len(longest_neg):
                longest_neg = current_neg

            if (len(longest_neg) >= min_consec and
                    sum(abs(d) for d in longest_neg) >= total_min_rad):
                for k in range(i, min(i + window, N)):
                    right_curve_mask[k] = True
            i += 1

        # ── Build contiguous blocks ──────────────────────────────────────
        in_block = False
        start_b = 0
        for j in range(N):
            if right_curve_mask[j] and not in_block:
                in_block = True
                start_b = j
            elif not right_curve_mask[j] and in_block:
                in_block = False
                if j - start_b >= 2:
                    right_blocks.append((start_b, j - 1))
        if in_block and N - start_b >= 2:
            right_blocks.append((start_b, N - 1))

        left_curves_ignored = 0
        # Count left curves for log (positive deltas)
        left_mask = [False] * N
        ii = 0
        while ii <= len(deltas) - window:
            ws = deltas[ii:ii + window]
            cn = []
            cur = []
            for d in ws:
                if d > 0.01:
                    cur.append(d)
                else:
                    if len(cur) > len(cn):
                        cn = cur
                    cur = []
            if len(cur) > len(cn):
                cn = cur
            if len(cn) >= min_consec and sum(abs(d) for d in cn) >= total_min_rad:
                left_curves_ignored += 1
            ii += 1

        if not right_blocks:
            self.get_logger().info(
                f'[RIGHT_CURVE_PROT] No right curves detected. '
                f'Left curves ignored={left_curves_ignored}.'
            )
            return

        # ── Apply protection ─────────────────────────────────────────────
        offset_target = self.right_curve_protection_offset_m
        offset_max = self.right_curve_protection_max_offset_m
        max_range = self.right_curve_protection_distance_m
        ramp_in = self.right_curve_protection_ramp_in_m
        ramp_out = self.right_curve_protection_ramp_out_m
        boundary_trigger = self.right_curve_boundary_trigger_m
        check_step = self.right_curve_check_step_m
        apply_pos = self.right_curve_apply_to_positions
        apply_yaw = self.right_curve_apply_to_yaw
        yaw_offset_rad = math.radians(self.right_curve_yaw_offset_deg)

        modified_pos = 0
        modified_yaw = 0
        rejected_safe = 0
        rejected_no_boundary = 0

        def _dist(a, b):
            return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

        for blk_start, blk_end in right_blocks:
            # Compute arc-length within the block
            arc_lengths = [0.0]
            for k in range(blk_start + 1, blk_end + 1):
                arc_lengths.append(
                    arc_lengths[-1] + _dist(self.semi_goals[k - 1], self.semi_goals[k])
                )
            total_arc = arc_lengths[-1]
            if total_arc < 1e-6:
                continue

            for k_rel, k in enumerate(range(blk_start, blk_end + 1)):
                # Skip the very last goal
                if k == N - 1:
                    continue

                arc = arc_lengths[k_rel]

                # Ramp envelope
                ramp_factor = 1.0
                if arc < ramp_in and ramp_in > 0:
                    ramp_factor = arc / ramp_in
                elif arc > (total_arc - ramp_out) and ramp_out > 0:
                    ramp_factor = (total_arc - arc) / ramp_out
                ramp_factor = max(0.0, min(1.0, ramp_factor))
                if ramp_factor <= 0.01:
                    continue

                mag = min(offset_target * ramp_factor, offset_max)

                pt = self.semi_goals[k]
                yaw_base = headings[k] if k < len(headings) else pt[2]

                # Check right boundary
                clearance = self._get_right_clearance(
                    pt[0], pt[1], yaw_base,
                    max_dist=boundary_trigger + 0.1,
                    step=check_step
                )

                if clearance > boundary_trigger:
                    rejected_no_boundary += 1
                    continue

                # ── Position offset ──────────────────────────────────────
                if apply_pos and mag > 1e-4:
                    left_x = -math.sin(yaw_base)
                    left_y = math.cos(yaw_base)
                    new_x = pt[0] + left_x * mag
                    new_y = pt[1] + left_y * mag

                    # Lane-safe validation
                    safe = True
                    if k > 0:
                        prev = self.semi_goals[k - 1]
                        safe = safe and self._is_segment_lane_safe(
                            prev[0], prev[1], new_x, new_y, step=0.02
                        )
                    if k < N - 1:
                        nxt = self.semi_goals[k + 1]
                        safe = safe and self._is_segment_lane_safe(
                            new_x, new_y, nxt[0], nxt[1], step=0.02
                        )

                    if safe:
                        self.semi_goals[k] = (new_x, new_y, pt[2])
                        modified_pos += 1
                    else:
                        rejected_safe += 1

                # ── Optional yaw offset ──────────────────────────────────
                if apply_yaw and ramp_factor > 0.01:
                    x, y, yaw_cur = self.semi_goals[k]
                    # For a right curve, open slightly left (positive yaw offset)
                    yaw_adj = yaw_offset_rad * ramp_factor
                    yaw_new = self._normalize_angle(yaw_cur + yaw_adj)
                    self.semi_goals[k] = (x, y, yaw_new)
                    modified_yaw += 1

        # ── Logging ──────────────────────────────────────────────────────
        block_strs = [f'[{s}-{e}]' for s, e in right_blocks]
        self.get_logger().info(
            f'[RIGHT_CURVE_PROT] right_curves={len(right_blocks)} {",".join(block_strs)}, '
            f'left_ignored={left_curves_ignored}, '
            f'pos_modified={modified_pos}, yaw_modified={modified_yaw}, '
            f'rejected_safe={rejected_safe}, rejected_no_boundary={rejected_no_boundary}, '
            f'max_offset={offset_max}m, N={N}'
        )

    # =====================================================================
    # Semi-Goal Yaw Validation & Repair
    # =====================================================================
    def _validate_and_repair_semigoal_yaws(self):
        """
        Final-stage robust yaw validation. Recalculates orientations using
        local tangents from valid-distance neighbours, detects jumps beyond
        max_yaw_jump_deg, repairs them, and optionally smooths small changes.
        Never moves (x, y) positions. Never deletes semi-goals.
        """
        N = len(self.semi_goals)
        if N < 2:
            return

        min_dist = self.min_yaw_segment_distance_m
        max_jump_rad = math.radians(self.max_yaw_jump_deg)
        window = self.yaw_smoothing_window
        preserve_final = self.preserve_final_goal_yaw
        do_repair = self.repair_bad_yaw_using_neighbors

        # ── Step 1: Compute robust yaw_base for every point ──────────────
        # For each point i, find forward/backward neighbours that are
        # at least min_dist away, then compute tangent-based yaw.

        yaw_base = [0.0] * N
        skipped_pairs = 0

        def _dist(a, b):
            return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

        for i in range(N):
            xi, yi, _ = self.semi_goals[i]

            # Search forward for a valid neighbour
            fwd_idx = None
            for j in range(i + 1, N):
                if _dist(self.semi_goals[i], self.semi_goals[j]) >= min_dist:
                    fwd_idx = j
                    break

            # Search backward for a valid neighbour
            bwd_idx = None
            for j in range(i - 1, -1, -1):
                if _dist(self.semi_goals[i], self.semi_goals[j]) >= min_dist:
                    bwd_idx = j
                    break

            if fwd_idx is not None and bwd_idx is not None:
                # Central tangent: P_prev → P_next
                px, py, _ = self.semi_goals[bwd_idx]
                nx, ny, _ = self.semi_goals[fwd_idx]
                yaw_base[i] = math.atan2(ny - py, nx - px)
            elif fwd_idx is not None:
                # Forward-only tangent: P_i → P_next
                nx, ny, _ = self.semi_goals[fwd_idx]
                yaw_base[i] = math.atan2(ny - yi, nx - xi)
            elif bwd_idx is not None:
                # Backward-only tangent: P_prev → P_i
                px, py, _ = self.semi_goals[bwd_idx]
                yaw_base[i] = math.atan2(yi - py, xi - px)
            else:
                # No valid neighbour at all — keep current yaw
                yaw_base[i] = self.semi_goals[i][2]
                skipped_pairs += 1
                self.get_logger().warn(
                    f'[YAW_VALIDATION] Point {i} has no valid neighbour '
                    f'(all within {min_dist}m) — keeping original yaw.'
                )

        # ── Step 2: Detect and repair invalid yaws ───────────────────────
        invalid_count = 0
        repaired_count = 0
        last_valid_idx = N - 1 if preserve_final else N

        for i in range(N):
            # Skip the final goal if we're preserving it
            if preserve_final and i == N - 1:
                continue

            x, y, yaw_current = self.semi_goals[i]
            diff = abs(self._normalize_angle(yaw_current - yaw_base[i]))

            if diff > max_jump_rad:
                invalid_count += 1
                if do_repair:
                    self.get_logger().debug(
                        f'[YAW_VALIDATION] Repaired point {i}: '
                        f'yaw_orig={math.degrees(yaw_current):.1f}° → '
                        f'yaw_base={math.degrees(yaw_base[i]):.1f}° '
                        f'(jump={math.degrees(diff):.1f}°)'
                    )
                    self.semi_goals[i] = (x, y, self._normalize_angle(yaw_base[i]))
                    repaired_count += 1
                else:
                    self.get_logger().warn(
                        f'[YAW_VALIDATION] Invalid yaw at point {i}: '
                        f'yaw={math.degrees(yaw_current):.1f}°, '
                        f'yaw_base={math.degrees(yaw_base[i]):.1f}°, '
                        f'jump={math.degrees(diff):.1f}° — NOT repaired (repair disabled).'
                    )

        # ── Step 3: Optional angular smoothing ───────────────────────────
        if window >= 3 and N >= window:
            smoothed_yaws = [self.semi_goals[i][2] for i in range(N)]
            half_w = window // 2

            smooth_end = (N - 1) if preserve_final else N

            for i in range(1, smooth_end):
                start_j = max(0, i - half_w)
                end_j = min(N, i + half_w + 1)

                # Collect neighbour yaws relative to current to handle wrapping
                ref = smoothed_yaws[i]
                total = 0.0
                count = 0
                for j in range(start_j, end_j):
                    d = self._normalize_angle(smoothed_yaws[j] - ref)
                    total += d
                    count += 1

                if count > 0:
                    avg_offset = total / count
                    candidate = self._normalize_angle(ref + avg_offset)

                    # Only apply smoothing if it does NOT create a big jump
                    # relative to the raw yaw_base direction
                    base_diff = abs(self._normalize_angle(candidate - yaw_base[i]))
                    if base_diff <= max_jump_rad:
                        smoothed_yaws[i] = candidate

            # Write smoothed yaws back
            for i in range(1, smooth_end):
                x, y, _ = self.semi_goals[i]
                self.semi_goals[i] = (x, y, smoothed_yaws[i])

        # ── Step 4: Final goal yaw check ─────────────────────────────────
        final_yaw_warning = 0
        if N >= 2:
            x_last, y_last, yaw_last = self.semi_goals[-1]
            # Compare final yaw against the direction from penultimate to last
            x_pen, y_pen, _ = self.semi_goals[-2]
            d_final = _dist(self.semi_goals[-2], self.semi_goals[-1])
            if d_final >= min_dist:
                path_dir = math.atan2(y_last - y_pen, x_last - x_pen)
                final_diff = abs(self._normalize_angle(yaw_last - path_dir))
                if final_diff > max_jump_rad:
                    final_yaw_warning = 1
                    if preserve_final:
                        self.get_logger().warn(
                            f'[YAW_VALIDATION] Final goal yaw differs strongly '
                            f'from path direction: goal_yaw={math.degrees(yaw_last):.1f}°, '
                            f'path_dir={math.degrees(path_dir):.1f}°, '
                            f'diff={math.degrees(final_diff):.1f}° — preserved as requested.'
                        )
                    else:
                        self.semi_goals[-1] = (x_last, y_last, self._normalize_angle(path_dir))
                        repaired_count += 1

        # ── Logging ──────────────────────────────────────────────────────
        self.get_logger().info(
            f'[YAW_VALIDATION] N={N}, skipped_pairs={skipped_pairs}, '
            f'invalid_detected={invalid_count}, repaired={repaired_count}, '
            f'final_yaw_warnings={final_yaw_warning}'
        )


    # =====================================================================
    # Semi-Goal Sanitizer  (distance filter + duplicate removal)
    # =====================================================================
    def _sanitize_semigoals_before_publish(self):
        """
        Final obligatory cleanup before publishing semi-goals to Nav2.
        1) Remove exact duplicates (within epsilon).
        2) Remove too-close points (unless removal breaks lane safety).
        3) Multi-iteration pass to catch cascading close pairs.
        4) Recalculate all yaws robustly.
        """
        N_before = len(self.semi_goals)
        if N_before < 2:
            return

        eps = self.duplicate_semigoal_epsilon_m
        min_dist = self.min_semigoal_distance_m
        min_dist_curve = self.min_semigoal_distance_curve_m
        max_iters = self.max_semigoal_filter_iterations
        max_removals = self.max_semigoals_removed_per_path
        preserve_first = self.preserve_first_semigoal
        preserve_final = self.preserve_final_goal

        total_duplicates = 0
        total_too_close = 0
        total_kept_for_safety = 0
        total_removed = 0

        def _dist(a, b):
            return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

        for iteration in range(max_iters):
            if len(self.semi_goals) < 3:
                break

            filtered = []
            removed_this_iter = 0
            N = len(self.semi_goals)

            # Always keep the first point
            filtered.append(self.semi_goals[0])

            for i in range(1, N):
                # Always keep the last point
                if preserve_final and i == N - 1:
                    filtered.append(self.semi_goals[i])
                    continue

                # Don't remove if we've hit the removal limit
                if total_removed >= max_removals:
                    filtered.append(self.semi_goals[i])
                    continue

                prev = filtered[-1]
                curr = self.semi_goals[i]
                d = _dist(prev, curr)

                # Case 1: exact duplicate
                if d < eps:
                    total_duplicates += 1
                    total_removed += 1
                    removed_this_iter += 1
                    continue

                # Case 2: too close — use curve distance if available
                threshold = min_dist_curve if d < min_dist else min_dist
                if d < threshold:
                    # Check if skipping this point keeps lane safety
                    # Look ahead to find the next point that will be kept
                    next_kept = None
                    for j in range(i + 1, N):
                        next_kept = self.semi_goals[j]
                        break

                    if next_kept is not None:
                        safe = self._is_segment_lane_safe(
                            prev[0], prev[1], next_kept[0], next_kept[1],
                            step=0.02
                        )
                        if safe:
                            total_too_close += 1
                            total_removed += 1
                            removed_this_iter += 1
                            continue
                        else:
                            total_kept_for_safety += 1

                filtered.append(curr)

            self.semi_goals = filtered

            if removed_this_iter == 0:
                break

        # ── Final duplicate sweep ────────────────────────────────────────
        if len(self.semi_goals) >= 3:
            final_filtered = [self.semi_goals[0]]
            for i in range(1, len(self.semi_goals)):
                if _dist(final_filtered[-1], self.semi_goals[i]) >= eps:
                    final_filtered.append(self.semi_goals[i])
                else:
                    if preserve_final and i == len(self.semi_goals) - 1:
                        final_filtered.append(self.semi_goals[i])
                    else:
                        total_duplicates += 1
                        total_removed += 1
            self.semi_goals = final_filtered

        # ── Robust yaw recalculation ─────────────────────────────────────
        self._recompute_semigoal_yaws_robust()

        # ── Compute stats ────────────────────────────────────────────────
        N_after = len(self.semi_goals)
        min_final_dist = float('inf')
        for i in range(1, N_after):
            d = _dist(self.semi_goals[i-1], self.semi_goals[i])
            if d < min_final_dist:
                min_final_dist = d

        self.get_logger().info(
            f'[SANITIZER] Before={N_before}, After={N_after}, '
            f'duplicates={total_duplicates}, too_close={total_too_close}, '
            f'kept_for_safety={total_kept_for_safety}, '
            f'total_removed={total_removed}, '
            f'min_final_dist={min_final_dist:.4f}m'
        )

        if min_final_dist < eps and N_after > 2:
            self.get_logger().warn(
                f'[SANITIZER] Warning: min distance {min_final_dist:.4f}m '
                f'is below epsilon {eps}m — some close points could not be '
                f'removed for lane-safety reasons.'
            )

    # =====================================================================
    # Robust Yaw Recalculation
    # =====================================================================
    def _recompute_semigoal_yaws_robust(self):
        """
        Recalculate all semi-goal yaws using valid-distance tangent neighbours.
        - Ignores pairs closer than min_semigoal_distance_curve_m for atan2.
        - Uses central tangent (P_prev→P_next) when both neighbours valid.
        - Falls back to unilateral tangent.
        - Preserves final goal yaw if configured.
        - Never moves (x,y) positions.
        """
        N = len(self.semi_goals)
        if N < 2:
            return

        min_dist = getattr(self, 'min_semigoal_distance_curve_m', 0.08)
        preserve_final = getattr(self, 'preserve_final_goal', True)
        yaw_repaired = 0

        def _dist(a, b):
            return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

        for i in range(N):
            # Skip the last point if preserving final goal yaw
            if preserve_final and i == N - 1:
                continue

            xi, yi, yaw_old = self.semi_goals[i]

            # Search forward for a valid neighbour
            fwd_idx = None
            for j in range(i + 1, N):
                if _dist(self.semi_goals[i], self.semi_goals[j]) >= min_dist:
                    fwd_idx = j
                    break

            # Search backward for a valid neighbour
            bwd_idx = None
            for j in range(i - 1, -1, -1):
                if _dist(self.semi_goals[i], self.semi_goals[j]) >= min_dist:
                    bwd_idx = j
                    break

            yaw_new = yaw_old  # default: keep

            if fwd_idx is not None and bwd_idx is not None:
                px, py, _ = self.semi_goals[bwd_idx]
                nx, ny, _ = self.semi_goals[fwd_idx]
                yaw_new = math.atan2(ny - py, nx - px)
            elif fwd_idx is not None:
                nx, ny, _ = self.semi_goals[fwd_idx]
                yaw_new = math.atan2(ny - yi, nx - xi)
            elif bwd_idx is not None:
                px, py, _ = self.semi_goals[bwd_idx]
                yaw_new = math.atan2(yi - py, xi - px)

            yaw_new = self._normalize_angle(yaw_new)

            if abs(self._normalize_angle(yaw_new - yaw_old)) > 1e-3:
                yaw_repaired += 1

            self.semi_goals[i] = (xi, yi, yaw_new)

        if yaw_repaired > 0:
            self.get_logger().info(
                f'[SANITIZER_YAW] Recalculated {yaw_repaired}/{N} yaws using robust tangent.'
            )

    # =====================================================================
    # Lane-safety check for a straight segment
    # =====================================================================
    def _is_segment_lane_safe(self, x1, y1, x2, y2, step=None):
        """Return True if all sampled points along (x1,y1)→(x2,y2) lie
        inside a valid lane mask (or inside the map if no lanes loaded).
        Sampling step is semigoal_segment_check_step_m unless overridden."""
        if step is None:
            step = getattr(self, 'semigoal_segment_check_step_m', 0.02)
        dx = x2 - x1
        dy = y2 - y1
        seg_len = math.sqrt(dx * dx + dy * dy)
        if seg_len < 1e-9:
            return True

        n_samples = max(2, int(math.ceil(seg_len / step)) + 1)

        for s in range(n_samples):
            t = s / max(n_samples - 1, 1)
            px = x1 + dx * t
            py = y1 + dy * t
            col, row = self._world_to_grid(px, py)

            if not self._in_bounds(col, row):
                return False

            # Check lane masks
            if self.use_independent_lanes and self.lanes_loaded and self.lane_grids:
                in_lane = False
                for ldata in self.lane_grids.values():
                    if ldata['mask'][row, col] > 0.5:
                        in_lane = True
                        break
                if not in_lane:
                    return False
            elif self.enforce_lane_mask and self.latest_layers is not None:
                global_mask = self.latest_layers.get('lane_mask', None)
                if global_mask is not None and global_mask[row, col] < 0.5:
                    return False

        return True

    # =====================================================================
    # Semi-goal zigzag cleanup — local correction without global smoothing
    # =====================================================================
    def _cleanup_semigoal_zigzags(self):
        """Analyze consecutive triples of semi-goals and remove or project
        middle points that form zigzags, but ONLY if the resulting segments
        are lane-safe.  First and last semi-goals are never modified."""
        sg = self.semi_goals
        if len(sg) < 3:
            self.get_logger().info('[ZIGZAG_CLEANUP] < 3 semi-goals — nothing to clean.')
            return

        angle_thresh_rad = math.radians(self.zigzag_angle_threshold_deg)
        lateral_thresh = self.zigzag_lateral_threshold_m
        max_shift = self.max_zigzag_fix_shift_m

        n_before = len(sg)
        n_zigzags_detected = 0
        n_removed = 0
        n_corrected = 0
        n_rejected = 0
        n_blocked_by_limit = 0
        n_blocked_by_length = 0
        n_blocked_by_min_keep = 0

        # Save last semi-goal orientation (must be preserved)
        last_yaw = sg[-1][2]

        # We iterate backwards so index removals don't shift remaining items
        i = len(sg) - 2  # start at second-to-last (skip last)
        while i >= 1:     # skip first (index 0)
            px, py, _ = sg[i - 1]
            cx, cy, _ = sg[i]
            nx, ny, _ = sg[i + 1]

            # ── Direction vectors ────────────────────────────────────────
            v1x = cx - px
            v1y = cy - py
            v2x = nx - cx
            v2y = ny - cy
            mag1 = math.sqrt(v1x * v1x + v1y * v1y)
            mag2 = math.sqrt(v2x * v2x + v2y * v2y)

            if mag1 < 1e-9 or mag2 < 1e-9:
                i -= 1
                continue

            # Angle between consecutive direction vectors
            dot = (v1x * v2x + v1y * v2y) / (mag1 * mag2)
            dot = max(-1.0, min(1.0, dot))  # clamp for acos safety
            angle = math.acos(dot)

            # ── Lateral distance of P_curr from segment P_prev→P_next ───
            seg_dx = nx - px
            seg_dy = ny - py
            seg_len = math.sqrt(seg_dx * seg_dx + seg_dy * seg_dy)
            if seg_len < 1e-9:
                i -= 1
                continue
            # Signed perpendicular distance (cross product / length)
            lateral_dist = abs((seg_dx * (cy - py) - seg_dy * (cx - px)) / seg_len)

            # ── Is this a zigzag? ────────────────────────────────────────
            if angle < angle_thresh_rad or lateral_dist < lateral_thresh:
                i -= 1
                continue

            n_zigzags_detected += 1

            def attempt_correction():
                # Parametric projection of (cx,cy) onto line (px,py)→(nx,ny)
                t_proj = ((cx - px) * seg_dx + (cy - py) * seg_dy) / (seg_len * seg_len)
                t_proj = max(0.0, min(1.0, t_proj))
                proj_x = px + seg_dx * t_proj
                proj_y = py + seg_dy * t_proj

                # Limit displacement
                shift_dx = proj_x - cx
                shift_dy = proj_y - cy
                shift_dist = math.sqrt(shift_dx * shift_dx + shift_dy * shift_dy)
                if shift_dist > max_shift:
                    ratio = max_shift / shift_dist
                    proj_x = cx + shift_dx * ratio
                    proj_y = cy + shift_dy * ratio

                # Validate both sub-segments
                if (self._is_segment_lane_safe(px, py, proj_x, proj_y) and
                        self._is_segment_lane_safe(proj_x, proj_y, nx, ny)):
                    return (proj_x, proj_y, 0.0)
                return None

            def attempt_deletion():
                nonlocal n_blocked_by_limit, n_blocked_by_min_keep, n_blocked_by_length
                if n_removed >= self.max_semigoal_deletions_per_path:
                    n_blocked_by_limit += 1
                    return False
                if len(sg) <= self.min_semigoals_to_keep:
                    n_blocked_by_min_keep += 1
                    return False
                if seg_len > self.max_direct_segment_after_deletion_m:
                    n_blocked_by_length += 1
                    return False
                if self._is_segment_lane_safe(px, py, nx, ny):
                    return True
                return False

            action_taken = False
            if self.prefer_zigzag_correction_over_deletion:
                corr = attempt_correction()
                if corr:
                    sg[i] = corr
                    n_corrected += 1
                    action_taken = True
                else:
                    if attempt_deletion():
                        sg.pop(i)
                        n_removed += 1
                        action_taken = True
            else:
                if attempt_deletion():
                    sg.pop(i)
                    n_removed += 1
                    action_taken = True
                else:
                    corr = attempt_correction()
                    if corr:
                        sg[i] = corr
                        n_corrected += 1
                        action_taken = True

            if not action_taken:
                n_rejected += 1

            i -= 1

        # ── Recalculate orientations ─────────────────────────────────────
        for i in range(len(sg) - 1):
            sx, sy, _ = sg[i]
            nx2, ny2, _ = sg[i + 1]
            yaw = math.atan2(ny2 - sy, nx2 - sx)
            sg[i] = (sx, sy, yaw)
        # Restore last semi-goal orientation (from /bt/goal)
        if sg:
            lx, ly, _ = sg[-1]
            sg[-1] = (lx, ly, last_yaw)

        self.semi_goals = sg

        self.get_logger().info(
            f'[ZIGZAG_CLEANUP] '
            f'before={n_before}, '
            f'zigzags_detected={n_zigzags_detected}, '
            f'corrected={n_corrected}, '
            f'removed={n_removed}, '
            f'blocked_limit={n_blocked_by_limit}, '
            f'blocked_length={n_blocked_by_length}, '
            f'blocked_min_keep={n_blocked_by_min_keep}, '
            f'rejected_unsafe={n_rejected}, '
            f'after={len(sg)}'
        )

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
            _sc = getattr(self, 'semigoal_marker_scale', 0.12)
            m.scale.x = _sc
            m.scale.y = _sc
            m.scale.z = _sc

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
