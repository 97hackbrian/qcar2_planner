#!/usr/bin/env python3
# =============================================================================
# map_processor_node.py — Perception Engine (Node 1)
# =============================================================================
#
# Base grid comes from /map (Cartographer). nvblox mesh overlay determines
# which FREE cells have been 3D-scanned. Uncertainty = unscanned free cells.
#
# LAYERS published via grid_map_msgs/GridMap:
#   occupancy   — from /map: 0=free, 1=wall, -1=unknown
#   nvblox_hits — count of nvblox mesh vertices per cell
#   uncertainty — 1.0 = free but not scanned, 0.0 = fully scanned
#   dir_x/dir_y — legal direction unit vector (from robot TF heading)
#
# RViz2 outputs via nav_msgs/OccupancyGrid:
#   /planner_occupancy  — walls/free from Cartographer
#   /planner_uncertainty — nvblox coverage heatmap (0=scanned, 100=unscanned)
#
# GUARDRAIL: This node does NOT publish any motor commands.
# =============================================================================

import math
import traceback
import numpy as np
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import OccupancyGrid
from grid_map_msgs.msg import GridMap as GridMapMsg
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

import tf2_ros
from tf2_ros import Buffer, TransformListener


class MapProcessorNode(Node):
    """Fuses /map (Cartographer) with nvblox mesh to compute uncertainty."""

    def __init__(self):
        super().__init__('map_processor_node')
        self.get_logger().info('=== MapProcessorNode __init__ START ===')

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter('mesh_marker_topic', '/nvblox_node/mesh_marker')
        self.declare_parameter('occupancy_grid_topic', '/map')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'odom')
        self.declare_parameter('map_resolution', 0.05)
        self.declare_parameter('map_size_x', 20.0)
        self.declare_parameter('map_size_y', 20.0)
        self.declare_parameter('map_origin_x', 0.0)
        self.declare_parameter('map_origin_y', 0.0)
        self.declare_parameter('z_min', -0.5)
        self.declare_parameter('z_max', 2.0)
        self.declare_parameter('hits_threshold', 10)
        self.declare_parameter('publish_rate', 2.0)
        self.declare_parameter('wall_occupy_threshold', 50)
        self.declare_parameter('occupancy_out_topic', '/planner_occupancy')
        self.declare_parameter('uncertainty_out_topic', '/planner_uncertainty')

        self.mesh_topic = str(self.get_parameter('mesh_marker_topic').value)
        self.occgrid_topic = str(self.get_parameter('occupancy_grid_topic').value)
        self.map_frame = str(self.get_parameter('map_frame').value)
        self.base_frame = str(self.get_parameter('base_frame').value)
        self.resolution = float(self.get_parameter('map_resolution').value)
        self.size_x = float(self.get_parameter('map_size_x').value)
        self.size_y = float(self.get_parameter('map_size_y').value)
        self.origin_x = float(self.get_parameter('map_origin_x').value)
        self.origin_y = float(self.get_parameter('map_origin_y').value)
        self.z_min = float(self.get_parameter('z_min').value)
        self.z_max = float(self.get_parameter('z_max').value)
        self.hits_threshold = int(self.get_parameter('hits_threshold').value)
        self.publish_rate = float(self.get_parameter('publish_rate').value)
        self.wall_threshold = int(self.get_parameter('wall_occupy_threshold').value)
        occ_out = str(self.get_parameter('occupancy_out_topic').value)
        unc_out = str(self.get_parameter('uncertainty_out_topic').value)

        # ── Grid ────────────────────────────────────────────────────────────
        self.cols = int(self.size_x / self.resolution)
        self.rows = int(self.size_y / self.resolution)
        self.corner_x = self.origin_x - self.size_x / 2.0
        self.corner_y = self.origin_y - self.size_y / 2.0

        self.get_logger().info(
            f'Grid: {self.cols}x{self.rows} cells, '
            f'corner=({self.corner_x:.1f}, {self.corner_y:.1f})'
        )

        # Layers — start all unknown
        self.occupancy = np.full((self.rows, self.cols), -1.0, dtype=np.float32)
        self.nvblox_hits = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.uncertainty = np.ones((self.rows, self.cols), dtype=np.float32)
        self.dir_x = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.dir_y = np.zeros((self.rows, self.cols), dtype=np.float32)

        self.lock = threading.Lock()
        self.got_map = False
        self.publish_count = 0

        # ── TF2 ─────────────────────────────────────────────────────────────
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── Subscribers ─────────────────────────────────────────────────────
        qos_mesh = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.mesh_sub = self.create_subscription(
            MarkerArray, self.mesh_topic, self.mesh_callback, qos_mesh
        )
        self.get_logger().info(f'Subscribed to mesh: {self.mesh_topic}')

        qos_map = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.map_sub = self.create_subscription(
            OccupancyGrid, self.occgrid_topic,
            self.occupancy_grid_callback, qos_map
        )
        self.get_logger().info(f'Subscribed to map: {self.occgrid_topic}')

        # ── Publishers ──────────────────────────────────────────────────────
        self.gridmap_pub = self.create_publisher(GridMapMsg, '/grid_map', 10)
        self.occ_pub = self.create_publisher(OccupancyGrid, occ_out, 10)
        self.unc_pub = self.create_publisher(OccupancyGrid, unc_out, 10)
        self.get_logger().info(
            f'Publishers: /grid_map, {occ_out}, {unc_out}'
        )

        # ── Timers ──────────────────────────────────────────────────────────
        period = 1.0 / max(0.1, self.publish_rate)
        self.pub_timer = self.create_timer(period, self.publish_all)
        self.dir_timer = self.create_timer(0.2, self.record_direction)
        self.get_logger().info(
            f'Timers: publish every {period:.2f}s, direction every 0.2s'
        )

        self.get_logger().info('=== MapProcessorNode __init__ DONE ===')

    # =====================================================================
    # /map callback — Cartographer → occupancy base layer (vectorized)
    # =====================================================================
    def occupancy_grid_callback(self, msg: OccupancyGrid):
        try:
            og_res = msg.info.resolution
            og_w = msg.info.width
            og_h = msg.info.height
            og_ox = msg.info.origin.position.x
            og_oy = msg.info.origin.position.y

            raw = np.array(msg.data, dtype=np.int8).reshape((og_h, og_w))

            # World coords for each /map cell
            gx_world = og_ox + (np.arange(og_w, dtype=np.float64) + 0.5) * og_res
            gy_world = og_oy + (np.arange(og_h, dtype=np.float64) + 0.5) * og_res

            # Map to our grid
            col_idx = ((gx_world - self.corner_x) / self.resolution).astype(int)
            row_idx = ((gy_world - self.corner_y) / self.resolution).astype(int)

            c_ok = np.where((col_idx >= 0) & (col_idx < self.cols))[0]
            r_ok = np.where((row_idx >= 0) & (row_idx < self.rows))[0]

            if len(c_ok) == 0 or len(r_ok) == 0:
                self.get_logger().warn('/map has no overlap with our grid!')
                return

            sub = raw[np.ix_(r_ok, c_ok)]
            dst_c = col_idx[c_ok]
            dst_r = row_idx[r_ok]

            result = np.full(sub.shape, -1.0, dtype=np.float32)
            result[(sub >= 0) & (sub < self.wall_threshold)] = 0.0
            result[sub >= self.wall_threshold] = 1.0

            with self.lock:
                self.occupancy[np.ix_(dst_r, dst_c)] = result

            self.got_map = True
            n_free = int(np.sum(result == 0.0))
            n_wall = int(np.sum(result == 1.0))
            self.get_logger().info(
                f'/map integrated: {og_w}x{og_h} → '
                f'free={n_free}, wall={n_wall}',
                throttle_duration_sec=3.0,
            )
        except Exception as e:
            self.get_logger().error(
                f'occupancy_grid_callback ERROR: {e}\n{traceback.format_exc()}'
            )

    # =====================================================================
    # Mesh callback — nvblox 3D → 2D scanned overlay (vectorized)
    # =====================================================================
    def mesh_callback(self, msg: MarkerArray):
        try:
            total = 0
            with self.lock:
                for marker in msg.markers:
                    if marker.type not in (Marker.TRIANGLE_LIST, Marker.POINTS):
                        continue
                    if len(marker.points) == 0:
                        continue

                    # Batch → (N,3)
                    pts = np.array(
                        [[p.x, p.y, p.z] for p in marker.points],
                        dtype=np.float64,
                    )

                    # Marker pose
                    mq = marker.pose.orientation
                    mp = marker.pose.position
                    has_rot = not (
                        mq.x == 0.0 and mq.y == 0.0 and
                        mq.z == 0.0 and mq.w == 1.0
                    )
                    if has_rot:
                        pts = self._qrot(pts, mq.x, mq.y, mq.z, mq.w)

                    pts[:, 0] += mp.x
                    pts[:, 1] += mp.y
                    pts[:, 2] += mp.z

                    # Z-filter
                    z_ok = (pts[:, 2] >= self.z_min) & (pts[:, 2] <= self.z_max)
                    pts = pts[z_ok]
                    if len(pts) == 0:
                        continue

                    # World → grid
                    cols = ((pts[:, 0] - self.corner_x) / self.resolution).astype(int)
                    rows = ((pts[:, 1] - self.corner_y) / self.resolution).astype(int)

                    ok = (
                        (cols >= 0) & (cols < self.cols) &
                        (rows >= 0) & (rows < self.rows)
                    )
                    cols, rows = cols[ok], rows[ok]
                    if len(cols) == 0:
                        continue

                    # Hits on free cells only
                    free = self.occupancy[rows, cols] == 0.0
                    np.add.at(self.nvblox_hits, (rows[free], cols[free]), 1.0)
                    total += int(np.sum(free))

                # Recompute uncertainty on free cells
                ratio = np.minimum(
                    1.0, self.nvblox_hits / max(1.0, float(self.hits_threshold))
                )
                self.uncertainty = np.where(
                    self.occupancy == 0.0, 1.0 - ratio, 1.0
                )

            if total > 0:
                self.get_logger().info(
                    f'nvblox: {total} hits on free cells',
                    throttle_duration_sec=2.0,
                )
        except Exception as e:
            self.get_logger().error(
                f'mesh_callback ERROR: {e}\n{traceback.format_exc()}'
            )

    @staticmethod
    def _qrot(pts, qx, qy, qz, qw):
        """Batch quaternion rotation on (N,3)."""
        px, py, pz = pts[:, 0], pts[:, 1], pts[:, 2]
        tx = 2.0 * (qy * pz - qz * py)
        ty = 2.0 * (qz * px - qx * pz)
        tz = 2.0 * (qx * py - qy * px)
        return np.column_stack((
            px + qw * tx + (qy * tz - qz * ty),
            py + qw * ty + (qz * tx - qx * tz),
            pz + qw * tz + (qx * ty - qy * tx),
        ))

    # =====================================================================
    # Direction recording (TF heading → dir_x/dir_y)
    # =====================================================================
    def record_direction(self):
        try:
            t = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time()
            )
        except Exception:
            return  # TF not ready yet, silently skip

        q = t.transform.rotation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        dx, dy = math.cos(yaw), math.sin(yaw)
        wx = t.transform.translation.x
        wy = t.transform.translation.y

        col = int((wx - self.corner_x) / self.resolution)
        row = int((wy - self.corner_y) / self.resolution)
        if not (0 <= col < self.cols and 0 <= row < self.rows):
            return

        with self.lock:
            self.dir_x[row, col] = dx
            self.dir_y[row, col] = dy
            r0, r1 = max(0, row - 3), min(self.rows, row + 4)
            c0, c1 = max(0, col - 3), min(self.cols, col + 4)
            p_occ = self.occupancy[r0:r1, c0:c1]
            p_dx = self.dir_x[r0:r1, c0:c1]
            p_dy = self.dir_y[r0:r1, c0:c1]
            empty = (p_occ == 0.0) & (p_dx == 0.0) & (p_dy == 0.0)
            p_dx[empty] = dx
            p_dy[empty] = dy

    # =====================================================================
    # Publish ALL outputs (timer callback)
    # =====================================================================
    def publish_all(self):
        self.publish_count += 1

        try:
            with self.lock:
                occ = self.occupancy.copy()
                hits = self.nvblox_hits.copy()
                unc = self.uncertainty.copy()
                dx = self.dir_x.copy()
                dy = self.dir_y.copy()

            stamp = self.get_clock().now().to_msg()

            # 1) OccupancyGrid — occupancy
            self._pub_occ(stamp, occ)

            # 2) OccupancyGrid — uncertainty
            self._pub_unc(stamp, occ, unc)

            # 3) GridMap — full multi-layer
            self._pub_gridmap(stamp, occ, hits, unc, dx, dy)

            # Diagnostic every 5s
            n_free = int(np.sum(occ == 0.0))
            n_scanned = int(np.sum((occ == 0.0) & (hits > 0)))
            self.get_logger().info(
                f'[pub #{self.publish_count}] free={n_free} '
                f'scanned={n_scanned} got_map={self.got_map}',
                throttle_duration_sec=5.0,
            )

        except Exception as e:
            self.get_logger().error(
                f'publish_all ERROR: {e}\n{traceback.format_exc()}'
            )

    # ── OccupancyGrid publishers ────────────────────────────────────────
    def _pub_occ(self, stamp, occ):
        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = self.map_frame
        msg.info.resolution = float(self.resolution)
        msg.info.width = int(self.cols)
        msg.info.height = int(self.rows)
        msg.info.origin.position.x = float(self.corner_x)
        msg.info.origin.position.y = float(self.corner_y)
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.x = 0.0
        msg.info.origin.orientation.y = 0.0
        msg.info.origin.orientation.z = 0.0
        msg.info.origin.orientation.w = 1.0

        grid = np.full(self.rows * self.cols, -1, dtype=np.int8)
        flat_occ = occ.ravel()
        grid[flat_occ == 0.0] = 0
        grid[flat_occ == 1.0] = 100
        msg.data = grid.tolist()
        self.occ_pub.publish(msg)

    def _pub_unc(self, stamp, occ, unc):
        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = self.map_frame
        msg.info.resolution = float(self.resolution)
        msg.info.width = int(self.cols)
        msg.info.height = int(self.rows)
        msg.info.origin.position.x = float(self.corner_x)
        msg.info.origin.position.y = float(self.corner_y)
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.x = 0.0
        msg.info.origin.orientation.y = 0.0
        msg.info.origin.orientation.z = 0.0
        msg.info.origin.orientation.w = 1.0

        flat_occ = occ.ravel()
        flat_unc = unc.ravel()
        grid = np.full(self.rows * self.cols, -1, dtype=np.int8)
        free_mask = flat_occ == 0.0
        grid[free_mask] = np.clip(flat_unc[free_mask] * 100.0, 0, 100).astype(
            np.int8
        )
        msg.data = grid.tolist()
        self.unc_pub.publish(msg)

    # ── GridMap publisher ───────────────────────────────────────────────
    def _pub_gridmap(self, stamp, occ, hits, unc, dx, dy):
        msg = GridMapMsg()
        msg.header.stamp = stamp
        msg.header.frame_id = self.map_frame

        msg.info.resolution = float(self.resolution)
        msg.info.length_x = float(self.size_x)
        msg.info.length_y = float(self.size_y)
        msg.info.pose.position.x = float(self.origin_x)
        msg.info.pose.position.y = float(self.origin_y)
        msg.info.pose.position.z = 0.0
        msg.info.pose.orientation.w = 1.0

        names = ['occupancy', 'nvblox_hits', 'uncertainty', 'dir_x', 'dir_y']
        arrays = [occ, hits, unc, dx, dy]
        msg.layers = names
        msg.basic_layers = ['occupancy', 'uncertainty']

        for layer_data in arrays:
            arr = Float32MultiArray()
            # grid_map Eigen convention:
            #   Eigen rows = n_x (cells along X), row 0 = max X
            #   Eigen cols = n_y (cells along Y), col 0 = max Y
            #   dim[0] = column_index = Eigen cols = n_y
            #   dim[1] = row_index   = Eigen rows = n_x
            n_x = self.cols  # cells along X
            n_y = self.rows  # cells along Y
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
            # numpy[y,x] row0=minY, col0=minX → Eigen[x,y] row0=maxX, col0=maxY
            eigen_data = layer_data[::-1, ::-1].T.astype(np.float32)
            arr.data = eigen_data.flatten(order='F').tolist()
            msg.data.append(arr)

        msg.outer_start_index = 0
        msg.inner_start_index = 0
        self.gridmap_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MapProcessorNode()
    node.get_logger().info('Entering spin loop...')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
