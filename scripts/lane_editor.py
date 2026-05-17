#!/usr/bin/env python3
import os
import math
import cv2
import yaml
import shutil

# =============================================================================
# Lane Editor for qcar2_planner
# =============================================================================
# Controls:
#   Left Click : Add point to current lane
#   n          : Save current lane into memory
#   u          : Undo last point of current lane
#   s          : Save all lanes to lanes.yaml
#   q          : Quit
#
# Behavior:
#   - Loads existing lanes.yaml automatically.
#   - Adds new lanes without deleting previous lanes.
#   - Creates lanes.yaml.bak before saving.
#   - Can auto-close loops by appending the first point at the end.
# =============================================================================

current_points = []
lanes = []

resolution = 0.05
pgm_origin_x = 0.0
pgm_origin_y = 0.0

scaled_h = 0
scaled_w = 0
scaled_img = None

DEFAULT_LANE_WIDTH = 0.80
AUTO_CLOSE_THRESHOLD_M = 0.35
SUCCESSOR_THRESHOLD_M = 0.30


def pixel_to_meter(px, py):
    y_idx = scaled_h - 1 - py

    x_m = pgm_origin_x + (px + 0.5) * resolution
    y_m = pgm_origin_y + (y_idx + 0.5) * resolution

    return round(x_m, 3), round(y_m, 3)


def meter_to_pixel(mx, my):
    px = int((mx - pgm_origin_x) / resolution - 0.5)

    y_idx = int((my - pgm_origin_y) / resolution - 0.5)
    py = scaled_h - 1 - y_idx

    return px, py


def distance_2d(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return (dx * dx + dy * dy) ** 0.5


def maybe_close_loop(points, threshold=AUTO_CLOSE_THRESHOLD_M):
    """
    If the last point is close to the first one, append the first point
    to close the lane loop.
    """
    if len(points) < 3:
        return points

    first = points[0]
    last = points[-1]

    if first == last:
        return points

    dist = distance_2d(first, last)

    if dist <= threshold:
        print(f"Loop closed automatically. Endpoint distance = {dist:.3f} m")
        return points + [first]

    print(f"Loop not closed. Endpoint distance = {dist:.3f} m")
    print(f"Threshold is {threshold:.3f} m. Add one point closer to the start if you want auto-close.")
    return points


def load_existing_lanes(lanes_yaml_path):
    global lanes

    if not os.path.exists(lanes_yaml_path):
        print(f"No existing lanes.yaml found. Starting with 0 lanes.")
        lanes = []
        return

    try:
        with open(lanes_yaml_path, "r") as f:
            data = yaml.safe_load(f) or {}

        loaded_lanes = data.get("lanes", [])

        if not isinstance(loaded_lanes, list):
            print("Warning: lanes.yaml has invalid format. Starting with 0 lanes.")
            lanes = []
            return

        clean_lanes = []

        for i, lane in enumerate(loaded_lanes):
            if not isinstance(lane, dict):
                continue

            points = lane.get("points", [])

            if not isinstance(points, list) or len(points) < 2:
                continue

            name = lane.get("name", lane.get("id", f"lane_{i + 1}"))

            try:
                width = float(lane.get("width", DEFAULT_LANE_WIDTH))
            except Exception:
                width = DEFAULT_LANE_WIDTH

            clean_lanes.append({
                "name": name,
                "points": points,
                "width": width
            })

        lanes = clean_lanes
        print(f"Loaded {len(lanes)} existing lanes from {lanes_yaml_path}")

    except Exception as e:
        print(f"Failed to load existing lanes.yaml: {e}")
        lanes = []


def _detect_successors(all_lanes, threshold=SUCCESSOR_THRESHOLD_M):
    """
    For each lane, detect which other lanes are valid successors based on:
      - Distance from this lane's last point to another lane's point (not final) <= threshold
      - Angle consistency: the dot product of the exit vector and the entry vector >= 0
        (i.e. traffic flows in the same general direction, no head-on connections).
    Returns the lanes list with 'successors' field populated.
    """
    for lane in all_lanes:
        points = lane.get("points", [])
        if len(points) < 2:
            lane["successors"] = []
            continue

        last_pt = points[-1]
        prev_pt = points[-2]
        vA_x = last_pt[0] - prev_pt[0]
        vA_y = last_pt[1] - prev_pt[1]

        successors = set()

        for other in all_lanes:
            if other.get("name") == lane.get("name"):
                continue

            o_points = other.get("points", [])
            if len(o_points) < 2:
                continue

            for i in range(len(o_points) - 1):
                pt = o_points[i]
                dist = math.sqrt(
                    (last_pt[0] - pt[0]) ** 2 + (last_pt[1] - pt[1]) ** 2
                )
                if dist <= threshold:
                    next_pt = o_points[i + 1]
                    vB_x = next_pt[0] - pt[0]
                    vB_y = next_pt[1] - pt[1]
                    dot = vA_x * vB_x + vA_y * vB_y
                    if dot >= 0:
                        successors.add(other["name"])
                    break

        lane["successors"] = sorted(list(successors))

    return all_lanes


def save_lanes(lanes_yaml_path):
    if os.path.exists(lanes_yaml_path):
        backup_path = lanes_yaml_path + ".bak"
        shutil.copyfile(lanes_yaml_path, backup_path)
        print(f"Backup created: {backup_path}")

    # Auto-detect successors before saving
    _detect_successors(lanes, SUCCESSOR_THRESHOLD_M)
    n_with = sum(1 for l in lanes if l.get("successors"))
    print(f"Successors auto-detected for {n_with}/{len(lanes)} lanes (threshold={SUCCESSOR_THRESHOLD_M}m)")

    out_data = {
        "lanes": lanes
    }

    with open(lanes_yaml_path, "w") as f:
        yaml.dump(out_data, f, default_flow_style=False, sort_keys=False)

    print(f"Saved {len(lanes)} lanes to {lanes_yaml_path}")


def redraw(img):
    display = img.copy()

    # Draw saved lanes
    for lane in lanes:
        pts = [meter_to_pixel(p[0], p[1]) for p in lane["points"]]

        for i in range(len(pts) - 1):
            cv2.line(display, pts[i], pts[i + 1], (0, 255, 0), 2)

        for p in pts:
            cv2.circle(display, p, 3, (0, 0, 255), -1)

        if pts:
            cv2.putText(
                display,
                str(lane.get("name", "lane")),
                pts[0],
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )

    # Draw current unsaved lane
    pts = [meter_to_pixel(p[0], p[1]) for p in current_points]

    for i in range(len(pts) - 1):
        cv2.line(display, pts[i], pts[i + 1], (255, 0, 0), 2)

    for p in pts:
        cv2.circle(display, p, 4, (0, 165, 255), -1)

    if pts:
        cv2.putText(
            display,
            "current_lane",
            pts[0],
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 0, 0),
            1,
            cv2.LINE_AA
        )

    cv2.imshow("Lane Editor", display)


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        mx, my = pixel_to_meter(x, y)
        current_points.append([mx, my])

        print(f"Point added: [{mx}, {my}]")
        redraw(scaled_img)


def main():
    global resolution, pgm_origin_x, pgm_origin_y
    global scaled_h, scaled_w, scaled_img

    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    map_yaml_path = os.path.join(src_dir, "config", "map_ros.yaml")
    params_yaml_path = os.path.join(src_dir, "config", "params.yaml")
    lanes_yaml_path = os.path.join(src_dir, "config", "lanes.yaml")

    if not os.path.exists(map_yaml_path):
        print(f"Cannot find map_ros.yaml at {map_yaml_path}")
        return

    with open(map_yaml_path, "r") as f:
        map_meta = yaml.safe_load(f) or {}

    pgm_name = map_meta.get("image", "map_ros.pgm")
    resolution = float(map_meta.get("resolution", 0.05))

    origin = map_meta.get("origin", [0.0, 0.0, 0.0])
    pgm_origin_x = float(origin[0])
    pgm_origin_y = float(origin[1])

    pgm_scale_factor = 1.0

    if os.path.exists(params_yaml_path):
        with open(params_yaml_path, "r") as f:
            params = yaml.safe_load(f) or {}

        try:
            pgm_scale_factor = float(
                params["map_overlay_node"]["ros__parameters"]["pgm_scale_factor"]
            )
        except KeyError:
            print("pgm_scale_factor not found in params.yaml. Using 1.0")

    pgm_path = os.path.join(os.path.dirname(map_yaml_path), pgm_name)

    raw_img = cv2.imread(pgm_path, cv2.IMREAD_COLOR)

    if raw_img is None:
        print(f"Failed to load image: {pgm_path}")
        return

    if pgm_scale_factor != 1.0:
        scaled_w = int(raw_img.shape[1] * pgm_scale_factor)
        scaled_h = int(raw_img.shape[0] * pgm_scale_factor)

        scaled_img = cv2.resize(
            raw_img,
            (scaled_w, scaled_h),
            interpolation=cv2.INTER_NEAREST
        )

        print(f"PGM scale factor: {pgm_scale_factor}")
        print(f"Original size: {raw_img.shape[1]}x{raw_img.shape[0]}")
        print(f"Scaled size: {scaled_w}x{scaled_h}")
    else:
        scaled_img = raw_img
        scaled_h, scaled_w = scaled_img.shape[:2]

        print("PGM scale factor: 1.0")
        print(f"Image size: {scaled_w}x{scaled_h}")

    load_existing_lanes(lanes_yaml_path)

    cv2.namedWindow("Lane Editor", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Lane Editor", mouse_callback)

    print("")
    print("Lane Editor Controls:")
    print("  Left Click : Add point to current lane")
    print("  n          : Save current lane into memory")
    print("  u          : Undo last point of current lane")
    print("  s          : Save all lanes to lanes.yaml")
    print("  q          : Quit")
    print("")
    print("Important:")
    print("  Existing lanes are loaded automatically.")
    print("  Pressing 's' saves existing lanes + new lanes.")
    print("  A lane is not added until you press 'n'.")
    print("  If the last point is near the first point, the loop can be auto-closed.")
    print("")

    redraw(scaled_img)

    while True:
        key = cv2.waitKey(10) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("u"):
            if current_points:
                removed = current_points.pop()
                print(f"Removed point: {removed}")
                redraw(scaled_img)
            else:
                print("No current points to undo.")

        elif key == ord("n"):
            if len(current_points) >= 2:
                name = input("Enter lane name: ").strip()

                if not name:
                    name = f"lane_{len(lanes) + 1}"

                width_str = input(
                    f"Enter lane width in meters default {DEFAULT_LANE_WIDTH}: "
                ).strip()

                try:
                    width = float(width_str) if width_str else DEFAULT_LANE_WIDTH
                except ValueError:
                    print(f"Invalid width. Using default {DEFAULT_LANE_WIDTH}")
                    width = DEFAULT_LANE_WIDTH

                close_str = input(
                    "Close loop automatically if endpoints are near? [Y/n]: "
                ).strip().lower()

                close_loop = close_str in ["", "y", "yes", "s", "si", "sí"]

                lane_points = current_points.copy()

                if close_loop:
                    before_len = len(lane_points)
                    lane_points = maybe_close_loop(lane_points)

                    if len(lane_points) > before_len:
                        print("First point appended at the end.")

                lanes.append({
                    "name": name,
                    "points": lane_points,
                    "width": width
                })

                print(
                    f"Lane '{name}' added with {len(lane_points)} points "
                    f"and width={width}."
                )

                current_points.clear()
                redraw(scaled_img)

            else:
                print("A lane must have at least 2 points.")

        elif key == ord("s"):
            if current_points:
                print("Warning: You have unsaved current points.")
                print("Press 'n' first to add the current lane, then press 's' to save.")
            else:
                save_lanes(lanes_yaml_path)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()