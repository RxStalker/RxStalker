"""
Build a Wi-Fi fingerprint gradient map from per-AP reference points measurements.
"""

import ast
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.linalg import norm
from sklearn.linear_model import LinearRegression

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils import const

DATASET_DIR = PROJECT_DIR / "dataset"
INPUT_CSV = DATASET_DIR / "wifi_fingerprinting.csv"
OUTPUT_CSV = DATASET_DIR / "gradient_map.csv"


def build_room_grid():
    """Divide room into cells; x left->right, y top->bottom."""
    num_x = int(const.room_width / const.cell_width)
    num_y = int(const.room_length / const.cell_length)
    cells = [(x, y) for y in range(num_y) for x in range(num_x)]
    return num_x, num_y, cells


def parse_loc(value):
    if isinstance(value, (tuple, list)):
        return int(value[0]), int(value[1])
    text = str(value).strip().strip("()[]")
    parts = text.split(",")
    return int(float(parts[0].strip())), int(float(parts[1].strip()))


def parse_series(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=float)
    return np.asarray(ast.literal_eval(str(value)), dtype=float)


def hampel_filter(series, window=31, n_sigmas=3.0):
    """Remove multipath outliers in RSSI (Hampel filter)."""
    x = np.asarray(series, dtype=float)
    n = len(x)
    if n == 0:
        return x
    if n < window:
        window = max(3, n if n % 2 == 1 else n - 1)

    filtered = x.copy()
    half = window // 2
    for i in range(n):
        segment = x[max(0, i - half) : min(n, i + half + 1)]
        median = np.median(segment)
        mad = np.median(np.abs(segment - median))
        if mad < 1e-9:
            continue
        if abs(x[i] - median) > n_sigmas * 1.4826 * mad:
            filtered[i] = median
    return filtered


def denoise_rssi(rssi_series):
    """Denoise RSSI time series before gradient construction."""
    x = hampel_filter(parse_series(rssi_series))
    if len(x) >= 5:
        kernel = np.ones(5) / 5
        x = np.convolve(x, kernel, mode="same")
    return x


def denoise_rtt(rtt_series):
    """RTT is already denoised in preprocessing; keep full series."""
    return parse_series(rtt_series)


def fit_line(x, y):
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    return model.coef_[0], model.intercept_


def calculate_rssi_gradient(rssi1, rssi2):
    """Slope-direction gradient between two RSSI series (same AP)."""
    a = np.asarray(rssi1, dtype=float)
    b = np.asarray(rssi2, dtype=float)
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    a, b = a[:n], b[:n]
    x = np.arange(n)
    slope1, _ = fit_line(x, a)
    slope2, _ = fit_line(x, b)
    return float(norm(np.array([1.0, slope1]) - np.array([1.0, slope2])))


def calculate_rtt_gradient(rtt1, rtt2):
    a = np.asarray(rtt1, dtype=float)
    b = np.asarray(rtt2, dtype=float)
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    a, b = a[:n], b[:n]
    x = np.arange(n)
    slope1, _ = fit_line(x, a)
    slope2, _ = fit_line(x, b)
    return float(norm(np.array([1.0, slope1]) - np.array([1.0, slope2])))


def extrapolate_series(base, neighbor, steps):
    """
    Per-sample linear extrapolation along one grid step.
    More precise than applying a single scalar gradient to the full series.
    """
    if steps == 0:
        return np.asarray(base, dtype=float)

    base_arr = np.asarray(base, dtype=float)
    neighbor_arr = np.asarray(neighbor, dtype=float)
    n = min(len(base_arr), len(neighbor_arr))
    if n == 0:
        return base_arr

    rate = neighbor_arr[:n] - base_arr[:n]
    result = base_arr.copy()
    result[:n] += rate * steps
    return result


def load_fingerprints(csv_path):
    """
    Index fingerprints by (location, AP) so each reference point keeps
    a separate Wi-Fi profile per AP.
    """
    df = pd.read_csv(csv_path)
    measured = {}
    by_rf_ap = {}

    for _, row in df.iterrows():
        loc = parse_loc(row["rf_loc"])
        rf_id = int(row["rf_id"])
        ap_mac = str(row["ap_mac"]).strip()
        entry = {
            "ap_mac": ap_mac,
            "ap_loc": row["ap_loc"],
            "rf_id": rf_id,
            "loc": loc,
            "rssi": denoise_rssi(row["rssi"]),
            "rtt": denoise_rtt(row["rtt"]),
        }
        measured[(loc, ap_mac)] = entry
        by_rf_ap.setdefault(rf_id, {}).setdefault(ap_mac, {})[loc] = entry

    return measured, by_rf_ap


def collect_ap_macs(by_rf_ap):
    ap_macs = set()
    for ap_data in by_rf_ap.values():
        ap_macs.update(ap_data.keys())
    return sorted(ap_macs)


def neighbor_offsets():
    return {
        "left": (-1, 0),
        "right": (1, 0),
        "top": (0, -1),
        "bottom": (0, 1),
    }


def find_center_loc(locs):
    for loc in locs:
        x, y = loc
        needed = {(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)}
        if needed.issubset(locs.keys()):
            return loc
    return sorted(locs.keys())[0]


def get_reference_bundles(by_rf_ap):
    """
    Build bundles per AP and reference point:
    {ap_mac: {rf_id: {center, neighbors}}}.
    """
    bundles = {}
    offsets = neighbor_offsets()

    for rf_id in range(1, const.reference_point_number + 1):
        if rf_id not in by_rf_ap:
            continue

        for ap_mac, locs in by_rf_ap[rf_id].items():
            center_loc = find_center_loc(locs)
            x, y = center_loc
            neighbors = {}
            for name, (dx, dy) in offsets.items():
                nloc = (x + dx, y + dy)
                if nloc in locs:
                    neighbors[name] = locs[nloc]

            bundles.setdefault(ap_mac, {})[rf_id] = {
                "rf_id": rf_id,
                "ap_mac": ap_mac,
                "center": locs[center_loc],
                "neighbors": neighbors,
            }

    return bundles


def apply_axis_shift(est_rssi, est_rtt, center, neighbor, steps, use_series=True):
    """Apply one-axis gradient shift using per-sample and slope-based terms."""
    if steps == 0 or neighbor is None:
        return est_rssi, est_rtt

    if use_series:
        est_rssi = extrapolate_series(est_rssi, neighbor["rssi"], steps)
        est_rtt = extrapolate_series(est_rtt, neighbor["rtt"], steps)
    else:
        est_rssi = est_rssi + calculate_rssi_gradient(neighbor["rssi"], center["rssi"]) * steps
        est_rtt = est_rtt + calculate_rtt_gradient(neighbor["rtt"], center["rtt"]) * steps

    return est_rssi, est_rtt


def estimate_from_reference(cell, bundle):
    """
    Estimate fingerprint at cell from one reference point + neighbors (same AP).
    y increases top->bottom, x increases left->right.
    """
    cx, cy = cell
    rx, ry = bundle["center"]["loc"]
    center = bundle["center"]
    neighbors = bundle["neighbors"]

    est_rssi = np.asarray(center["rssi"], dtype=float).copy()
    est_rtt = np.asarray(center["rtt"], dtype=float).copy()

    if cy > ry and cx > rx:
        mov_y, mov_x = cy - ry, cx - rx
        est_rssi, est_rtt = apply_axis_shift(
            est_rssi, est_rtt, center, neighbors.get("bottom"), mov_y
        )
        est_rssi, est_rtt = apply_axis_shift(
            est_rssi, est_rtt, center, neighbors.get("right"), mov_x
        )

    elif cy > ry and cx < rx:
        mov_y, mov_x = cy - ry, rx - cx
        est_rssi, est_rtt = apply_axis_shift(
            est_rssi, est_rtt, center, neighbors.get("bottom"), mov_y
        )
        est_rssi, est_rtt = apply_axis_shift(
            est_rssi, est_rtt, center, neighbors.get("left"), mov_x
        )

    elif cy < ry and cx > rx:
        mov_y, mov_x = ry - cy, cx - rx
        est_rssi, est_rtt = apply_axis_shift(
            est_rssi, est_rtt, center, neighbors.get("top"), mov_y
        )
        est_rssi, est_rtt = apply_axis_shift(
            est_rssi, est_rtt, center, neighbors.get("right"), mov_x
        )

    elif cy < ry and cx < rx:
        mov_y, mov_x = ry - cy, rx - cx
        est_rssi, est_rtt = apply_axis_shift(
            est_rssi, est_rtt, center, neighbors.get("top"), mov_y
        )
        est_rssi, est_rtt = apply_axis_shift(
            est_rssi, est_rtt, center, neighbors.get("left"), mov_x
        )

    elif cy > ry:
        est_rssi, est_rtt = apply_axis_shift(
            est_rssi, est_rtt, center, neighbors.get("bottom"), cy - ry
        )

    elif cy < ry:
        est_rssi, est_rtt = apply_axis_shift(
            est_rssi, est_rtt, center, neighbors.get("top"), ry - cy
        )

    elif cx > rx:
        est_rssi, est_rtt = apply_axis_shift(
            est_rssi, est_rtt, center, neighbors.get("right"), cx - rx
        )

    elif cx < rx:
        est_rssi, est_rtt = apply_axis_shift(
            est_rssi, est_rtt, center, neighbors.get("left"), rx - cx
        )

    return est_rssi, est_rtt


def reference_weight(cell, ref_loc):
    """Inverse-distance weight; closer references contribute more."""
    cx, cy = cell
    rx, ry = ref_loc
    dist = math.sqrt((cx - rx) ** 2 + (cy - ry) ** 2)
    return 1.0 / (dist + 1.0) ** 2


def fuse_weighted_series(candidates):
    """
    Fuse multiple (series, weight) tuples from different reference points.
    """
    if not candidates:
        return np.array([]), np.array([])

    total_w = sum(weight for _, weight in candidates)
    if total_w <= 0:
        return candidates[0][0], candidates[0][0]

    length = max(len(item[0]) for item in candidates)
    fused = np.zeros(length, dtype=float)
    for series, weight in candidates:
        arr = np.asarray(series, dtype=float)
        if len(arr) < length:
            arr = np.pad(arr, (0, length - len(arr)), mode="edge")
        fused += arr[:length] * (weight / total_w)

    return fused


def estimate_cell_for_ap(cell, ap_mac, bundles, measured):
    """Estimate one AP fingerprint at a cell using all reference points for that AP."""
    key = (cell, ap_mac)
    if key in measured:
        row = measured[key]
        return row["rssi"], row["rtt"]

    ap_bundles = bundles.get(ap_mac, {})
    if not ap_bundles:
        return np.array([]), np.array([])

    rssi_candidates = []
    rtt_candidates = []

    for bundle in ap_bundles.values():
        est_rssi, est_rtt = estimate_from_reference(cell, bundle)
        weight = reference_weight(cell, bundle["center"]["loc"])
        rssi_candidates.append((est_rssi, weight))
        rtt_candidates.append((est_rtt, weight))

    return (
        fuse_weighted_series(rssi_candidates),
        fuse_weighted_series(rtt_candidates),
    )


def build_gradient_map(input_csv=INPUT_CSV, output_csv=OUTPUT_CSV):
    if not Path(input_csv).exists():
        raise FileNotFoundError(f"Missing fingerprint data: {input_csv}")

    num_x, num_y, cells = build_room_grid()
    measured, by_rf_ap = load_fingerprints(input_csv)
    ap_macs = collect_ap_macs(by_rf_ap)
    bundles = get_reference_bundles(by_rf_ap)

    print(f"Room grid: {num_x} x {num_y} = {len(cells)} cells")
    print(f"APs: {len(ap_macs)}, reference bundles: {sum(len(v) for v in bundles.values())}")

    rows = []
    for cell in cells:
        for ap_mac in ap_macs:
            est_rssi, est_rtt = estimate_cell_for_ap(cell, ap_mac, bundles, measured)
            if est_rssi.size == 0:
                continue
            rows.append(
                {
                    "loc": str(cell),
                    "ap_mac": ap_mac,
                    "rssi": list(np.round(est_rssi, 4)),
                    "rtt": list(np.round(est_rtt, 4)),
                }
            )

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["loc", "ap_mac", "rssi", "rtt"]).to_csv(
        output_csv, index=False
    )
    print(f"Gradient map saved to {output_csv} ({len(rows)} rows)")
    return rows


if __name__ == "__main__":
    build_gradient_map()
