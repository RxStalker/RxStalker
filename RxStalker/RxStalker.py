"""
RxStalker: gradient-map localization with real-time Wi-Fi fingerprinting and tracking.
"""

import ast
import math
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter
from scapy.all import AsyncSniffer, conf
from scapy.layers.dot11 import Dot11, RadioTap
from scapy.layers.inet import ICMP, IP, TCP
from sklearn.neighbors import KNeighborsClassifier

from utils import const
from utils.rssi_denoising import denoise_rssi_multipath
from utils.rtt_denoising import denoise_rtt_optimization

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

DATASET_DIR = PROJECT_DIR / "dataset"
GRADIENT_MAP_CSV = DATASET_DIR / "gradient_map.csv"
LINUX_INTERFACE = "wlp0s20f3"
DELTA = 0.1
DEFAULT_K_NEIGHBORS = 4
FEATURE_OPTION = 3


# ---------------------------------------------------------------------------
# Gradient map
# ---------------------------------------------------------------------------


def get_capture_interface():
    if platform.system() == "Linux":
        return LINUX_INTERFACE
    return conf.iface


def load_gradient_map(filepath=None):
    """Load gradient_map.csv; returns gradient_data and data_by_loc."""
    csv_path = Path(filepath) if filepath else GRADIENT_MAP_CSV
    if not csv_path.exists():
        raise FileNotFoundError(f"Gradient map not found: {csv_path}")

    df = pd.read_csv(
        csv_path,
        converters={
            "loc": ast.literal_eval,
            "rssi": ast.literal_eval,
            "rtt": ast.literal_eval,
        },
    )

    gradient_data = {}
    data_by_loc = defaultdict(list)

    for _, row in df.iterrows():
        loc = tuple(row["loc"]) if isinstance(row["loc"], list) else row["loc"]
        ap_mac = str(row["ap_mac"]).strip()
        entry = {
            "loc": loc,
            "ap_mac": ap_mac,
            "rssi": row["rssi"],
            "rtt": row["rtt"],
            "rf_id": row.get("rf_id", ""),
        }
        gradient_data[(loc, ap_mac)] = entry
        data_by_loc[loc].append(entry)

    print(f"Loaded gradient map: {len(df)} rows, {len(data_by_loc)} cells")
    return gradient_data, dict(data_by_loc)


def collect_ap_macs(gradient_data):
    return sorted({mac for (_, mac) in gradient_data.keys()})


# ---------------------------------------------------------------------------
# Feature / WFP helpers
# ---------------------------------------------------------------------------


def build_feature(rssi_val, rtt, option=FEATURE_OPTION):
    rtt_list = np.asarray(rtt, dtype=float).tolist()
    if option == 1:
        return [float(rssi_val)]
    if option == 2:
        return rtt_list
    return [float(rssi_val)] + rtt_list


def fingerprint_to_feature(fingerprint, option=FEATURE_OPTION):
    rssi = fingerprint.get("rssi", [])
    rtt = fingerprint.get("rtt", [])
    rssi_val = float(np.median(rssi)) if len(rssi) else 0.0
    return build_feature(rssi_val, rtt, option)


def wfp_data_for_ap(gradient_data, ap_mac, option=FEATURE_OPTION):
    """Build location-indexed WFP database for one AP MAC."""
    data_by_loc = defaultdict(list)
    for (loc, mac), entry in gradient_data.items():
        if mac != ap_mac:
            continue
        rssi = entry["rssi"]
        rssi_val = float(np.median(rssi)) if isinstance(rssi, (list, tuple)) else float(rssi)
        feature = build_feature(rssi_val, entry["rtt"], option)
        data_by_loc[loc].append((feature, entry.get("rf_id", "")))
    return dict(data_by_loc)


def loc_for_rf(wfp_data, rf_label):
    for loc, items in wfp_data.items():
        for _, rf in items:
            if rf == rf_label:
                return loc
    return None


# ---------------------------------------------------------------------------
# Adaptive weighted KNN (entropy)
# ---------------------------------------------------------------------------


def compute_entropy(prob_list):
    return -sum(p * math.log2(p) for p in prob_list if p > 0)


def compute_entropy_map(data_by_loc):
    entropy_map = {}
    for loc, data in data_by_loc.items():
        rfs = [rf for _, rf in data]
        if not rfs:
            continue
        rf_counts = pd.Series(rfs).value_counts(normalize=True)
        entropy_map[loc] = compute_entropy(rf_counts.values)
    return entropy_map


def compute_weights(entropy_map):
    weights = {}
    locs = list(entropy_map.keys())
    for i in range(len(locs)):
        for j in range(len(locs)):
            if i != j:
                loc_i, loc_j = locs[i], locs[j]
                entropy_diff = abs(entropy_map[loc_i] - entropy_map[loc_j])
                weights[(loc_i, loc_j)] = math.exp(-entropy_diff / DELTA)
    return weights


def compute_weighted_distances(data_by_loc, tfp_feature, weights):
    tfp = np.asarray(tfp_feature, dtype=float)
    distances = {}
    for loc_i in data_by_loc:
        for loc_j in data_by_loc:
            if loc_i == loc_j or (loc_i, loc_j) not in weights:
                continue
            for features, _ in data_by_loc[loc_j]:
                dist = float(np.linalg.norm(np.asarray(features, dtype=float) - tfp))
                distances[(loc_i, loc_j)] = weights[(loc_i, loc_j)] * dist
    return distances


def classify_knn(data_by_loc, tfp_feature, k):
    X, y = [], []
    for data in data_by_loc.values():
        for features, rf in data:
            X.append(features)
            y.append(rf)
    if not X:
        return None
    k = min(k, len(X))
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X, y)
    return clf.predict([tfp_feature])[0]


def adaptive_weighted_knn(wfp_data, tfp_feature, k=DEFAULT_K_NEIGHBORS):
    """
    Entropy-weighted KNN: find closest reference fingerprint and cell under this AP.
    """
    if not wfp_data:
        return None

    entropy_map = compute_entropy_map(wfp_data)
    weights = compute_weights(entropy_map)
    distances = compute_weighted_distances(wfp_data, tfp_feature, weights)
    pred_rf = classify_knn(wfp_data, tfp_feature, k)
    ref_loc = loc_for_rf(wfp_data, pred_rf) if pred_rf is not None else None

    if ref_loc is None:
        ref_loc = min(
            wfp_data.keys(),
            key=lambda loc: min(
                np.linalg.norm(np.asarray(f, dtype=float) - np.asarray(tfp_feature, dtype=float))
                for f, _ in wfp_data[loc]
            ),
        )

    return {
        "distances": distances,
        "tfp_feature": tfp_feature,
        "pred_rf": pred_rf,
        "ref_loc": ref_loc,
    }


def estimate_location_from_gradient(est_feature, wfp_data):
    """Match estimated fingerprint to the closest gradient-map cell."""
    est = np.asarray(est_feature, dtype=float)
    min_dist = float("inf")
    best_loc = None

    for loc, entries in wfp_data.items():
        for features, _ in entries:
            dist = float(np.linalg.norm(np.asarray(features, dtype=float) - est))
            if dist < min_dist:
                min_dist = dist
                best_loc = loc

    return best_loc


# ---------------------------------------------------------------------------
# Kalman filter tracking
# ---------------------------------------------------------------------------


def create_kalman_filter(dim):
    kf = KalmanFilter(dim_x=dim, dim_z=dim)
    kf.x = np.zeros(dim)
    kf.F = np.eye(dim)
    kf.H = np.eye(dim)
    kf.P *= 1000.0
    kf.R = np.eye(dim) * 5
    kf.Q = np.eye(dim) * 0.1
    return kf


def apply_kalman_filter(kf, measurement):
    kf.predict()
    kf.update(np.asarray(measurement, dtype=float))
    return kf.x.copy()


def track_fingerprint_step(fingerprint, gradient_data, ap_mac, tracker_state, k, option):
    """
    One localization step for a single AP:
    AWKNN -> movement delta -> Kalman -> gradient cell match.
    """
    wfp_data = wfp_data_for_ap(gradient_data, ap_mac, option)
    if not wfp_data:
        return None

    tfp_feature = fingerprint_to_feature(fingerprint, option)
    awknn = adaptive_weighted_knn(wfp_data, tfp_feature, k)
    if awknn is None:
        return None

    current_tfp = np.asarray(tfp_feature, dtype=float)
    ref_loc = awknn["ref_loc"]
    distances = awknn["distances"]

    if tracker_state["kf"] is None:
        tracker_state["kf"] = create_kalman_filter(len(current_tfp))

    if tracker_state["step"] == 0:
        base_fp = np.asarray(wfp_data[ref_loc][0][0], dtype=float)
        denom = distances.get((ref_loc, ref_loc), 1.0)
        if denom == 0 or math.isnan(denom):
            denom = 1.0
        delta_tfp = (current_tfp - base_fp) / denom
    else:
        prev_tfp = np.asarray(tracker_state["prev_tfp"], dtype=float)
        delta_tfp = current_tfp - prev_tfp

    filtered_delta = apply_kalman_filter(tracker_state["kf"], delta_tfp)
    est_feature = (current_tfp + filtered_delta).tolist()
    estimated_loc = estimate_location_from_gradient(est_feature, wfp_data)

    tracker_state["prev_tfp"] = current_tfp
    tracker_state["step"] += 1

    return {
        "ap_mac": ap_mac,
        "estimated_loc": estimated_loc,
        "ref_loc": ref_loc,
        "pred_rf": awknn["pred_rf"],
        "est_feature": est_feature,
    }


def fuse_ap_locations(ap_results):
    """Fuse per-AP cell estimates into one coordinate (component-wise median)."""
    locs = [r["estimated_loc"] for r in ap_results.values() if r and r.get("estimated_loc")]
    if not locs:
        return None
    xs = [loc[0] for loc in locs]
    ys = [loc[1] for loc in locs]
    return (int(round(float(np.median(xs)))), int(round(float(np.median(ys)))))


def cyberstalker_tracking(gradient_data, fingerprint_stream, k=DEFAULT_K_NEIGHBORS, option=FEATURE_OPTION):
    """
    Track a moving device over a stream of denoised real-time fingerprints.
    """
    ap_macs = collect_ap_macs(gradient_data)
    tracker_states = {ap: {"prev_tfp": None, "kf": None, "step": 0} for ap in ap_macs}
    path = []

    for fingerprint in fingerprint_stream:
        ap_results = {}
        for ap_mac in ap_macs:
            result = track_fingerprint_step(
                fingerprint, gradient_data, ap_mac, tracker_states[ap_mac], k, option
            )
            if result:
                ap_results[ap_mac] = result

        fused_loc = fuse_ap_locations(ap_results)
        path.append(
            {
                "fused_loc": fused_loc,
                "ap_results": ap_results,
                "fingerprint": fingerprint,
            }
        )

    return path


# ---------------------------------------------------------------------------
# Real-time capture + denoising
# ---------------------------------------------------------------------------


def extract_rssi(pkt):
    try:
        return float(-(256 - pkt.notdecoded[-4]))
    except (IndexError, TypeError):
        return None


def packet_involves_target(ip_layer, target_ip):
    return ip_layer.src == target_ip or ip_layer.dst == target_ip


def fingerprint_snapshot(target_ip, rssi_samples, rtt_samples):
    return {
        "target_ip": target_ip,
        "rssi": list(rssi_samples),
        "rtt": list(rtt_samples),
    }


def denoise_wifi_fingerprint(fingerprint):
    raw_rssi = list(fingerprint.get("rssi", []))
    raw_rtt = list(fingerprint.get("rtt", []))

    denoised_rssi = denoise_rssi_multipath(raw_rssi)
    if len(raw_rtt) == 0:
        denoised_rtt = np.array([], dtype=float)
    else:
        denoised_rtt = denoise_rtt_optimization(raw_rtt, lambda_reg=const.lambda_reg)

    return {
        "target_ip": fingerprint["target_ip"],
        "rssi": list(np.round(denoised_rssi, 4)),
        "rtt": list(np.round(denoised_rtt, 4)),
        "rssi_raw": raw_rssi,
        "rtt_raw": raw_rtt,
    }


def real_time_wifi_fingerprint(target_ip, timeout=None, iface=None, interval=1.0):
    """Capture and yield denoised Wi-Fi fingerprints persistently."""
    target_ip = str(target_ip).strip()
    iface = iface or get_capture_interface()

    rssi_samples = []
    rtt_samples = []
    pending_icmp = {}
    pending_tcp = {}

    def handle_packet(pkt):
        if not pkt.haslayer(IP):
            return

        ip_layer = pkt[IP]
        if not packet_involves_target(ip_layer, target_ip):
            return

        if pkt.haslayer(RadioTap) and pkt.haslayer(Dot11):
            rssi = extract_rssi(pkt)
            if rssi is not None:
                rssi_samples.append(rssi)

        if pkt.haslayer(ICMP):
            icmp = pkt[ICMP]
            if icmp.type == 8:
                key = (ip_layer.src, ip_layer.dst, int(icmp.id), int(icmp.seq))
                pending_icmp[key] = float(pkt.time)
            elif icmp.type == 0:
                key = (ip_layer.dst, ip_layer.src, int(icmp.id), int(icmp.seq))
                sent_time = pending_icmp.pop(key, None)
                if sent_time is not None:
                    rtt_samples.append((float(pkt.time) - sent_time) * 1000.0)

        if pkt.haslayer(TCP):
            tcp = pkt[TCP]
            if tcp.flags == 0x02 and ip_layer.dst == target_ip:
                key = (ip_layer.src, ip_layer.sport, ip_layer.dst, ip_layer.dport)
                pending_tcp[key] = float(pkt.time)
            elif tcp.flags == 0x12:
                key = (ip_layer.dst, ip_layer.dport, ip_layer.src, ip_layer.sport)
                sent_time = pending_tcp.pop(key, None)
                if sent_time is not None:
                    rtt_samples.append((float(pkt.time) - sent_time) * 1000.0)

    print(f"Capturing fingerprint for {target_ip} on {iface} (Ctrl+C to stop)")
    sniffer = AsyncSniffer(iface=iface, prn=handle_packet, store=False)
    sniffer.start()
    start_time = time.time()

    try:
        while True:
            raw = fingerprint_snapshot(target_ip, rssi_samples, rtt_samples)
            yield denoise_wifi_fingerprint(raw)

            if timeout is not None and (time.time() - start_time) >= timeout:
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    finally:
        sniffer.stop()

    yield denoise_wifi_fingerprint(
        fingerprint_snapshot(target_ip, rssi_samples, rtt_samples)
    )


# ---------------------------------------------------------------------------
# Attack entry point
# ---------------------------------------------------------------------------


def activate_rxstalker_attack(
    target_ip,
    timeout=None,
    interval=1.0,
    k=DEFAULT_K_NEIGHBORS,
    option=FEATURE_OPTION,
):
    """
    Full RxStalker pipeline:
    load gradient map -> capture/denoise fingerprints -> AWKNN per AP ->
    gradient localization -> Kalman-smoothed path.
    """
    gradient_data, _ = load_gradient_map()

    fingerprint_stream = real_time_wifi_fingerprint(
        target_ip, timeout=timeout, interval=interval
    )
    path = cyberstalker_tracking(
        gradient_data, fingerprint_stream, k=k, option=option
    )

    for i, step in enumerate(path):
        fused = step["fused_loc"]
        print(f"[step {i + 1}] estimated location: {fused}")
        for ap_mac, res in step["ap_results"].items():
            print(f"  AP {ap_mac}: cell {res['estimated_loc']}, ref_rf {res['pred_rf']}")

    return {
        "gradient_data": gradient_data,
        "path": path,
        "final_location": path[-1]["fused_loc"] if path else None,
    }
