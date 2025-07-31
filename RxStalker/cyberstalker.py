import pandas as pd
import numpy as np
import ast
import math
from collections import defaultdict
from sklearn.neighbors import KNeighborsClassifier
from filterpy.kalman import KalmanFilter

# Parameters
DELTA = 0.1  # tuning parameter for entropy weight

# Load the text file using pandas
df = pd.read_csv("../dataset/dgradient_map.txt", converters={"rtt": ast.literal_eval})
def load_wfp_data(filepath):
    df = pd.read_csv(filepath, converters={"rtt": ast.literal_eval, "loc": ast.literal_eval})
    data_by_loc = defaultdict(list)
    for _, row in df.iterrows():
        feature = [row['rssi']] + row['rtt']
        data_by_loc[row['loc']].append((feature, row['rf']))
    return data_by_loc

def load_tfp_data(filepath):
    df = pd.read_csv(filepath, converters={"rtt": ast.literal_eval})
    tfp_features = [df.iloc[0]['rssi']] + df.iloc[0]['rtt']
    return tfp_features

def compute_entropy(prob_list):
    return -sum(p * math.log2(p) for p in prob_list if p > 0)

def compute_entropy_map(data_by_loc):
    entropy_map = {}
    for loc, data in data_by_loc.items():
        rfs = [rf for _, rf in data]
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
    distances = {}
    for loc_i, data_i in data_by_loc.items():
        for loc_j, data_j in data_by_loc.items():
            if loc_i == loc_j or (loc_i, loc_j) not in weights:
                continue
            for features, _ in data_j:
                dist = np.linalg.norm(np.array(features) - np.array(tfp_feature))
                weighted_dist = weights[(loc_i, loc_j)] * dist
                distances[(loc_i, loc_j)] = weighted_dist
    return distances

def classify_knn(data_by_loc, tfp_feature, k):
    X = []
    y = []
    for data in data_by_loc.values():
        for features, rf in data:
            X.append(features)
            y.append(rf)
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X, y)
    pred = clf.predict([tfp_feature])
    return pred[0]

def adaptive_weighted_knn(wfp_file, tfp_file, K):
    data_by_loc = load_wfp_data(wfp_file)
    tfp_feature = load_tfp_data(tfp_file)

    entropy_map = compute_entropy_map(data_by_loc)
    weights = compute_weights(entropy_map)
    distances = compute_weighted_distances(data_by_loc, tfp_feature, weights)
    pred_rf = classify_knn(data_by_loc, tfp_feature, k=K)

    return distances, (tfp_feature, pred_rf)

def create_kalman_filter(dim):
    kf = KalmanFilter(dim_x=dim, dim_z=dim)
    kf.x = np.zeros(dim)                # Initial state
    kf.F = np.eye(dim)                  # State transition matrix
    kf.H = np.eye(dim)                  # Measurement function
    kf.P *= 1000.                       # Covariance matrix
    kf.R = np.eye(dim) * 5              # Measurement noise
    kf.Q = np.eye(dim) * 0.1            # Process noise
    return kf

def apply_kalman_filter(kf, measurement):
    kf.predict()
    kf.update(measurement)
    return kf.x.copy()

def gradient_map(current_tfp, wfp_data):
    # Placeholder: simply return location of closest WFP sample
    min_dist = float('inf')
    best_loc = None
    for loc, data in wfp_data.items():
        for features, _ in data:
            dist = np.linalg.norm(np.array(features) - np.array(current_tfp))
            if dist < min_dist:
                min_dist = dist
                best_loc = loc
    return best_loc
def cyberstalker_tracking(wfp_file, tfp_file, n_neighbors):

    wfp_data = load_wfp_data(wfp_file)
    tfp_sequence = load_tfp_data(tfp_file)


    feature_dim = len(tfp_sequence[0])
    kf = create_kalman_filter(feature_dim)


    prev_tfp = None

    locations = []

    for t, tfp_t in enumerate(tfp_sequence):
        tfp_t = np.array(tfp_t)


        current_tfp = tfp_t

        # Step: AWKNN to find estimated WFP match
        distances, (tfp_feature, pred_rf) = adaptive_weighted_knn(wfp_file, tfp_file, n_neighbors)

        # Step: Compute movement vector
        if t == 0:
            base_fp = wfp_data[pred_rf][0][0]  # first ref point's features
            base_fp = np.array(base_fp)
            delta_tfp = (current_tfp - base_fp) / distances[(pred_rf, pred_rf)]
        else:
            delta_tfp = current_tfp - prev_tfp

        # Step: Kalman Filter smoothing
        filtered_delta = apply_kalman_filter(kf, delta_tfp)

        # Step: Estimate new location
        est_input = current_tfp + filtered_delta
        location = gradient_map(est_input, wfp_data)
        locations.append(location)

        prev_tfp = current_tfp

    return locations

if __name__ == '__main__':
    wfp_file = "../dataset/gradient_map.txt"
    tfp_file = "../dataset/wifi_fingerprinting_ip_192.168.130.13.txt"
    cyberstalker_tracking(wfp_file, tfp_file, 4)

