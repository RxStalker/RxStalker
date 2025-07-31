import numpy as np
from sklearn.linear_model import LinearRegression
from numpy.linalg import norm
from math import atan2, degrees

def fit_line(x, y):
    # Fit line y = ax + b
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    slope = model.coef_[0]
    intercept = model.intercept_
    return slope, intercept

def calculate_csi_gradient(csi1, csi2):
    n = len(csi1)
    x = np.arange(n)

    # Fit lines to both CSI data
    slope1, intercept1 = fit_line(x, csi1)
    slope2, intercept2 = fit_line(x, csi2)

    # Gradient as difference in slope (angle between lines)
    # angle1 = atan2(slope1, 1)
    # angle2 = atan2(slope2, 1)
    #angle_diff_deg = degrees(abs(angle1 - angle2))

    # Gradient as L2 norm between the direction vectors of the two lines
    dir1 = np.array([1, slope1])
    dir2 = np.array([1, slope2])
    l2_distance = norm(dir1 - dir2)

    delta_csi = l2_distance

    return delta_csi

    # return {
    #     "slope1": slope1,
    #     "slope2": slope2,
    #     "angle_diff_degrees": angle_diff_deg,
    #     "direction_l2_distance": l2_distance
    # }

def calculate_distance_gradient(star_pixel, mid_pixel, end_pixel, ratio):
    distance_gradient = ((end_pixel+mid_pixel)/2 - (star_pixel+mid_pixel)/2) * ratio
    return distance_gradient

def calculate_rssi_gradient(rssi1, rssi2):
    n = len(rssi1)
    x = np.arange(n)

    slope1, intercept1 = fit_line(x, rssi1)
    slope2, intercept2 = fit_line(x, rssi2)

    dir1 = np.array([1, slope1])
    dir2 = np.array([1, slope2])
    l2_distance = norm(dir1 - dir2)

    delta_rssi = l2_distance

    return delta_rssi

def calculate_rtt_gradient(rtt1, rtt2):
    n = len(rtt1)
    x = np.arange(n)

    slope1, intercept1 = fit_line(x, rtt1)
    slope2, intercept2 = fit_line(x, rtt2)

    dir1 = np.array([1, slope1])
    dir2 = np.array([1, slope2])
    l2_distance = norm(dir1 - dir2)

    delta_rtt = l2_distance

    return delta_rtt


def estimate_rssi_from_csi(amplitudes, phases):
    amplitudes = np.array(amplitudes)
    phases = np.array(phases)

    # Reconstruct complex CSI: A * e^(jφ)
    complex_csi = amplitudes * np.exp(1j * phases)

    # Compute total received power (sum of |CSI|^2 = A^2)
    power_linear = np.sum(np.abs(complex_csi) ** 2)

    # Convert to dBm
    rssi_dbm = 10 * np.log10(power_linear)

    return rssi_dbm
