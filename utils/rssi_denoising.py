"""
RSSI multipath noise mitigation for Wi-Fi localization.

Time-domain: Hampel filter + moving-average smoothing.

CSI-assisted (RCAR): robust subcarrier filtering, CSI power aggregation
(10*log10(sum |H_k|^2)), multipath-aware fusion with ESP32 header RSSI,
optional sniffing-based calibration, then temporal Hampel smoothing.
Inspired by FapFi subcarrier outlier filtering and RSSI-CSI power aggregation.
"""

import numpy as np

DEFAULT_MP_REF = 0.35
DEFAULT_SUBCARRIER_N_SIGMAS = 3.0


def hampel_filter(series, window=31, n_sigmas=3.0):
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


def denoise_rssi_multipath(rssi_samples, window=31, n_sigmas=3.0):
    """
    Mitigate multipath noise in captured RSSI samples.
    """
    x = np.asarray(rssi_samples, dtype=float)
    if x.size == 0:
        return x

    x = hampel_filter(x, window=window, n_sigmas=n_sigmas)
    if len(x) >= 5:
        kernel = np.ones(5) / 5
        x = np.convolve(x, kernel, mode="same")
    return x


def filter_subcarrier_outliers(amplitudes, n_sigmas=DEFAULT_SUBCARRIER_N_SIGMAS):
    """Remove outlier subcarriers using median + MAD (FapFi-style)."""
    amps = np.asarray(amplitudes, dtype=float)
    if amps.size == 0:
        return amps

    median = np.median(amps)
    mad = np.median(np.abs(amps - median))
    if mad < 1e-9:
        return amps

    thresh = n_sigmas * 1.4826 * mad
    return amps[np.abs(amps - median) <= thresh]


def csi_rssi_proxy_dbm(amplitudes_clean, eps=1e-12):
    """RSSI-equivalent power from filtered CSI amplitudes: 10*log10(sum |H_k|^2)."""
    amps = np.asarray(amplitudes_clean, dtype=float)
    if amps.size == 0:
        return np.nan
    return float(10.0 * np.log10(np.sum(amps ** 2) + eps))


def fuse_rssi_with_csi(raw_rssi, csi_proxy, amplitudes_clean, mp_ref=DEFAULT_MP_REF):
    """
    Blend ESP32 header RSSI with CSI proxy; weight CSI more under strong multipath.
    """
    if not np.isfinite(csi_proxy):
        return float(raw_rssi)

    amps = np.asarray(amplitudes_clean, dtype=float)
    if amps.size == 0:
        return float(raw_rssi)

    mp_metric = float(np.std(amps) / (np.mean(amps) + 1e-9))
    weight = float(np.clip(mp_metric / mp_ref, 0.4, 0.85))
    return weight * float(csi_proxy) + (1.0 - weight) * float(raw_rssi)


def compute_sniff_calibration_bias(sniff_rssi_values, csi_proxy_series):
    """Align CSI proxy scale to sniffed RSSI using median offset."""
    sniff = np.asarray(sniff_rssi_values, dtype=float)
    proxy = np.asarray(csi_proxy_series, dtype=float)
    valid = np.isfinite(proxy)
    if sniff.size == 0 or not np.any(valid):
        return 0.0
    return float(np.median(sniff) - np.median(proxy[valid]))


def compute_csi_proxy_series(
    amplitudes_per_packet,
    subcarrier_n_sigmas=DEFAULT_SUBCARRIER_N_SIGMAS,
):
    """Per-packet CSI power proxy before scale calibration."""
    proxies = []
    for amps in amplitudes_per_packet:
        amps_clean = filter_subcarrier_outliers(amps, n_sigmas=subcarrier_n_sigmas)
        proxies.append(csi_rssi_proxy_dbm(amps_clean))
    return np.asarray(proxies, dtype=float)


def fuse_rssi_series_with_csi(
    rssi_raw,
    amplitudes_per_packet,
    calibration_bias=0.0,
    mp_ref=DEFAULT_MP_REF,
    subcarrier_n_sigmas=DEFAULT_SUBCARRIER_N_SIGMAS,
):
    """
    Per-packet CSI-assisted fusion. Returns (rssi_csi_calibrated, rssi_fused) arrays.
    calibration_bias shifts CSI proxy into the same scale as ESP32/sniff RSSI (dBm).
    """
    raw = np.asarray(rssi_raw, dtype=float)
    n = len(raw)
    csi_proxy = np.full(n, np.nan)
    csi_calibrated = np.full(n, np.nan)
    fused = np.copy(raw)

    for i in range(n):
        amps = amplitudes_per_packet[i] if i < len(amplitudes_per_packet) else np.array([])
        amps_clean = filter_subcarrier_outliers(amps, n_sigmas=subcarrier_n_sigmas)
        proxy = csi_rssi_proxy_dbm(amps_clean)
        csi_proxy[i] = proxy
        if np.isfinite(proxy):
            proxy_cal = proxy + calibration_bias
            csi_calibrated[i] = proxy_cal
            fused[i] = fuse_rssi_with_csi(raw[i], proxy_cal, amps_clean, mp_ref=mp_ref)

    return csi_proxy, csi_calibrated, fused


def denoise_rssi_with_csi(
    rssi_raw,
    amplitudes_per_packet,
    sniff_rssi_values=None,
    window=31,
    n_sigmas=3.0,
    mp_ref=DEFAULT_MP_REF,
    subcarrier_n_sigmas=DEFAULT_SUBCARRIER_N_SIGMAS,
):
    """
    Full RCAR pipeline: CSI fusion, optional sniff calibration, temporal Hampel.
    """
    raw = np.asarray(rssi_raw, dtype=float)
    if raw.size == 0:
        return {
            "rssi_raw": raw,
            "rssi_csi": raw,
            "rssi_fused": raw,
            "rssi_denoised": raw,
            "sniff_bias": 0.0,
        }

    csi_proxy = compute_csi_proxy_series(
        amplitudes_per_packet, subcarrier_n_sigmas=subcarrier_n_sigmas
    )

    if sniff_rssi_values is not None and len(sniff_rssi_values) > 0:
        calibration_bias = compute_sniff_calibration_bias(sniff_rssi_values, csi_proxy)
    else:
        valid = np.isfinite(csi_proxy)
        calibration_bias = (
            float(np.median(raw) - np.median(csi_proxy[valid])) if np.any(valid) else 0.0
        )

    _, csi_calibrated, fused = fuse_rssi_series_with_csi(
        raw,
        amplitudes_per_packet,
        calibration_bias=calibration_bias,
        mp_ref=mp_ref,
        subcarrier_n_sigmas=subcarrier_n_sigmas,
    )

    denoised = denoise_rssi_multipath(fused, window=window, n_sigmas=n_sigmas)

    return {
        "rssi_raw": raw,
        "rssi_csi": csi_calibrated,
        "rssi_fused": fused,
        "rssi_denoised": denoised,
        "calibration_bias": calibration_bias,
    }
