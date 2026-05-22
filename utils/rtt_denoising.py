import re
import shutil
import subprocess
from statistics import mean

import numpy as np

RTT_MS_PATTERN = re.compile(r"rtt=([\d.]+)\s*ms", re.IGNORECASE)


def parse_rtt_from_hping_output(text):
    """Extract RTT values (ms) from hping/hping3 stdout or stderr."""
    if not text:
        return []
    return [float(value) for value in RTT_MS_PATTERN.findall(text)]


def find_hping_executable():
    """Return path to hping3 or hping, or None if not installed."""
    for name in ("hping3", "hping"):
        path = shutil.which(name)
        if path:
            return path
    return None


def capture_rtt_hping(target_ip, count=3, timeout=3.0):
    """
    Measure RTT to target_ip using hping ICMP mode (-1).
    Requires hping3/hping on PATH; often needs root/CAP_NET_RAW on Linux.
    """
    hping = find_hping_executable()
    if hping is None:
        return []

    cmd = [hping, "-1", "-c", str(int(count)), "-n", str(target_ip)]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=max(timeout, 1.0) + int(count) * 0.5,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return []

    output = (result.stdout or "") + (result.stderr or "")
    return parse_rtt_from_hping_output(output)


def read_rtt_data(input_file):
    rtt_values = []
    with open(input_file, "r", encoding="utf-8") as file:
        for line in file:
            rtt_values.extend(parse_rtt_from_hping_output(line))
    return rtt_values


def average_rtt(rtt_values):
    rtt_data = []
    for i in range(100, 20000 + 100):
        rtt_data.append(mean(rtt_values[:i]))
    return rtt_data


def denoise_rtt_optimization(raw_rtt, lambda_reg=1):
    rtts = np.array(raw_rtt)
    mean = np.mean(rtts)
    std = np.std(rtts)
    filtered = rtts[rtts <= mean + lambda_reg * std]
    return filtered
