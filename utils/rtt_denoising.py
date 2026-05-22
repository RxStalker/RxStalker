import re
from statistics import mean
import numpy as np

def read_rtt_data(input_file):
    # Regex pattern to extract rtt values
    pattern = r'rtt=([\d.]+) ms'
    rtt_values = []

    with open(input_file, 'r') as file:
        for line in file:
            matches = re.findall(pattern, line)
            for match in matches:
                rtt_values.append(float(match))
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