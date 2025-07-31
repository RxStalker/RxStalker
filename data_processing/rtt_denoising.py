import re
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt

rtt_file = "../dataset/TTL_ip18.txt"

def save_to_file(data, filename="../dataset/example.txt"):
    with open(filename, "w") as f:
        for value in data:
            f.write(f"{value}\n")

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


def plot_fig1(raw_rtt, ave_rtt):
    fig = plt.figure(figsize=(10, 7))

    # Create two axes with specific height ratios (1:1)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.05)
    ax1 = fig.add_subplot(gs[0])  # Upper subplot (100-1000)
    ax2 = fig.add_subplot(gs[1])  # Lower subplot (0-100)

    # Plot data on both axes
    for ax in [ax1, ax2]:
        ax.plot(raw_rtt, label='Original RTTs', color='blue', alpha=0.9)
        ax.plot(ave_rtt, label='Average RTTs', color='red', alpha=0.9)

    # Set different scales for the two axes
    ax1.set_ylim(100, 1100)  # Upper plot range
    ax2.set_ylim(0, 100)  # Lower plot range

    # Hide the bottom tick labels of the upper plot
    ax1.set_xticklabels([])

    # Add legend only to the upper plot
    ax1.legend(fontsize=18, loc="center right")

    # Set title on the top plot
    ax1.set_title('RTT: Original vs Denoised with λ = 0.5, 1', fontsize=24, pad=20)

    # Set labels
    ax2.set_xlabel('Time Sequence(Index)', fontsize=24)
    fig.text(0.04, 0.5, 'RTT (ms)', va='center', rotation='vertical', fontsize=24)

    # Set tick font sizes
    for ax in [ax1, ax2]:
        ax.tick_params(axis='both', which='major', labelsize=18)

    # plt.show()
    plt.savefig('../images/rtt_denoising_ave.png', dpi=1200, transparent=False)
    plt.close()
    plt.grid(False)
    plt.tight_layout()


def plot_fig2(raw_rtt, denoised_rtt5, denoised_rtt1):
    fig = plt.figure(figsize=(10, 7))

    # Create two axes with specific height ratios (1:1)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.05)
    ax1 = fig.add_subplot(gs[0])  # Upper subplot (100-1000)
    ax2 = fig.add_subplot(gs[1])  # Lower subplot (0-100)

    # Plot data on both axes
    for ax in [ax1, ax2]:
        ax.plot(raw_rtt, label='Original RTTs', color='blue', alpha=0.9)
        ax.plot(denoised_rtt1, label='Denoised RTT with λ = 1', color='red', alpha=0.9)
        ax.plot(denoised_rtt5, label='Denoised RTT with λ = 0.5', color='green', alpha=0.9)

    # Set different scales for the two axes
    ax1.set_ylim(100, 1100)  # Upper plot range
    ax2.set_ylim(0, 100)  # Lower plot range

    # Hide the bottom tick labels of the upper plot
    ax1.set_xticklabels([])

    # Add legend only to the upper plot
    ax1.legend(fontsize=18, loc="center right")

    # Set title on the top plot
    ax1.set_title('RTT: Original vs Denoised with λ = 0.5, 1', fontsize=24, pad=20)

    # Set labels
    ax2.set_xlabel('Time Sequence(Index)', fontsize=24)
    fig.text(0.04, 0.5, 'RTT (ms)', va='center', rotation='vertical', fontsize=24)

    # Set tick font sizes
    for ax in [ax1, ax2]:
        ax.tick_params(axis='both', which='major', labelsize=18)

    # plt.show()
    plt.savefig('../images/rtt_denoising_lambda.png', dpi=1200, transparent=False)
    plt.close()
    plt.grid(False)
    plt.tight_layout()


def plot_results(raw_rtt, ave_rtt, denoised_rtt5, denoised_rtt1, denoised_rtt10):
    if plt is None:
        print("Plotting disabled due to matplotlib import error")
        return

    plot_fig1(raw_rtt, ave_rtt)
    plot_fig2(raw_rtt, denoised_rtt5, denoised_rtt1)


def main():
    # Read and process RTT data
    rtt = read_rtt_data(rtt_file)[200:]
    rtt = rtt[:21000]
    raw_rtt = rtt[100:20100]

    lambda_reg5 = 0.5  # Regularization parameter for smoothness
    lambda_reg1 = 1
    lambda_reg10 = 10

    # Calculate average and denoised RTT
    ave_rtt = average_rtt(rtt)

    save_to_file(ave_rtt, filename="../data_processing/TTL_ip18_ave_rtt.txt")

    # Perform denoising
    denoised_rtt5 = denoise_rtt_optimization(rtt[100:], lambda_reg5)
    denoised_rtt1 = denoise_rtt_optimization(rtt[100:], lambda_reg1)
    denoised_rtt10 = denoise_rtt_optimization(rtt[100:], lambda_reg10)

    save_to_file(denoised_rtt5, filename="../data_processing/TTL_ip18_denoised_rtt5.txt")
    save_to_file(denoised_rtt1, filename="../data_processing/TTL_ip18_denoised_rtt1.txt")
    save_to_file(denoised_rtt10, filename="../data_processing/TTL_ip18_denoised_rtt10.txt")

    # Plot results
    plot_results(raw_rtt, ave_rtt, denoised_rtt5[:20000], denoised_rtt1[:20000], denoised_rtt10[:20000])


if __name__ == "__main__":
    main()