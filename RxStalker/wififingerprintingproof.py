import numpy as np
import matplotlib.pyplot as plt


def read_rssi_data(file_name):
    rssi_value = []
    with open(file_name, 'r') as file:
        rssi_value.append(float(file.readlines()))
    rssi_value = np.array(rssi_value)
    return np.average(rssi_value)


def read_rtt_data(file_name):
    rtt_value = []
    with open(file_name, 'r') as file:
        rtt_value.append(float(file.readlines()))
    rtt_value = np.array(rtt_value)
    return rtt_value

def image_plot(RSSI, RTT):
    # Sample indices
    x = range(len(RSSI))

    # Create subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))

    # Plot RSSI
    axs[0, 0].plot(x, RSSI, marker='o', color='b', label="RSSI")
    axs[0, 0].set_ylabel("RSSI (dBm)", fontsize=18)
    axs[0, 0].set_xlabel("distance (x2m)", fontsize=18)
    axs[0, 0].set_title("RSSI Trend", fontsize=24)
    axs[0, 0].tick_params(axis='both', labelsize=16)
    axs[0, 0].grid(True)
    axs[0, 0].legend(fontsize=16)

    # # Plot CSI
    # axs[0, 1].plot(x, CSI, marker='o', color='g', label="CSI")
    # axs[0, 1].set_ylabel("CSI amplitude", fontsize=18)
    # axs[0, 1].set_xlabel("distance (x2m)", fontsize=18)
    # axs[0, 1].set_title("CSI amplitude Trend",  fontsize=24)
    # axs[0, 1].tick_params(axis='both', labelsize=16)
    # axs[0, 1].grid(True)
    # axs[0, 1].legend(fontsize=16)

    # # Plot phases
    # axs[1, 0].plot(x, phases, marker='o', color='r', label="Phase")
    # axs[1, 0].set_ylabel("Phase (radians)", fontsize=18)
    # axs[1, 0].set_xlabel("distance (x2m)", fontsize=18)
    # axs[1, 0].set_title("Phase Trend", fontsize=24)
    # axs[1, 0].tick_params(axis='both', labelsize=16)
    # axs[1, 0].grid(True)
    # axs[1, 0].legend(fontsize=16)

    # Plot RTT
    axs[1, 0].plot(x, RTT, marker='o', color='purple', label="RTT")
    axs[1, 0].set_xlabel("distance (x2m)", fontsize=18)
    axs[1, 0].set_ylabel("RTT (ms)", fontsize=18)
    axs[1, 0].set_title("RTT Trend", fontsize=24)
    axs[1, 0].tick_params(axis='both', labelsize=16)
    axs[1, 0].grid(True)
    axs[1, 0].legend(fontsize=16)


    # Adjust layout and show plot
    plt.tight_layout()
    #plt.show()
    plt.savefig("images/wififingerprintproof.png", dpi=1200)


if __name__ == "__main__":
    rssis = []
    amplitudes = []
    phases = []
    rtts = []
    for i in range(1, 6):
        amplitude_file_name = f"../dataset/amplitude_parser_{i}.txt"
        phase_file_name = f"../dataset/phase_parser_{i}.txt"
        rssi_file_name = f"../dataset/rssi_parser_{i}.txt"
        rtt_file_name = f"../dataset/denoised_rtt_{i}.txt"
        rssi = read_rssi_data(rssi_file_name)
        rtt = read_rtt_data(rtt_file_name)
        rssis.append(rssi)
        rtts.append(np.average(rtt))
    image_plot(rssis, rtts)
