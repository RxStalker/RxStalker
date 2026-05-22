"""
Extract and denoise RSSI using Wi-Fi sniffing (pcap) and ESP32 CSI data.

Sniffed pcap RSSI optionally calibrates CSI-derived power estimates.
Multipath mitigation uses robust CSI subcarrier filtering + fusion + Hampel.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scapy.all import rdpcap
from scapy.layers.dot11 import Dot11, IP, RadioTap

DATA_PROCESSING_DIR = Path(__file__).resolve().parent
PROJECT_DIR = DATA_PROCESSING_DIR.parent
DATASET_DIR = PROJECT_DIR / "dataset"

for path in (PROJECT_DIR, DATA_PROCESSING_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from csiparser import ESP32
from utils.rssi_denoising import denoise_rssi_with_csi


def save_values_to_file(values, filename):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for value in values:
            f.write(f"{value}\n")


def extract_rssi_from_radiotap(pkt):
    try:
        return float(-(256 - pkt.notdecoded[-4]))
    except (IndexError, TypeError):
        return None


def extract_sniff_rssi(pcap_file, target_ip):
    """Extract RSSI values from packets destined to target_ip."""
    packets = rdpcap(str(pcap_file))
    records = []
    rssi_values = []

    for pkt in packets:
        if not (pkt.haslayer(RadioTap) and pkt.haslayer(Dot11) and pkt.haslayer(IP)):
            continue

        ip_layer = pkt[IP]
        if ip_layer.dst != target_ip:
            continue

        rssi = extract_rssi_from_radiotap(pkt)
        if rssi is None:
            continue

        src_mac = pkt[Dot11].addr2
        records.append([src_mac, target_ip, rssi])
        rssi_values.append(rssi)

    return records, np.asarray(rssi_values, dtype=float)


def load_esp32_amplitudes(csi_csv):
    """Parse ESP32 CSI CSV and return per-packet amplitudes and header RSSI."""
    parser = (
        ESP32(str(csi_csv))
        .get_csi()
        .remove_null_subcarriers()
        .get_amplitude_from_csi()
        .get_RSSI()
    )
    return np.asarray(parser.rssi_data, dtype=float), list(parser.amplitude)


def process_esp32_rssi(
    csi_csv,
    pcap_file=None,
    target_ip=None,
    output_dir=None,
    window=31,
    n_sigmas=3.0,
):
    """
    Denoise ESP32 RSSI using CSI (RCAR) and optional pcap sniff calibration.
    """
    output_dir = Path(output_dir or DATASET_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    rssi_raw, amplitudes = load_esp32_amplitudes(csi_csv)

    sniff_rssi = None
    if pcap_file and target_ip:
        records, sniff_rssi = extract_sniff_rssi(pcap_file, target_ip)
        save_values_to_file(records, output_dir / "wireshark_capture_rssi.txt")

    result = denoise_rssi_with_csi(
        rssi_raw,
        amplitudes,
        sniff_rssi_values=sniff_rssi,
        window=window,
        n_sigmas=n_sigmas,
    )

    denoised_path = output_dir / "esp32_rssi_denoised.txt"
    save_values_to_file(result["rssi_denoised"], denoised_path)

    detail_path = output_dir / "esp32_rssi_detail.csv"
    pd.DataFrame(
        {
            "rssi_raw": result["rssi_raw"],
            "rssi_csi": result["rssi_csi"],
            "rssi_fused": result["rssi_fused"],
            "rssi_denoised": result["rssi_denoised"],
        }
    ).to_csv(detail_path, index=False)

    print(f"Packets: {len(rssi_raw)}")
    print(f"CSI calibration bias: {result['calibration_bias']:.4f} dB")
    print(f"Saved denoised RSSI: {denoised_path}")
    print(f"Saved detail CSV: {detail_path}")

    return result


def extract_rssi_and_dst_ip(pcap_file, target_ip, output_file=None):
    """Legacy helper: extract sniffed RSSI for a target IP."""
    output_file = output_file or DATASET_DIR / "wireshark_capture_rssi.txt"
    records, _ = extract_sniff_rssi(pcap_file, target_ip)
    save_values_to_file(records, output_file)
    for src_mac, ip, rssi in records:
        print(f"Mac: {src_mac} | ip: {ip} | RSSI: {rssi} dBm")
    return records


def parse_args():
    parser = argparse.ArgumentParser(
        description="RSSI extraction and CSI-assisted multipath denoising"
    )
    parser.add_argument(
        "--csi-csv",
        type=str,
        help="ESP32 CSI CSV file (enables CSI-assisted denoising)",
    )
    parser.add_argument(
        "--pcap",
        type=str,
        help="Sniffed pcap/pcapng for optional RSSI calibration",
    )
    parser.add_argument(
        "--target-ip",
        type=str,
        help="Target IP for pcap RSSI extraction",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DATASET_DIR),
        help="Output directory (default: dataset/)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=31,
        help="Hampel filter window size",
    )
    parser.add_argument(
        "--n-sigmas",
        type=float,
        default=3.0,
        help="Hampel filter sigma threshold",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.csi_csv:
        if args.pcap and not args.target_ip:
            print("Warning: --pcap given without --target-ip; skipping sniff calibration")
        process_esp32_rssi(
            args.csi_csv,
            pcap_file=args.pcap,
            target_ip=args.target_ip,
            output_dir=args.output_dir,
            window=args.window,
            n_sigmas=args.n_sigmas,
        )
    elif args.pcap and args.target_ip:
        extract_rssi_and_dst_ip(args.pcap, args.target_ip)
    else:
        print(
            "Usage examples:\n"
            "  python RSSI_parser.py --csi-csv ../dataset/1.csv\n"
            "  python RSSI_parser.py --csi-csv ../dataset/1.csv "
            "--pcap ../dataset/publix1.pcapng --target-ip 100.100.33.165\n"
            "  python RSSI_parser.py --pcap ../dataset/publix1.pcapng "
            "--target-ip 100.100.33.165"
        )


if __name__ == "__main__":
    main()
