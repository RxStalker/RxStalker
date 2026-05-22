import ipaddress
import platform
from pathlib import Path

from scapy.all import conf, sniff
from scapy.layers.inet import IP

LINUX_INTERFACE = "wlp0s20f3"

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_DIR / "dataset"
SNIFFING_IPS_OUTPUT = DATASET_DIR / "activate_sniffing_devices.txt"


def save_capture(ips, filename):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ip in sorted(ips):
            f.write(f"{ip}\n")
    print(f"Saved {len(ips)} IP(s) to {path}")


def get_capture_interface():
    if platform.system() == "Linux":
        return LINUX_INTERFACE
    return conf.iface


def ip_in_subnet(ip, network):
    try:
        return ipaddress.ip_address(ip) in network
    except ValueError:
        return False


def activate_device_sniffing(sub_network, timeout=None):
    """
    Capture live network traffic with Scapy and return IP addresses
    observed within the given local subnetwork (e.g. '192.168.1.0/24').
    Linux uses NIC wlp0s20f3; Windows uses the default active interface.
    """
    network = ipaddress.ip_network(sub_network, strict=False)
    iface = get_capture_interface()
    sniffing_ips = set()

    def handle_packet(pkt):
        if not pkt.haslayer(IP):
            return

        ip_layer = pkt[IP]
        for ip in (ip_layer.src, ip_layer.dst):
            if ip and ip_in_subnet(ip, network):
                sniffing_ips.add(ip)

    print(f"Sniffing on {iface} for subnet {sub_network} (Ctrl+C to stop)")
    try:
        sniff(iface=iface, prn=handle_packet, store=False, timeout=timeout)
    except KeyboardInterrupt:
        pass

    save_capture(sniffing_ips, SNIFFING_IPS_OUTPUT)

    return sorted(sniffing_ips)
