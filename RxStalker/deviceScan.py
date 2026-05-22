import ipaddress
import platform
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_DIR / "dataset"
SCANNING_IPS_OUTPUT = DATASET_DIR / "activate_scanning_devices.txt"
MAX_WORKERS = 50


def save_scan_results(ips, filename):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ip in ips:
            f.write(f"{ip}\n")
    print(f"Saved {len(ips)} IP(s) to {path}")


def ping_host(ip, timeout):
    """Return True if the host responds to a single ping."""
    if platform.system() == "Windows":
        timeout_ms = max(1, int(timeout * 1000))
        cmd = ["ping", "-n", "1", "-w", str(timeout_ms), ip]
    else:
        wait_sec = max(1, int(timeout))
        cmd = ["ping", "-c", "1", "-W", str(wait_sec), ip]

    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0,
    )
    return result.returncode == 0


def ping_scan(sub_network, timeout=1):
    network = ipaddress.ip_network(sub_network, strict=False)
    hosts = [str(host) for host in network.hosts()]
    if not hosts:
        return set()

    alive_ips = set()
    workers = min(MAX_WORKERS, len(hosts))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(ping_host, host, timeout): host for host in hosts}
        for future in as_completed(futures):
            host = futures[future]
            if future.result():
                alive_ips.add(host)
                print(f"Found device: {host}")

    return alive_ips


def activate_device_scanning(sub_network, timeout=1):
    """
    Simple host discovery: ping every address in sub_network in parallel.
    """
    print(f"Scanning {sub_network} ({ipaddress.ip_network(sub_network, strict=False).num_addresses - 2} hosts)...")

    scanning_ips = sorted(ping_scan(sub_network, timeout))
    save_scan_results(scanning_ips, SCANNING_IPS_OUTPUT)

    return scanning_ips
