
import argparse

from RxStalker.deviceScan import activate_device_scanning
from RxStalker.deviceSniffing import activate_device_sniffing
from RxStalker.gradient_map import build_gradient_map
from RxStalker.RxStalker import activate_rxstalker_attack


def parse_args():
    parser = argparse.ArgumentParser(description="RxStalker tracking")

    parser.add_argument(
        "-g",
        action="store_true",
        help="Wi-Fi fingerprinting gradient map building",
    )

    parser.add_argument(
        "-o",
        type=int,
        choices=[1, 2],
        help="Feature option: 1=Active device sniffing, 2=Active device scan",
    )

    parser.add_argument(
        "-a",
        action="store_true",
        help="RxStalker attack (real-time tracking)",
    )
    parser.add_argument(
        "-ip",
        type=str,
        default="",
        help="Target IP address",
    )
    parser.add_argument(
        "-t",
        type=int,
        default=None,
        help="Capture timeout in seconds (default: run until Ctrl+C)",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=4,
        help="K neighbors for adaptive weighted KNN",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    sub_network = "100.100.33.1/22"

    if args.g:
        build_gradient_map()

    if args.o == 1:
        activate_device_sniffing(sub_network)
    elif args.o == 2:
        activate_device_scanning(sub_network)

    if args.a and not args.ip:
        print("Target IP address is required for RxStalker attack (-ip)")
    elif args.a and args.ip:
        timeout = args.t if args.t and args.t > 0 else None
        activate_rxstalker_attack(
            args.ip,
            timeout=timeout,
            k=args.k,
        )
