# RxStalker

Wi-Fi indoor localization and tracking using RSSI/RTT fingerprinting, a gradient-map reference model, and optional ESP32 CSI-assisted multipath mitigation.

## Requirements

- **Python 3.12+**
- Dependencies: `pip install -r requirements.txt`

## Enabling monitor mode on wireless interface

```
- Set network interface wlan0 into monitor mode using Aircrack-ng
# airmon-ng start wlan0
- Find nearby channels
# iw dev wlan0mon scan | grep DS
- Set wlan0mon to Wi-Fi channel 40
# iwconfig wlan0mon channel 40
- Stop monitor mode
# airmon-ng stop wlan0mon

```


## Project structure

```
RxStalker/
├── main.py                 # CLI entry point
├── requirements.txt
├── dataset/                # Input/output data (not shipped; see below)
├── RxStalker/
│   ├── RxStalker.py        # Real-time tracking and attack pipeline
│   ├── gradient_map.py     # Build gradient_map.csv from fingerprints
│   ├── deviceSniffing.py   # Active device discovery via sniffing
│   └── deviceScan.py       # Active device discovery via ping scan
├── data_processing/
│   ├── RSSI_parser.py      # Pcap + ESP32 CSI RSSI denoising
│   ├── csiparser.py        # ESP32 CSI CSV parser
│   └── csi_collect.py      # ESP32 CSI serial logger
└── utils/
    ├── const.py            # Room grid and RTT regularization
    ├── rssi_denoising.py   # Hampel + CSI-assisted (RCAR) denoising
    └── rtt_denoising.py    # RTT denoising for fingerprints
```

Expected files under `dataset/` (create the folder as needed):

| File | Description |
|------|-------------|
| `wifi_fingerprinting.csv` | Reference fingerprints for gradient-map building |
| `gradient_map.csv` | Generated gradient map (output of `-g`) |
| `*.pcap` / `*.pcapng` | Captured traffic for sniffed RSSI |
| `*.csv` | ESP32 CSI exports ([ESP32-CSI-Tool](https://stevenmhernandez.github.io/ESP32-CSI-Tool/)) |

## Usage

Run all commands from the project root unless noted.

### Main pipeline (`main.py`)

```bash
# Build gradient map from wifi_fingerprinting.csv → dataset/gradient_map.csv
python main.py -g

# Discover devices on subnet (sniffing)
python main.py -o 1

# Discover devices on subnet (ping scan)
python main.py -o 2

# Real-time tracking against gradient map
python main.py -a -ip <TARGET_IP>

# Optional: stop after N seconds, set KNN neighbors
python main.py -a -ip <TARGET_IP> -t 60 -k 4
```

| Flag | Description |
|------|-------------|
| `-g` | Build Wi-Fi fingerprint gradient map |
| `-o 1` | Active device sniffing |
| `-o 2` | Active device scanning |
| `-a` | Start RxStalker real-time tracking |
| `-ip` | Target device IP (required with `-a`) |
| `-t` | Capture timeout in seconds |
| `-k` | K neighbors for adaptive weighted KNN (default: 4) |

The default subnet in `main.py` is `100.100.33.1/22`; adjust in code if your deployment differs.

### Data processing

**ESP32 CSI collection** (`data_processing/csi_collect.py`):

- Reads serial from ESP32 (default `COM3` on Windows, `/dev/ttyACM0` on Linux).
- Writes raw lines to `dataset/ESP32_CSI_v11.txt`.

**CSI parsing** (`data_processing/csiparser.py`):

- Parses ESP32 CSI CSV (`data`, `rssi` columns).
- Extracts amplitude, phase, and RSSI; removes null subcarriers.

**RSSI extraction and denoising** (`data_processing/RSSI_parser.py`):

Mitigates multipath noise using CSI subcarrier filtering, CSI power aggregation, fusion with ESP32 RSSI, optional pcap calibration, and temporal Hampel smoothing.

```bash
cd data_processing

# ESP32 CSI only
python RSSI_parser.py --csi-csv ../dataset/1.csv

# CSI + Wireshark sniff calibration
python RSSI_parser.py --csi-csv ../dataset/1.csv \
  --pcap ../dataset/capture.pcapng --target-ip 100.100.33.165

# Sniffed RSSI only (pcap)
python RSSI_parser.py --pcap ../dataset/capture.pcapng --target-ip 100.100.33.165
```

Outputs (under `dataset/` by default):

- `esp32_rssi_denoised.txt` — denoised RSSI series (one value per line)
- `esp32_rssi_detail.csv` — `rssi_raw`, `rssi_csi`, `rssi_fused`, `rssi_denoised`
- `wireshark_capture_rssi.txt` — sniffed records when `--pcap` is used

### Utilities

- **`utils/rssi_denoising.py`** — Hampel filter and CSI-assisted RCAR pipeline; used by `RSSI_parser.py` and live tracking in `RxStalker.py`.
- **`utils/rtt_denoising.py`** — RTT smoothing for fingerprint RTT features.
- **`utils/const.py`** — Room dimensions, cell size, and `lambda_reg` for RTT denoising.

## Typical workflow

1. Collect reference fingerprints → `dataset/wifi_fingerprinting.csv`
2. `python main.py -g` → `dataset/gradient_map.csv`
3. (Optional) Collect ESP32 CSI / pcap and run `RSSI_parser.py` for cleaner RSSI
4. `python main.py -a -ip <target>` for real-time localization

## Configuration

Edit `utils/const.py` for room grid geometry and RTT regularization. On Linux, set the capture interface in `RxStalker/RxStalker.py` (`LINUX_INTERFACE`).
