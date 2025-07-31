from scapy.all import rdpcap
from scapy.layers.dot11 import Dot11, RadioTap, IP

def save_to_file(data, filename="../dataset/example.txt"):
    with open(filename, "w") as f:
        for value in data:
            f.write(f"{value}\n")

def extract_rssi_and_dst_ip(pcap_file, target_ip):
    packets = rdpcap(pcap_file)
    data_pair = []

    for pkt in packets:
        if pkt.haslayer(RadioTap) and pkt.haslayer(Dot11) and pkt.haslayer(IP):
            ip_layer = pkt[IP]
            if ip_layer.dst == target_ip:
                try:
                    # RSSI is often 4th byte from the end of the undecoded Radiotap header
                    rssi = -(256 - pkt.notdecoded[-4])
                    src_mac = pkt[Dot11].addr2
                    #dst_mac = pkt[Dot11].addr1
                    data_pair.append([src_mac, target_ip, rssi])
                    print(f"Mac: {src_mac} | ip: {target_ip} | RSSI: {rssi} dBm")
                except:
                    continue
    save_to_file(data_pair, "../dataset/wireshark_capture_rssi.txt")

# Run the function
extract_rssi_and_dst_ip("../dataset/publix1.pcapng", "74.125.6.166")