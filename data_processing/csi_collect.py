import serial

# Initialize serial connection
ser = serial.Serial("/dev/ttyACM0", 115200, timeout=1)
if not ser.is_open:
    ser = serial.Serial("COM3", 115200, timeout=1)

output_file = '../dataset/ESP32_CSI_v11.txt'
with open(output_file, 'w') as f:
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').rstrip()
            f.write(line + '\n')