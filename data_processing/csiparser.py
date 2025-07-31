import numpy as np
import pandas as pd


class ESP32:
    """Parse ESP32 Wi-Fi Channel State Information (CSI) obtained using ESP32 CSI Toolkit by Hernandez and Bulut.
    ESP32 CSI Toolkit: https://stevenmhernandez.github.io/ESP32-CSI-Tool/
    csiparser: https://github.com/RikeshMMM/ESP32-CSI-Python-Parser
    """

    # View README.md for more information on null subcarriers
    NULL_SUBCARRIERS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 64, 65, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
                        128, 129, 130, 131, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260,
                        261, 262, 263, 264, 265, 266, 267, 382, 383]

    def __init__(self, csi_file):
        self.csi_file = csi_file
        self.__read_file()

    def __read_file(self):
        """Read RAW CSI file (.csv) using Pandas and return a Pandas dataframe
        """
        self.csi_df = pd.read_csv(self.csi_file)

    def seek_file(self):
        """Seek RAW CSI file
        """
        return self.csi_df

    def get_csi(self):
        raw_csi_data = self.csi_df['data'].copy()
        csi_data = np.array([np.fromstring(csi_datum.strip('[ ]'), dtype=int, sep=',') for csi_datum in raw_csi_data])
        self.csi_data = csi_data

        return self

    # NOTE: Currently does not provide support for all signal subcarrier types
    def remove_null_subcarriers(self):
        """Remove NULL subcarriers from CSI
        """

        # Non-HT Signals (20 Mhz) - non STBC
        if self.csi_data.shape[1] == 128:
            remove_null_subcarriers = self.NULL_SUBCARRIERS[:24]
        # HT Signals (40 Mhz) - non STBC
        elif self.csi_data.shape[1] == 384:
            remove_null_subcarriers = self.NULL_SUBCARRIERS
        else:
            return self

        csi_data_T = self.csi_data.T
        csi_data_T_clean = np.delete(csi_data_T, remove_null_subcarriers, 0)
        csi_data_clean = csi_data_T_clean.T
        self.csi_data = csi_data_clean

        return self

    def get_amplitude_from_csi(self):
        """Calculate the Amplitude (or Magnitude) from CSI
        Ref: https://farside.ph.utexas.edu/teaching/315/Waveshtml/node88.html
        """
        amplitude = np.array([np.sqrt(data[::2] ** 2 + data[1::2] ** 2) for data in self.csi_data])
        self.amplitude = amplitude
        return self

    def get_phase_from_csi(self):
        """Calculate the Amplitude (or Magnitude) from CSI
        Ref: https://farside.ph.utexas.edu/teaching/315/Waveshtml/node88.html
        """
        phase = np.array([np.arctan2(data[::2], data[1::2]) for data in self.csi_data])
        self.phase = phase
        return self

    def get_RSSI(self):
        rssi_data = np.array(self.csi_df['rssi'].copy(), type=int)
        self.rssi_data = rssi_data
        print(rssi_data)
        return self

    def save_to_file(self, filename="../dataset/example"):
        if "rssi" in filename:
            with open(filename+"_parser.txt", "w") as f:
                for value in self.rssi_data:
                    f.write(f"{value}\n")
        elif "amplitude" in filename:
            with open(filename+"_parser.txt", "w") as f:
                for value in self.amplitude:
                    f.write(f"{value}\n")
        elif "phase" in filename:
            with open(filename+"_parser.txt", "w") as f:
                for value in self.phase:
                    f.write(f"{value}\n")

if __name__ == '__main__':
    csi_data=ESP32("../dataset/1.csv").get_csi()
    csi_data.remove_null_subcarriers()
    csi_amplitude = csi_data.get_amplitude_from_csi()
    csi_data.save_to_file("../dataset/amplitude")
    print(np.array([np.average(amplitude) for amplitude in csi_amplitude.amplitude]))
