import matplotlib.pyplot as plt
from PIL import Image
from scipy.special import gradient
from sympy.codegen.ast import float32

import wiFinGradient as wg
import math

import numpy as np

from scipy.interpolate import Rbf
import pandas as pd

# Load the image
image_path = "floor.png"
image=Image.open(image_path)

# building space information using meter as the measurement
# attacker can get this information through AI tools or distance measurement tools with a smart phone
building_length = 23.3
building_width = 7.8

# define the grid cell size
grid_cell_length = 1
grid_cell_width = 1


image_np = np.array(image)

# Create a figure and axes
fig, ax = plt.subplots()

# Display the image
ax.imshow(image_np)

height, width, _ = image_np.shape


csv_path = '../dataset/mapping.csv'

# Load data
#layout = imread(image_path)
data = pd.read_csv(csv_path)

# Beacon IDs to plot
beacon = data['beacon']

grid_width = width-10 # 10 is adjustable pixel from your input image
grid_height = height-20 # 20 is adjustable pixel from your input image

# Resolution
num_x = int(width / 4)
num_y = int(num_x / (width / height))
print(f"Resolution: {num_x} x {num_y}")

# Grid generation
x = np.linspace(0, grid_width, num_x)
y = np.linspace(0, grid_height, num_y)

gx, gy = np.meshgrid(x, y)
gx, gy = gx.flatten(), gy.flatten()

# get reference points information
rf1 = data[data['type'] == 'rf1']
rf2 = data[data['type'] == 'rf2']
rf3 = data[data['type'] == 'rf3']
rf4 = data[data['type'] == 'rf4']
ap = data[data['type'] == 'ap']
attacker = data[data['type'] == 'attacker']

rf1l = data[data['type'] == 'rf1l'] # The left cell Wi-Fi fingerprinting for reference point 1
rf1r = data[data['type'] == 'rf1r'] # The right cell Wi-Fi fingerprinting for reference point 1
rf1t = data[data['type'] == 'rf1t'] # The up cell Wi-Fi fingerprinting for reference point 1
rf1b = data[data['type'] == 'rf1b'] # The below cell Wi-Fi fingerprinting for reference point 1

rf2l = data[data['type'] == 'rf2l'] # The left cell Wi-Fi fingerprinting for reference point 2
rf2r = data[data['type'] == 'rf2r'] # The right cell Wi-Fi fingerprinting for reference point 2
rf2t = data[data['type'] == 'rf2t'] # The up cell Wi-Fi fingerprinting for reference point 2
rf2b = data[data['type'] == 'rf2b'] # The below cell Wi-Fi fingerprinting for reference point 2

rf3l = data[data['type'] == 'rf3l'] # The left cell Wi-Fi fingerprinting for reference point 3
rf3r = data[data['type'] == 'rf3r'] # The right cell Wi-Fi fingerprinting for reference point 3
rf3t = data[data['type'] == 'rf3t'] # The up cell Wi-Fi fingerprinting for reference point 3
rf3b = data[data['type'] == 'rf3b'] # The below cell Wi-Fi fingerprinting for reference point 3

rf4l = data[data['type'] == 'rf4l'] # The left cell Wi-Fi fingerprinting for reference point 4
rf4r = data[data['type'] == 'rf4r'] # The right cell Wi-Fi fingerprinting for reference point 4
rf4t = data[data['type'] == 'rf4t'] # The up cell Wi-Fi fingerprinting for reference point 4
rf4b = data[data['type'] == 'rf4b'] # The below cell Wi-Fi fingerprinting for reference point 4


rbf = Rbf(data['Drawing X'], data['Drawing Y'], data[beacon], function='linear')
z = rbf(gx, gy).reshape((num_y, num_x))
# Render the interpolated data to the plot
ax.imshow(z, vmin=-85, vmax=-25, extent=(0, grid_width, grid_height, 0),
                           cmap='RdYlBu_r', alpha=1)

# Overlay layout
ax.imshow(image, extent=(0, grid_width, grid_height, 0), interpolation='bicubic', zorder=100, alpha=0.6)


num_xcell = int(building_width/grid_cell_width)
num_ycell = int(building_length/grid_cell_length)

width_axhline = int(width/num_xcell)
height_avhline = int(height/num_ycell)


# Horizontal lines for y axes
loc_y = []
loc_x = []
line_color = 'blue'
ax.axhline(y=0, color='black' , linestyle='-', linewidth=2)
loc_y.append(0)
for lines in range(height_avhline, height-20, height_avhline): # 20 pixel is the distance error, modified according to your image
    loc_y.append(lines)
    ax.axhline(y=lines, color=line_color , linestyle='--', linewidth=1)
ax.axhline(y=height-20, color='black' , linestyle='-', linewidth=2)
loc_y.append(height-20)

# # Vertical lines for x axes
ax.axvline(x=0, color='black' , linestyle='-', linewidth=2)  # Vertical lines
loc_x.append(0)
for lines in range(width_axhline, width-10, width_axhline): # 10 pixel is the distance error, modified according to your image
    loc_x.append(lines)
    ax.axvline(x=lines, color=line_color, linestyle='--', linewidth=1)  # Vertical lines
ax.axvline(x=width-10, color='black' , linestyle='-', linewidth=2)  # Vertical lines
loc_x.append(width-10)

positions = {}

for i in range(len(loc_y)-1):
    for j in range(len(loc_x)-1):
        positions[(i,j)] = [(loc_y[i], loc_x[j]), (loc_y[i+1], loc_x[j+1])] # each cell location range by pixels



# Define the coordinates for the star
ap_y, ap_x = ap['y'].iloc[0], ap['x'].iloc[0]  # Adjust as needed
#ap1_y, ap1_x = ap1['y'].iloc[0], ap1['x'].iloc[0]  # Adjust as needed
size = 100  # Adjust as needed

# Draw the star for the AP
ax.scatter(ap_x, ap_y, marker='*', s=size, color='red', zorder=102)

# draw attack position
attacker_x = attacker['x'].iloc[0]
attacker_y = attacker['y'].iloc[0]
ax.scatter(attacker_x, attacker_y, marker='.', s=250, color='purple', zorder=102)

# Add reference point
ax.scatter(rf1['x'].iloc[0], rf1['y'].iloc[0], marker='.', s=250, color='red', zorder=102)
ax.scatter(rf2['x'].iloc[0], rf2['y'].iloc[0], marker='.', s=250, color='red', zorder=102)
ax.scatter(rf3['x'].iloc[0], rf3['y'].iloc[0], marker='.', s=250, color='red', zorder=102)
ax.scatter(rf4['x'].iloc[0], rf4['y'].iloc[0], marker='.', s=250, color='red', zorder=102)

position_rf1 = (math.ceil(rf1['y'].iloc[0]/height_avhline), math.ceil(rf1['x'].iloc[0]/width_axhline))
position_rf2 = (math.ceil(rf2['y'].iloc[0]/height_avhline), math.ceil(rf2['x'].iloc[0]/width_axhline))
position_rf3 = (math.ceil(rf3['y'].iloc[0]/height_avhline), math.ceil(rf3['x'].iloc[0]/width_axhline))
position_rf4 = (math.ceil(rf4['y'].iloc[0]/height_avhline), math.ceil(rf4['x'].iloc[0]/width_axhline))

positions[position_rf1].append(rf1['type', 'csi', -1, 'rssi', 'rtt'])
positions[position_rf1].append(rf2['type', 'csi', -1, 'rssi', 'rtt'])
positions[position_rf1].append(rf3['type', 'csi', -1, 'rssi', 'rtt'])
positions[position_rf1].append(rf4['type', 'csi', -1, 'rssi', 'rtt'])

position_ap = (math.ceil(ap_y/height_avhline), math.ceil(ap_x/width_axhline))
#position_ap1 = (math.ceil(ap1_y/height_avhline), math.ceil(ap1_x/width_axhline))
position_attacker = (math.ceil(attacker_y/height_avhline), math.ceil(attacker_x/width_axhline))

positions[position_attacker].append(attacker['type', 'csi', -1, 'rssi', 'rtt'])

# # if there is an obstacle， call estimate_rssi_from_csi to check the gradient, then correct the RSSI
def correct_rssi_by_CSI(file_amplitude, file_phase, rssi):
    csi_amplitudes = []
    csi_phases = []

    with open(file_amplitude, 'r') as f:
        csi_amplitudes.append(np.array(f.readlines(), type=float32))
    with open(file_phase, 'r') as f:
        csi_phases.append(np.array(f.readlines(), type=float32))

    csi_rssi = wg.estimate_rssi_from_csi(csi_amplitudes, csi_phases)

    rssi_difference = wg.calculate_rssi_gradient(csi_rssi, rssi)
    return rssi_difference


# build a gradient map
for i in range(len(loc_y)-1):
    for j in range(len(loc_x)-1):
        if (i, j) not in [position_rf1, position_rf2, position_rf3, position_rf4, attacker]:
            gradient_rf1 = []
            gradient_rf2 = []
            gradient_rf3 = []
            gradient_rf4 = []
            if i > position_rf1[0] and j > position_rf1[1]:
                mov_y = i - position_rf1[0]
                mov_x = j - position_rf1[1]
                gradient_csi_rf1 = rf1['csi'] + wg.calculate_csi_gradient(rf1b['csi'], rf1['csi'])*mov_y + wg.calculate_csi_gradient(rf1r['csi'], rf1['csi'])*mov_x
                gradient_distance_rf1 = math.sqrt(mov_y**2+mov_x**2)
                gradient_rssi_rf1 = rf1['rssi'] + wg.calculate_rssi_gradient(rf1b['rssi'], rf1['rssi'])*mov_y + wg.calculate_rssi_gradient(rf1r['rssi'], rf1['rssi'])*mov_x
                gradient_rtt_rf1 = rf1['rtt'] + wg.calculate_rtt_gradient(rf1b['rtt'], rf1['rtt']) * mov_y + wg.calculate_rtt_gradient(rf1r['rtt'], rf1['rtt']) * mov_x
                gradient_rf1 = ['rf1', gradient_csi_rf1, gradient_distance_rf1, gradient_rssi_rf1, gradient_rtt_rf1]
            elif i > position_rf1[0] and j < position_rf1[1]:
                mov_y = i - position_rf1[0]
                mov_x = position_rf1[1] - j
                gradient_csi_rf1 = rf1['csi'] + wg.calculate_csi_gradient(rf1b['csi'], rf1[
                    'csi']) * mov_y + wg.calculate_csi_gradient(rf1l['csi'], rf1['csi']) * mov_x
                gradient_distance_rf1 = math.sqrt(mov_y ** 2 + mov_x ** 2)
                gradient_rssi_rf1 = rf1['rssi'] + wg.calculate_rssi_gradient(rf1b['rssi'], rf1[
                    'rssi']) * mov_y + wg.calculate_rssi_gradient(rf1l['rssi'], rf1['rssi']) * mov_x
                gradient_rtt_rf1 = rf1['rtt'] + wg.calculate_rtt_gradient(rf1b['rtt'], rf1[
                    'rtt']) * mov_y + wg.calculate_rtt_gradient(rf1l['rtt'], rf1['rtt']) * mov_x
                gradient_rf1 = ['rf1', gradient_csi_rf1, gradient_distance_rf1, gradient_rssi_rf1, gradient_rtt_rf1]
            elif i < position_rf1[0] and j > position_rf1[1]:
                mov_y = position_rf1[0] - i
                mov_x = j - position_rf1[1]
                gradient_csi_rf1 = rf1['csi'] + wg.calculate_csi_gradient(rf1t['csi'], rf1[
                    'csi']) * mov_y + wg.calculate_csi_gradient(rf1r['csi'], rf1['csi']) * mov_x
                gradient_distance_rf1 = math.sqrt(mov_y ** 2 + mov_x ** 2)
                gradient_rssi_rf1 = rf1['rssi'] + wg.calculate_rssi_gradient(rf1t['rssi'], rf1[
                    'rssi']) * mov_y + wg.calculate_rssi_gradient(rf1r['rssi'], rf1['rssi']) * mov_x
                gradient_rtt_rf1 = rf1['rtt'] + wg.calculate_rtt_gradient(rf1t['rtt'], rf1[
                    'rtt']) * mov_y + wg.calculate_rtt_gradient(rf1r['rtt'], rf1['rtt']) * mov_x
                gradient_rf1 = ['rf1', gradient_csi_rf1, gradient_distance_rf1, gradient_rssi_rf1, gradient_rtt_rf1]
            elif i < position_rf1[0] and j < position_rf1[1]:
                mov_y = position_rf1[0] - i
                mov_x = position_rf1[1] - j
                gradient_csi_rf1 = rf1['csi'] + wg.calculate_csi_gradient(rf1t['csi'], rf1[
                    'csi']) * mov_y + wg.calculate_csi_gradient(rf1l['csi'], rf1['csi']) * mov_x
                gradient_distance_rf1 = math.sqrt(mov_y ** 2 + mov_x ** 2)
                gradient_rssi_rf1 = rf1['rssi'] + wg.calculate_rssi_gradient(rf1t['rssi'], rf1[
                    'rssi']) * mov_y + wg.calculate_rssi_gradient(rf1l['rssi'], rf1['rssi']) * mov_x
                gradient_rtt_rf1 = rf1['rtt'] + wg.calculate_rtt_gradient(rf1t['rtt'], rf1[
                    'rtt']) * mov_y + wg.calculate_rtt_gradient(rf1l['rtt'], rf1['rtt']) * mov_x
                gradient_rf1 = ['rf1', gradient_csi_rf1, gradient_distance_rf1, gradient_rssi_rf1, gradient_rtt_rf1]

            if i > position_rf2[0] and j > position_rf2[1]:
                mov_y = i - position_rf2[0]
                mov_x = j - position_rf2[1]
                gradient_csi_rf2 = rf2['csi'] + wg.calculate_csi_gradient(rf2b['csi'], rf2[
                    'csi']) * mov_y + wg.calculate_csi_gradient(rf2r['csi'], rf2['csi']) * mov_x
                gradient_distance_rf2 = math.sqrt(mov_y ** 2 + mov_x ** 2)
                gradient_rssi_rf2 = rf2['rssi'] + wg.calculate_rssi_gradient(rf2b['rssi'], rf2[
                    'rssi']) * mov_y + wg.calculate_rssi_gradient(rf2r['rssi'], rf2['rssi']) * mov_x
                gradient_rtt_rf2 = rf2['rtt'] + wg.calculate_rtt_gradient(rf2b['rtt'], rf2[
                    'rtt']) * mov_y + wg.calculate_rtt_gradient(rf2r['rtt'], rf2['rtt']) * mov_x
                gradient_rf2 = ['rf2', gradient_csi_rf2, gradient_distance_rf2, gradient_rssi_rf2, gradient_rtt_rf2]
            elif i > position_rf2[0] and j < position_rf2[1]:
                mov_y = i - position_rf2[0]
                mov_x = position_rf2[1] - j
                gradient_csi_rf2 = rf2['csi'] + wg.calculate_csi_gradient(rf2b['csi'], rf2[
                    'csi']) * mov_y + wg.calculate_csi_gradient(rf2l['csi'], rf2['csi']) * mov_x
                gradient_distance_rf2 = math.sqrt(mov_y ** 2 + mov_x ** 2)
                gradient_rssi_rf2 = rf2['rssi'] + wg.calculate_rssi_gradient(rf2b['rssi'], rf2[
                    'rssi']) * mov_y + wg.calculate_rssi_gradient(rf2l['rssi'], rf2['rssi']) * mov_x
                gradient_rtt_rf2 = rf2['rtt'] + wg.calculate_rtt_gradient(rf2b['rtt'], rf2[
                    'rtt']) * mov_y + wg.calculate_rtt_gradient(rf2l['rtt'], rf2['rtt']) * mov_x
                gradient_rf2 = ['rf2', gradient_csi_rf2, gradient_distance_rf2, gradient_rssi_rf2, gradient_rtt_rf2]
            elif i < position_rf2[0] and j > position_rf2[1]:
                mov_y = position_rf2[0] - i
                mov_x = j - position_rf2[1]
                gradient_csi_rf2 = rf2['csi'] + wg.calculate_csi_gradient(rf2t['csi'], rf2[
                    'csi']) * mov_y + wg.calculate_csi_gradient(rf2r['csi'], rf2['csi']) * mov_x
                gradient_distance_rf2 = math.sqrt(mov_y ** 2 + mov_x ** 2)
                gradient_rssi_rf2 = rf2['rssi'] + wg.calculate_rssi_gradient(rf2t['rssi'], rf2[
                    'rssi']) * mov_y + wg.calculate_rssi_gradient(rf2r['rssi'], rf2['rssi']) * mov_x
                gradient_rtt_rf2 = rf2['rtt'] + wg.calculate_rtt_gradient(rf2t['rtt'], rf2[
                    'rtt']) * mov_y + wg.calculate_rtt_gradient(rf2r['rtt'], rf2['rtt']) * mov_x
                gradient_rf2 = ['rf2', gradient_csi_rf2, gradient_distance_rf2, gradient_rssi_rf2, gradient_rtt_rf2]
            elif i < position_rf2[0] and j < position_rf2[1]:
                mov_y = position_rf2[0] - i
                mov_x = position_rf2[1] - j
                gradient_csi_rf2 = rf2['csi'] + wg.calculate_csi_gradient(rf2t['csi'], rf2[
                    'csi']) * mov_y + wg.calculate_csi_gradient(rf2l['csi'], rf2['csi']) * mov_x
                gradient_distance_rf2 = math.sqrt(mov_y ** 2 + mov_x ** 2)
                gradient_rssi_rf2 = rf2['rssi'] + wg.calculate_rssi_gradient(rf2t['rssi'], rf2[
                    'rssi']) * mov_y + wg.calculate_rssi_gradient(rf2l['rssi'], rf2['rssi']) * mov_x
                gradient_rtt_rf2 = rf2['rtt'] + wg.calculate_rtt_gradient(rf2t['rtt'], rf2[
                    'rtt']) * mov_y + wg.calculate_rtt_gradient(rf2l['rtt'], rf2['rtt']) * mov_x
                gradient_rf2 = ['rf2', gradient_csi_rf2, gradient_distance_rf2, gradient_rssi_rf2, gradient_rtt_rf2]

            if i > position_rf2[0] and j > position_rf2[1]:
                mov_y = i - position_rf2[0]
                mov_x = j - position_rf2[1]
                gradient_csi_rf2 = rf2['csi'] + wg.calculate_csi_gradient(rf2b['csi'], rf2[
                    'csi']) * mov_y + wg.calculate_csi_gradient(rf2r['csi'], rf2['csi']) * mov_x
                gradient_distance_rf2 = math.sqrt(mov_y ** 2 + mov_x ** 2)
                gradient_rssi_rf2 = rf2['rssi'] + wg.calculate_rssi_gradient(rf2b['rssi'], rf2[
                    'rssi']) * mov_y + wg.calculate_rssi_gradient(rf2r['rssi'], rf2['rssi']) * mov_x
                gradient_rtt_rf2 = rf2['rtt'] + wg.calculate_rtt_gradient(rf2b['rtt'], rf2[
                    'rtt']) * mov_y + wg.calculate_rtt_gradient(rf2r['rtt'], rf2['rtt']) * mov_x
                gradient_rf2 = ['rf2', gradient_csi_rf2, gradient_distance_rf2, gradient_rssi_rf2, gradient_rtt_rf2]
            elif i > position_rf2[0] and j < position_rf2[1]:
                mov_y = i - position_rf2[0]
                mov_x = position_rf2[1] - j
                gradient_csi_rf2 = rf2['csi'] + wg.calculate_csi_gradient(rf2b['csi'], rf2[
                    'csi']) * mov_y + wg.calculate_csi_gradient(rf2l['csi'], rf2['csi']) * mov_x
                gradient_distance_rf2 = math.sqrt(mov_y ** 2 + mov_x ** 2)
                gradient_rssi_rf2 = rf2['rssi'] + wg.calculate_rssi_gradient(rf2b['rssi'], rf2[
                    'rssi']) * mov_y + wg.calculate_rssi_gradient(rf2l['rssi'], rf2['rssi']) * mov_x
                gradient_rtt_rf2 = rf2['rtt'] + wg.calculate_rtt_gradient(rf2b['rtt'], rf2[
                    'rtt']) * mov_y + wg.calculate_rtt_gradient(rf2l['rtt'], rf2['rtt']) * mov_x
                gradient_rf2 = ['rf2', gradient_csi_rf2, gradient_distance_rf2, gradient_rssi_rf2, gradient_rtt_rf2]
            elif i < position_rf2[0] and j > position_rf2[1]:
                mov_y = position_rf2[0] - i
                mov_x = j - position_rf2[1]
                gradient_csi_rf2 = rf2['csi'] + wg.calculate_csi_gradient(rf2t['csi'], rf2[
                    'csi']) * mov_y + wg.calculate_csi_gradient(rf2r['csi'], rf2['csi']) * mov_x
                gradient_distance_rf2 = math.sqrt(mov_y ** 2 + mov_x ** 2)
                gradient_rssi_rf2 = rf2['rssi'] + wg.calculate_rssi_gradient(rf2t['rssi'], rf2[
                    'rssi']) * mov_y + wg.calculate_rssi_gradient(rf2r['rssi'], rf2['rssi']) * mov_x
                gradient_rtt_rf2 = rf2['rtt'] + wg.calculate_rtt_gradient(rf2t['rtt'], rf2[
                    'rtt']) * mov_y + wg.calculate_rtt_gradient(rf2r['rtt'], rf2['rtt']) * mov_x
                gradient_rf2 = ['rf2', gradient_csi_rf2, gradient_distance_rf2, gradient_rssi_rf2, gradient_rtt_rf2]
            elif i < position_rf2[0] and j < position_rf2[1]:
                mov_y = position_rf2[0] - i
                mov_x = position_rf2[1] - j
                gradient_csi_rf2 = rf2['csi'] + wg.calculate_csi_gradient(rf2t['csi'], rf2[
                    'csi']) * mov_y + wg.calculate_csi_gradient(rf2l['csi'], rf2['csi']) * mov_x
                gradient_distance_rf2 = math.sqrt(mov_y ** 2 + mov_x ** 2)
                gradient_rssi_rf2 = rf2['rssi'] + wg.calculate_rssi_gradient(rf2t['rssi'], rf2[
                    'rssi']) * mov_y + wg.calculate_rssi_gradient(rf2l['rssi'], rf2['rssi']) * mov_x
                gradient_rtt_rf2 = rf2['rtt'] + wg.calculate_rtt_gradient(rf2t['rtt'], rf2[
                    'rtt']) * mov_y + wg.calculate_rtt_gradient(rf2l['rtt'], rf2['rtt']) * mov_x
                gradient_rf2 = ['rf2', gradient_csi_rf2, gradient_distance_rf2, gradient_rssi_rf2, gradient_rtt_rf2]

            if i > position_rf4[0] and j > position_rf4[1]:
                mov_y = i - position_rf4[0]
                mov_x = j - position_rf4[1]
                gradient_csi_rf4 = rf4['csi'] + wg.calculate_csi_gradient(rf4b['csi'], rf4[
                    'csi']) * mov_y + wg.calculate_csi_gradient(rf4r['csi'], rf4['csi']) * mov_x
                gradient_distance_rf4 = math.sqrt(mov_y ** 2 + mov_x ** 2)
                gradient_rssi_rf4 = rf4['rssi'] + wg.calculate_rssi_gradient(rf4b['rssi'], rf4[
                    'rssi']) * mov_y + wg.calculate_rssi_gradient(rf4r['rssi'], rf4['rssi']) * mov_x
                gradient_rtt_rf4 = rf4['rtt'] + wg.calculate_rtt_gradient(rf4b['rtt'], rf4[
                    'rtt']) * mov_y + wg.calculate_rtt_gradient(rf4r['rtt'], rf4['rtt']) * mov_x
                gradient_rf4 = ['rf4', gradient_csi_rf4, gradient_distance_rf4, gradient_rssi_rf4, gradient_rtt_rf4]
            elif i > position_rf4[0] and j < position_rf4[1]:
                mov_y = i - position_rf4[0]
                mov_x = position_rf4[1] - j
                gradient_csi_rf4 = rf4['csi'] + wg.calculate_csi_gradient(rf4b['csi'], rf4[
                    'csi']) * mov_y + wg.calculate_csi_gradient(rf4l['csi'], rf4['csi']) * mov_x
                gradient_distance_rf4 = math.sqrt(mov_y ** 2 + mov_x ** 2)
                gradient_rssi_rf4 = rf4['rssi'] + wg.calculate_rssi_gradient(rf4b['rssi'], rf4[
                    'rssi']) * mov_y + wg.calculate_rssi_gradient(rf4l['rssi'], rf4['rssi']) * mov_x
                gradient_rtt_rf4 = rf4['rtt'] + wg.calculate_rtt_gradient(rf4b['rtt'], rf4[
                    'rtt']) * mov_y + wg.calculate_rtt_gradient(rf4l['rtt'], rf4['rtt']) * mov_x
                gradient_rf4 = ['rf4', gradient_csi_rf4, gradient_distance_rf4, gradient_rssi_rf4, gradient_rtt_rf4]
            elif i < position_rf4[0] and j > position_rf4[1]:
                mov_y = position_rf4[0] - i
                mov_x = j - position_rf4[1]
                gradient_csi_rf4 = rf4['csi'] + wg.calculate_csi_gradient(rf4t['csi'], rf4[
                    'csi']) * mov_y + wg.calculate_csi_gradient(rf4r['csi'], rf4['csi']) * mov_x
                gradient_distance_rf4 = math.sqrt(mov_y ** 2 + mov_x ** 2)
                gradient_rssi_rf4 = rf4['rssi'] + wg.calculate_rssi_gradient(rf4t['rssi'], rf4[
                    'rssi']) * mov_y + wg.calculate_rssi_gradient(rf4r['rssi'], rf4['rssi']) * mov_x
                gradient_rtt_rf4 = rf4['rtt'] + wg.calculate_rtt_gradient(rf4t['rtt'], rf4[
                    'rtt']) * mov_y + wg.calculate_rtt_gradient(rf4r['rtt'], rf4['rtt']) * mov_x
                gradient_rf4 = ['rf4', gradient_csi_rf4, gradient_distance_rf4, gradient_rssi_rf4, gradient_rtt_rf4]
            elif i < position_rf4[0] and j < position_rf4[1]:
                mov_y = position_rf4[0] - i
                mov_x = position_rf4[1] - j
                gradient_csi_rf4 = rf4['csi'] + wg.calculate_csi_gradient(rf4t['csi'], rf4[
                    'csi']) * mov_y + wg.calculate_csi_gradient(rf4l['csi'], rf4['csi']) * mov_x
                gradient_distance_rf4 = math.sqrt(mov_y ** 2 + mov_x ** 2)
                gradient_rssi_rf4 = rf4['rssi'] + wg.calculate_rssi_gradient(rf4t['rssi'], rf4[
                    'rssi']) * mov_y + wg.calculate_rssi_gradient(rf4l['rssi'], rf4['rssi']) * mov_x
                gradient_rtt_rf4 = rf4['rtt'] + wg.calculate_rtt_gradient(rf4t['rtt'], rf4[
                    'rtt']) * mov_y + wg.calculate_rtt_gradient(rf4l['rtt'], rf4['rtt']) * mov_x
                gradient_rf4 = ['rf4', gradient_csi_rf4, gradient_distance_rf4, gradient_rssi_rf4, gradient_rtt_rf4]

            gradient_from_rf = max(np.average(gradient_rf1[3]), np.average(gradient_rf2[3]), np.average(gradient_rf3[3]), np.average(gradient_rf4[3]))
            if gradient_from_rf == np.average(gradient_rf1[3]):
                positions[(i, j)].append(gradient_rf1)
            elif gradient_from_rf == np.average(gradient_rf2[3]):
                positions[(i, j)].append(gradient_rf2)
            elif gradient_from_rf == np.average(gradient_rf3[3]):
                positions[(i, j)].append(gradient_rf3)
            elif gradient_from_rf == np.average(gradient_rf4[3]):
                positions[(i, j)].append(gradient_rf4)

with open("../dataset/gradient_map.txt", "w") as f:
    f.write("rf,loc,rssi,rtt,bssid\n")
    for key, value in positions.items():
        f.write(f"{value[-1][0]},{key},{value[-1][3]},{value[-1][4]}\n")

# Hide axes
ax.axis('off')
#ax.set_xticks(x)
#ax.set_yticks(y)

# Show the plot
#plt.show()
plt.savefig('output_floor_map.png', dpi=2400)