#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 15:05:29 2025

@author: sgrenier
"""

import kineticstoolkit.lab as ktk
import numpy as np
#import pandas as pd
#import matplotlib
#matplotlib.use('Qt5Agg')  # Set Qt5 as the backend
import matplotlib.pyplot as plt
import os
import sys
from access_file import filename

EPSILON = 1e-6  # Avoid division by zero


data = ktk.read_c3d(str(filename))
data

markers = ktk.read_c3d(str(filename))["Points"] # assigning the points data to "markers" this way makes each channel accesible to list
markers.data #you can list it here and see what exactly is in the variable for plotting or other manipulation

force = ktk.read_c3d(str(filename))["Analogs"] # assigning the analog data to "force" this way makes each channel accesible to list
force.data #you can list it here and see what exactly is in the variable for plotting or other manipulation
#X = MedioLateral direction, Right +ve
#Y = Antero-posterior, Forward +ve
#Z = Up-Down, Up +ve
#right Hand System

# Sampling frequency
ForceSF = 1200  # Hz
# Number of samples (assumed from force data)
num_samples = len(force.data["F1X"])  # Assuming all channels have the same length
# Generate time array
time = np.arange(0, num_samples / ForceSF, 1 / ForceSF)  # Creates time values at 1200Hz
force.data["Time"] = time


# Scale all numerical data in the TimeSeries by 1000
# Define the keys that need to be scaled
force_to_scale = ["F1X", "F1Y", "F1Z", "F2X", "F2Y", "F2Z"]
# Apply scaling only to the specified keys
force.data = {key: (value * -1000 if key in force_to_scale else value) for key, value in force.data.items()}

# Moments_to_scale = ["M1X", "M1Y", "M1Z", "M2X", "M2Y", "M2Z"]
# # Apply scaling only to the specified keys
# force.data = {key: (value * 0.0001 if key in Moments_to_scale else value) for key, value in force.data.items()}

# #reshape the force for proper calibration
# # Extract force & moment components for each force plate
# F1 = np.vstack([
#     force.data["F1X"], force.data["F1Y"], force.data["F1Z"], 
#     force.data["M1X"], force.data["M1Y"], force.data["M1Z"]
# ]).T  # Shape: (72000, 6)

# F2 = np.vstack([
#     force.data["F2X"], force.data["F2Y"], force.data["F2Z"], 
#     force.data["M2X"], force.data["M2Y"], force.data["M2Z"]
# ]).T  # Shape: (72000, 6)

# # Stack both force plates into a single array
# raw_forces = np.stack([F1, F2], axis=-1)  # Shape: (72000, 6, 2)

# print("Reconstructed force data shape:", raw_forces.shape)

# # Add the directory where readMATfiles.py is located
# sys.path.append(os.path.abspath("/home/sgrenier/.config/spyder-py3/"))  # Update this path
# from readMATfiles import ForcePlatformCalibration  # Adjust the filename to match your Python module
# print("Calibration matrix loaded from external file:", ForcePlatformCalibration.shape)

# # Create an empty array for calibrated forces
# calibrated_forces = np.zeros_like(raw_forces)  # Same shape: (72000, 6, 2)

# # Apply calibration separately for each force plate
# for plate_idx in range(2):  # Iterate over two force plates
#     calibrated_forces[:, :, plate_idx] = np.matmul(
#         raw_forces[:, :, plate_idx],  # Raw force data
#         ForcePlatformCalibration[:, :, plate_idx].T  # Transposed calibration matrix
#     )

# print("Calibrated forces shape:", calibrated_forces.shape)  # Should be (72000, 6, 2)
# print("Calibrated forces applied successfully!")

# # Assign calibrated values back to force.data, overwriting the original raw forces
# force.data["F1X_cal"] = calibrated_forces[:, 0, 0]
# force.data["F1Y_cal"] = calibrated_forces[:, 1, 0]
# force.data["F1Z_cal"] = calibrated_forces[:, 2, 0]
# force.data["M1X_cal"] = calibrated_forces[:, 3, 0]
# force.data["M1Y_cal"] = calibrated_forces[:, 4, 0]
# force.data["M1Z_cal"] = calibrated_forces[:, 5, 0]

# force.data["F2X_cal"] = calibrated_forces[:, 0, 1]
# force.data["F2Y_cal"] = calibrated_forces[:, 1, 1]
# force.data["F2Z_cal"] = calibrated_forces[:, 2, 1]
# force.data["M2X_cal"] = calibrated_forces[:, 3, 1]
# force.data["M2Y_cal"] = calibrated_forces[:, 4, 1]
# force.data["M2Z_cal"] = calibrated_forces[:, 5, 1]

# print("Force data successfully overwritten with calibrated values!")
# force.data.keys()
# # Extract force plate data

# force.plot(["F1Z", "F1Z"])
# force.plot(["M1Y", "M1Y"])
# force.plot(["F1X", "F1Y", "F1Z", "F2X", "F2Y", "F2Z"], legend=True)
#force.plot(["M1X", "M1Y", "M1Z", "M2X", "M2Y", "M2Z"], legend=True)

#get the baseline data & assign to bias
# Extract specific channels into a new dictionary
# Extract only the selected channels as a new TimeSeries
FP1 = force.get_subset(["F1X", "F1Y", "F1Z", "M1X", "M1Y", "M1Z"])
FP1_bias = FP1.get_ts_between_times(6.6, 6.9, inclusive=False)
#FP1.plot()

FP2 = force.get_subset(["F2X", "F2Y", "F2Z", "M2X", "M2Y", "M2Z"])
FP2_bias = FP2.get_ts_between_times(14.0, 14.25, inclusive=False)
#FP2.plot()

# De-bias each channel in the force data Force plate 1
for channel in FP1.data.keys():    
    # Subtract the baseline mean from the entire channel
    FP1.data[channel] -= np.mean(FP1_bias.data[channel])
    
# De-bias each channel in the force data Force plate 2
for channel in FP2.data.keys():    
    # Subtract the baseline mean from the entire channel
    FP2.data[channel] -= np.mean(FP2_bias.data[channel])    

FP1_filtered = ktk.filters.butter(FP1, fc=20)
FP2_filtered = ktk.filters.butter(FP2, fc=20)
FP1_filtered.data
#plt.figure()  # Create another new figure
#FP1_filtered.plot(["M1Y"])
#plt.figure()  # Create another new figure
#FP2_filtered.plot(["F2Y"])
#FP1_filtered.plot(["F1Y"])

# FP1F_with_events = ktk.cycles.detect_cycles(
#     FP1_filtered, "F1Z", event_names=["Heel Strike", "Toe Off"], thresholds=[100, 60]
# )

# FP2F_with_events = ktk.cycles.detect_cycles(
#     FP2_filtered, "F2Z", event_names=["Heel Strike", "Toe Off"], thresholds=[100, 60]
# )

# F1S = FP1F_with_events
# F2S = FP2F_with_events # use this for COP calcs for INv Dyn?

# #F1S.plot(["F1X","F1Y","F1Z"])
# #plt.figure() 
# #F2S.plot(["F2X","F2Y","F2Z"])

# #F1S.plot()


# #Keep only event between HS 30 and TO 35 - or whatever # is yellow - currently only works for one FS
# FP1F = FP1F_with_events.get_ts_between_events("Heel Strike", "Toe Off", 30, 35, inclusive=True)
# #remove all other events so end up with only a single footstrike
# FS1 = FP1F.trim_events()
# FP2F = FP2F_with_events.get_ts_between_events("Heel Strike", "Toe Off", 30, 35, inclusive=True)
# FS2 = FP2F.trim_events()

# #Get all detected Heel Strike and Toe Off events, remove data between TO and HS
# FP1_heel_strike_events = [FS1.get_index_at_event("Heel Strike", occurrence=i) for i in range(6)]
# FP1_toe_off_events = [FS1.get_index_at_event("Toe Off", occurrence=i) for i in range(5)]
# FP2_heel_strike_events = [FS2.get_index_at_event("Heel Strike", occurrence=i) for i in range(6)]
# FP2_toe_off_events = [FS2.get_index_at_event("Toe Off", occurrence=i) for i in range(5)]

# # Ensure at least 5 pairs exist
# FP1_num_pairs = min(len(FP1_heel_strike_events), len(FP1_toe_off_events))
# FP2_num_pairs = min(len(FP2_heel_strike_events), len(FP2_toe_off_events))

# # Zero out values between each Toe Off and the next Heel Strike FP1
# for i in range(FP1_num_pairs):
#     TO_index = FP1_toe_off_events[i]  # Get i-th Toe Off index
#     HS_index = FP1_heel_strike_events[i+1]  # Get the next Heel Strike index
    
# #  Zero out values between TO and and subsequent HS for 3 COP channels
#     for key in FS1.data.keys():
#             FS1.data["F1Z"][TO_index:HS_index] = np.nan  # Replace with zeros
#             FS1.data["M1Y"][TO_index:HS_index] = np.nan
#             FS1.data["M1X"][TO_index:HS_index] = np.nan
            
# # Zero out values between each Toe Off and the next Heel Strike FP2
# for i in range(FP2_num_pairs):
#     TO_index = FP2_toe_off_events[i]  # Get i-th Toe Off index
#     HS_index = FP2_heel_strike_events[i+1]  # Get the next Heel Strike index
    
# #  Zero out values between TO and and subsequent HS for 3 COP channels
#     for key in FS2.data.keys():
#             FS2.data["F2Z"][TO_index:HS_index] = np.nan  # Replace with zeros
#             FS2.data["M2Y"][TO_index:HS_index] = np.nan
#             FS2.data["M2X"][TO_index:HS_index] = np.nan            

# #Calculate COP for both plates for how ever many FS chosen.
# FS1.data["COPx"] =  - FS1.data["M1Y"] / (FS1.data["F1Z"] + EPSILON)
# FS1.data["COPy"] =  + FS1.data["M1X"] / (FS1.data["F1Z"] + EPSILON)
# FS2.data["COPx"] =  - FS2.data["M2Y"] / (FS2.data["F2Z"] + EPSILON)
# FS2.data["COPy"] =  + FS2.data["M2X"] / (FS2.data["F2Z"] + EPSILON)

# fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True, sharey=False)

# # Plot for Quadrant 1 (Top Left)
# axes[0, 0].plot(FS1.data["COPx"], label="COPx vs time (F1)", color='blue')
# axes[0, 0].set_title("Plot of F1 COPx")
# axes[0, 0].set_xlabel("Samples")
# axes[0, 0].set_ylabel("COPx")
# axes[0, 0].legend()
# axes[0, 0].grid(True)

# # Plot for Quadrant 2 (Top Right)
# axes[0, 1].plot(FS1.data["COPy"], label="COPy vs time (F1)", color='red')
# axes[0, 1].set_title("Plot of F1 COPy")
# axes[0, 1].set_xlabel("Samples")
# axes[0, 1].set_ylabel("COPy")
# axes[0, 1].legend()
# axes[0, 1].grid(True)

# # Plot for Quadrant 3 (bottom Left)
# axes[1, 0].plot(FS2.data["COPx"], label="COPx vs time (F2)", color='orange')
# axes[1, 0].set_title("Plot of F2 COPx")
# axes[1, 0].set_xlabel("Samples")
# axes[1, 0].set_ylabel("COPx")
# axes[1, 0].legend()
# axes[1, 0].grid(True)

# # Plot for Quadrant 4 (bottom Right)
# axes[1, 1].plot(FS2.data["COPy"], label="COPy vs time (F2)", color='green')
# axes[1, 1].set_title("Plot of F2 COPy")
# axes[1, 1].set_xlabel("Samples")
# axes[1, 1].set_ylabel("COPy")
# axes[1, 1].legend()
# axes[1, 1].grid(True)

# #you can use this ln 176-184) instead if you prefer but then comment ln 141 to 173 
# plt.figure()  # Create another new figure
# FS1.plot(["COPx"])
# plt.figure() 
# FS1.plot(["COPy"])

# plt.figure()  # Create another new figure
# FS2.plot(["COPx"])
# plt.figure() 
# FS2.plot(["COPy"])

# # Access two variables from force.data by their keys
# x1 = FS1.data["COPx"]  # Replace with the actual key for the first variable
# y1 = FS1.data["COPy"]  # Replace with the actual key for the second variable
# x2 = FS2.data["COPx"]  
# y2 = FS2.data["COPy"]  


# # Create a figure with two subplots (1 row, 2 columns)
# fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

# # First subplot: Plot of F1 COP
# axes[0].plot(x1, y1, label="COPx vs COPy (F1)")
# axes[0].set_xlabel("COPx")  
# axes[0].set_ylabel("COPy")  
# axes[0].set_title("Plot of F1 COP")
# axes[0].legend()
# axes[0].grid(True)

# # Second subplot: Plot of F2 COP
# axes[1].plot(x2, y2, label="COPx vs COPy (F2)", color='orange')  # Use different color for distinction
# axes[1].set_xlabel("COPx")  
# axes[1].set_ylabel("COPy")  
# axes[1].set_title("Plot of F2 COP")
# axes[1].legend()
# axes[1].grid(True)

# # Adjust layout to prevent overlap
# plt.tight_layout()
# plt.show()




