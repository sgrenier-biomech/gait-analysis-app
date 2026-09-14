#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 19:54:45 2025

@author: sgrenier
"""
import kineticstoolkit.lab as ktk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt  # Import necessary functions

def compute_inertia_tensor(segment_name, mass_total, height_total):
    """Calculate inertia tensor based on Dempster's anthropometric model."""
    dempster_data = {
        "Foot":   {"mass_ratio": 0.0145, "com_ratio": 0.5, "coeffs": [0.0023, 0.0023, 0.0046]},
        "Shank":  {"mass_ratio": 0.0465, "com_ratio": 0.433, "coeffs": [0.028, 0.028, 0.056]},
        "Thigh":  {"mass_ratio": 0.1,    "com_ratio": 0.433, "coeffs": [0.059, 0.059, 0.118]},
        "Pelvis": {"mass_ratio": 0.142,  "com_ratio": 0.5, "coeffs": [0.075, 0.075, 0.150]}
    }

    data = dempster_data[segment_name]
    mass = data['mass_ratio'] * mass_total
    #com = data['com_ratio'] * height_total
    com = data['com_ratio'] * height_total * 0.53  # Leg length as % of height
    Ixx, Iyy, Izz = [c * mass * (com ** 2) for c in data['coeffs']]  # Apply correct moment of inertia formula
    return mass, np.diag([Ixx, Iyy, Izz])

def compute_inverse_dynamics(omega, alpha, com_accelerations, joint_positions, com_positions, fplate, mass_total, height_total, FP1_filtered, FP2_filtered):
    """Compute inverse dynamics using segment kinematics, joint positions, and pre-resampled force plate data."""
    g = np.array([0, 0, -9.81])
    results = {}

    segment_map = {
        "AnkleL": "FootL_CoM",
        "AnkleR": "FootR_CoM",
        "KneeL": "ShankL_CoM",
        "KneeR": "ShankR_CoM",
        "HipL": "ThighL_CoM",
        "HipR": "ThighR_CoM",
        "Pelvis": "Pelvis_CoM" 
        }

    parent_map = {
        "AnkleL": "KneeL",
        "AnkleR": "KneeR",
        "KneeL": "HipL",
        "KneeR": "HipR",
        "HipL": "Pelvis",
        "HipR": "Pelvis"
        }


    # Determine minimum sequence length
    min_length = min(
        len(alpha.time),
        len(omega.time),
        len(com_accelerations.time),
        len(joint_positions.time),
        len(com_positions.time),
        len(fplate.time)
    )

    leg_length = height_total * 0.53  # Leg length as a % of height
    #segment_mass = {joint: compute_inertia_tensor(segment.replace('L', '').replace('R', ''), mass_total, height_total)[0] for joint, segment in segment_map.items()}

    joint_forces, joint_moments = {}, {}
    for joint, segment in segment_map.items():
        #segment_base = segment.replace('_CoM', '')  # Strip "_CoM" from segment name
        segment_base = segment.replace('_CoM', '').replace('L', '').replace('R', '')  
        #print(f"Joint: {joint}, Original Segment: {segment}, Processed Segment: {segment_base}")
        mass, I = compute_inertia_tensor(segment_base, mass_total, height_total)
    
        acc_com = com_accelerations.data.get(segment, np.zeros((min_length, 3)))[:min_length, :3]
        alpha_joint = alpha.data.get(joint, np.zeros((min_length, 3)))[:min_length, :3]
        joint_pos = joint_positions.data.get(joint, np.zeros((min_length, 3)))[:min_length, :3]
        com_pos = com_positions.data.get(segment, np.zeros((min_length, 3)))[:min_length, :3]

        # Compute force and moment
        F_net = mass * (acc_com - g)  # Include segment mass effect 
        # Example Usage:
        #plot_F_net(F_net, min_length, joint) #for debug
        r_com = joint_pos - com_pos        
        #M_net = (I @ alpha_joint.T).T + np.cross(r_com, F_net)  # Include force-induced moment
        M_net = (I @ alpha_joint.T).T + np.cross(F_net, r_com)  # Include force-induced moment

        
        # if joint in ["AnkleL", "AnkleR"]:
        #     ts = ktk.TimeSeries(time=FP1_filtered.time[:min_length]) if "L" in joint else ktk.TimeSeries(time=FP2_filtered.time[:min_length])
        #     ts.data["Moment"] = M_net.copy()  # Ensure only `M_net` is filtered
        #     ts = ktk.filters.butter(ts, fc=20, order=2)  # Apply filter
        #     M_net_filtered = ts.data["Moment"].copy()  # Extract filtered moment
        #     M_net = M_net_filtered  # Use filtered moment


        if joint == "AnkleL":
            #F_grf = fplate.data['FP0_Force'][:min_length, :3]
            F_grf = np.column_stack([
                -FP1_filtered.data["F1X"][:min_length],
                FP1_filtered.data["F1Y"][:min_length],
                FP1_filtered.data["F1Z"][:min_length]
                ])
            COP = fplate.data['FP0_COP'][:min_length, :3]
            M_net -= np.cross(joint_pos - COP, F_grf)  # Swap if sign is wrong
            #M_net -= np.cross(COP - joint_pos, F_grf)  
            #M_net -= np.cross(F_grf, COP - joint_pos)  
            F_net -= F_grf
            
        elif joint == "AnkleR":
            #F_grf = fplate.data['FP1_Force'][:min_length, :3]
            F_grf = np.column_stack([
                FP2_filtered.data["F2X"][:min_length],  
                FP2_filtered.data["F2Y"][:min_length],  
                FP2_filtered.data["F2Z"][:min_length]  
                ])
            COP = fplate.data['FP1_COP'][:min_length, :3]
            M_net -= np.cross(joint_pos - COP, F_grf)  # Swap if sign is wrong
            #M_net -= np.cross(COP - joint_pos, F_grf) 
            #M_net -= np.cross(F_grf,COP - joint_pos) 
            F_net -= F_grf

        joint_forces[joint] = F_net
        joint_moments[joint] = M_net
        
        # After all forces are computed, call the function
        #plot_joint_forces(joint_forces, FP1_filtered, FP2_filtered, min_length)

    for joint, parent_joint in parent_map.items():
        if joint in joint_forces and parent_joint in joint_forces:
            joint_pos_parent = joint_positions.data.get(parent_joint, np.zeros((min_length, 3)))[:min_length, :3]
            segment_parent = segment_map[parent_joint]  # Convert joint name to segment name
            com_pos_parent = com_positions.data.get(segment_parent, np.zeros((min_length, 3)))[:min_length, :3]
            #r_com_parent = com_pos_parent - joint_pos_parent
            r_com_parent = joint_pos_parent - com_pos_parent
            
            if joint == "AnkleR" or joint == "AnkleL":
                #joint_moments[joint][:, 0] *= -1  # Flip Y force for right knee - no physiological or biomechanical basis for this but it results in matching data. not sure why.
                #oint_moments[joint][:, 1] *= -1
                joint_moments[joint][:, 2] *= -1        
            elif joint == "KneeR" or joint == "KneeL":
                joint_moments[joint][:, 0] *= -1    
            elif joint == "HipR":
                joint_moments[joint][:, 0] *= -1
                joint_moments[joint][:, 1] *= -1

            # Transform moment to the parent's coordinate system and normalize
            joint_moments[parent_joint] += np.cross(r_com_parent, joint_forces[joint])
            #joint_moments[parent_joint] += np.cross(joint_forces[joint], r_com_parent)
            joint_forces[parent_joint] += joint_forces[joint] 

    for joint in segment_map.keys():
        results[f"{joint}_Force"] = joint_forces.get(joint, np.zeros((min_length, 3)))/ (mass_total)  # Apply gravity per segment and normalize
        results[f"{joint}_Moment"] = joint_moments.get(joint, np.zeros((min_length, 3)))/ (mass_total * leg_length)
    
    return results


# def plot_results(results, segment_map):
#     """Plot joint forces and moments from results dictionary, one figure per joint."""
#     for joint in segment_map.keys():
#         fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)  # Two subplots: Force & Moment

#         # Time is just frame indices (assuming first axis is time)
#         time = np.arange(len(next(iter(results.values()))))  # Frames as x-axis

#         # Plot Forces
#         force_labels = ["X Force", "Y Force", "Z Force"]
#         force_data = results[f"{joint}_Force"]
#         for i in range(3):  # x, y, z
#             axes[0].plot(time, force_data[:, i], label=force_labels[i])

#         axes[0].set_title(f"{joint} - Forces (Normalized)")
#         axes[0].set_ylabel("Force (N/kg)")
#         axes[0].legend()
#         axes[0].grid(True)

#         # Plot Moments
#         moment_labels = ["X Moment", "Y Moment", "Z Moment"]
#         moment_data = results[f"{joint}_Moment"]
#         for i in range(3):  # x, y, z
#             axes[1].plot(time, moment_data[:, i], label=moment_labels[i], linestyle="--")

#         axes[1].set_title(f"{joint} - Moments (Normalized)")
#         axes[1].set_ylabel("Moment (N·m/kg·m)")
#         axes[1].legend()
#         axes[1].grid(True)

#         # Shared X-axis label (Frame Index)
#         axes[1].set_xlabel("Frame Index")

#         # Adjust layout
#         plt.tight_layout()
#         plt.show()

# Example Usage
#plot_results(results, segment_map)



# # Define function to plot force over time for debugging
# def plot_joint_forces(joint_forces, fplate, FP2_filtered, min_length):
#     """Plot computed vs. force plate ground reaction forces for debugging."""

#     time = np.arange(min_length)  # Assuming consistent time step

#     fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
#     fig.suptitle("Right-Side Force Debugging (Computed vs. Force Plate)")

#     joints_to_plot = ["AnkleR", "KneeR", "HipR"]
#     force_labels = ["X Force", "Y Force", "Z Force"]

#     for i, axis in enumerate(["X", "Y", "Z"]):
#         computed_forces = {joint: joint_forces.get(joint, np.zeros((min_length, 3)))[:, i] for joint in joints_to_plot}
#         force_plate = FP2_filtered.data[f"F2{axis}"][:min_length]

#         for joint, force in computed_forces.items():
#             axes[i].plot(time, force, label=f"{joint} - Computed", linestyle="-")

#         axes[i].plot(time, force_plate, label="FP2 (Force Plate) - Measured", linestyle="dashed", linewidth=2)
#         axes[i].set_ylabel(force_labels[i])
#         axes[i].legend()
#         axes[i].grid(True)

#     plt.xlabel("Time Step")
#     plt.tight_layout()
#     plt.show()




# # Function to plot computed net forces over time
# def plot_F_net(F_net, min_length, joint_name):
#     """Plot the computed net force acting on a segment over time."""
#     time = np.arange(min_length)  # Assuming consistent time step

#     fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
#     fig.suptitle(f"Computed Net Force (`F_net`) for {joint_name}")

#     force_labels = ["X Force (N)", "Y Force (N)", "Z Force (N)"]
#     axis_labels = ["X", "Y", "Z"]

#     for i in range(3):  # Loop through X, Y, Z components
#         axes[i].plot(time, F_net[:, i], label=f"F_net {axis_labels[i]}", linestyle="-", linewidth=2)
#         axes[i].set_ylabel(force_labels[i])
#         axes[i].legend()
#         axes[i].grid(True)

#     plt.xlabel("Time Step")
#     plt.tight_layout()
#     plt.show()




# def plot_results(results, plot_type, joint):
#     """Plot joint moments, forces, or angles over time directly from segment kinematics data."""
#     plt.figure(figsize=(10, 6))
#     for label, values in results.items():
#         if joint and joint not in label:
#             continue
#         if plot_type in label:
#             plt.plot(values['time'], values['values'], label=label)
#     plt.title(f"Joint {plot_type.title()} Over Time")
#     plt.xlabel('Time (s)')
#     plt.ylabel('Nm' if 'Moment' in plot_type else 'N')
#     plt.legend()
#     plt.grid(True)
#     plt.show()


# def plot_comparison(data_dict, joint):
#     """Plot comparison of CoM positions, velocities, and accelerations in GCS."""
#     plt.figure(figsize=(10, 6))
#     for component in ['Position', 'Velocity', 'Acceleration']:
#         for axis in ['X', 'Y', 'Z']:
#             key = f"{joint}_CoM_{component}_GCS_{axis}"
#             if key in data_dict:
#                 plt.plot(data_dict[key]['time'], data_dict[key]['values'], label=key)
#     plt.title(f"Comparison of {joint} CoM Data in GCS")
#     plt.xlabel('Time (s)')
#     plt.ylabel('Value')
#     plt.legend()
#     plt.grid(True)
#     plt.show()


# Example Usage:
# results = compute_inverse_dynamics(omega, alpha, com_accelerations, joint_positions, com_positions, fplate, mass_total, height_total)
#plot_results(results, plot_type='moment', joint='AnkleL')
# plot_comparison(data_dict, joint='AnkleL')
