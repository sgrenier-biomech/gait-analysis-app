#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 15:10:00 2026

@author: sgrenier
"""
import json
import tempfile
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
import kineticstoolkit.lab as ktk
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from inverse_dynamics_final2 import compute_inverse_dynamics
#from COP_final2 import FP1_filtered, FP2_filtered


INTERCONNECTIONS = {
    "Pelvis": {
        "Color": "#ff00ff",
        "Links": [
            ["SACR", "LASI", "RASI", "SACR"],
            ["LASI", "LGTR", "RGTR", "RASI"],
        ],
    },
    "Left Leg": {
        "Color": "#ff8c00",
        "Links": [
            ["LGTR", "LLEP", "LLML"],
            ["LLML", "LCAL", "L5TH", "LLML"],
        ],
    },
    "Right Leg": {
        "Color": "#00bfff",
        "Links": [
            ["RGTR", "RLEP", "RLML"],
            ["RLML", "RCAL", "R5TH", "RLML"],
        ],
    },
}

def process_cop(c3d_file_path: str):
    """Processes COP and force plate data dynamically from the uploaded C3D file."""
    # Reads the exact temporary file created by the uploader
    c3d_data = ktk.read_c3d(str(c3d_file_path))
    markers = c3d_data["Points"]
    force = c3d_data["Analogs"]

   #force.data #you can list it here and see what exactly is in the variable for plotting or other manipulation
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

    return FP1, FP2, FP1_filtered, FP2_filtered


def transform_to_omega(angles_ts, omega_raw_ts, sequence="XYZ"):
    time = omega_raw_ts.time
    omega_transformed = ktk.TimeSeries(time=time, data={})

    for joint_name in omega_raw_ts.data.keys():
        angles = np.radians(angles_ts.data[joint_name][:-1])
        omega_raw = np.radians(omega_raw_ts.data[joint_name])
        R_matrices = R.from_euler(sequence, angles, degrees=False).as_matrix()

        omega_corrected = np.zeros_like(angles)
        for i in range(len(angles)):
            R_i = R_matrices[i]
            omega_corrected[i] = R_i @ omega_raw[i]

        omega_transformed.data[joint_name] = omega_corrected

    return omega_transformed


def build_threejs_standalone_viewer(markers, interconnections, step=2):
    """
    Builds a completely self-contained WebGL 3D player.
    All orbit, zoom, scrubbing, and playback run client-side with 0 page reloads.
    """
    times = markers.time[::step].tolist()
    n_frames = len(times)

    # Determine coordinate scale (convert mm to m if values > 50)
    all_pts = []
    for m in markers.data.values():
        all_pts.append(m[::step, :3])
    stacked = np.concatenate(all_pts, axis=0)
    scale = 0.001 if float(np.nanmax(np.abs(stacked))) > 50.0 else 1.0

    # Calculate center of subject
    center_x = float(np.nanmean(stacked[:, 0]) * scale)
    center_y = float(np.nanmean(stacked[:, 1]) * scale)
    center_z = float(np.nanmean(stacked[:, 2]) * scale)

    # Map links
    segments = []
    for group_name, info in interconnections.items():
        color = info.get("Color", "#00ffff")
        for link in info["Links"]:
            segments.append({"color": color, "markers": link})

    # Prepare marker coordinate tables
    marker_dict = {}
    for m_name, m_data in markers.data.items():
        downsampled = m_data[::step, :3] * scale
        cleaned = np.where(np.isnan(downsampled), None, np.round(downsampled, 4))
        marker_dict[m_name] = cleaned.tolist()

    # Three.js Coordinate Map: X=X (M-L), Y=Z (Up), Z=-Y (A-P)
    data_payload = json.dumps({
        "times": times,
        "n_frames": n_frames,
        "segments": segments,
        "markers": marker_dict,
        "center": [center_x, center_z, -center_y]
    })

    html_code = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; user-select: none; }}
    body {{ background: #0b0f19; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; overflow: hidden; width: 100vw; height: 100vh; }}
    #viewport {{ width: 100%; height: 100%; position: absolute; top: 0; left: 0; }}
    #controls-card {{
      position: absolute; bottom: 16px; left: 50%; transform: translateX(-50%);
      width: calc(100% - 32px); max-width: 960px;
      background: rgba(15, 23, 42, 0.88); backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 12px; padding: 12px 20px;
      display: flex; align-items: center; gap: 16px; color: #f8fafc; z-index: 10;
      box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }}
    .btn {{
      background: #3b82f6; color: white; border: none; padding: 8px 18px;
      border-radius: 6px; font-weight: 600; cursor: pointer; font-size: 13px;
      transition: background 0.15s ease;
    }}
    .btn:hover {{ background: #2563eb; }}
    .btn-preset {{
      background: rgba(255, 255, 255, 0.08); border: 1px solid rgba(255, 255, 255, 0.15);
      color: #cbd5e1; padding: 6px 12px; font-size: 12px; border-radius: 4px; cursor: pointer;
    }}
    .btn-preset:hover {{ background: rgba(255, 255, 255, 0.2); color: white; }}
    input[type=range] {{
      flex-grow: 1; accent-color: #3b82f6; cursor: pointer; height: 6px;
    }}
    .readout {{
      font-size: 13px; font-variant-numeric: tabular-nums; min-width: 100px;
      color: #94a3b8; text-align: right;
    }}
    .preset-group {{
      position: absolute; top: 16px; left: 16px; display: flex; gap: 8px; z-index: 10;
    }}
  </style>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
  <div class="preset-group">
    <button class="btn-preset" onclick="setPreset('side')">Sagittal (Side)</button>
    <button class="btn-preset" onclick="setPreset('front')">Frontal</button>
    <button class="btn-preset" onclick="setPreset('top')">Transverse (Top)</button>
    <button class="btn-preset" onclick="setPreset('iso')">Isometric</button>
  </div>

  <div id="viewport"></div>

  <div id="controls-card">
    <button id="playBtn" class="btn">Play</button>
    <input type="range" id="scrubber" min="0" max="{n_frames - 1}" value="0" step="1">
    <span id="readout" class="readout">0.00s | 0</span>
    <select id="speed" style="background:#1e293b; color:white; border:1px solid #475569; padding:5px 8px; border-radius:6px; font-size:12px;">
      <option value="0.25">0.25x</option>
      <option value="0.5">0.5x</option>
      <option value="1" selected>1.0x</option>
      <option value="1.5">1.5x</option>
      <option value="2">2.0x</option>
    </select>
  </div>

  <script id="motionData" type="application/json">
    {data_payload}
  </script>

  <script>
    const data = JSON.parse(document.getElementById('motionData').textContent);
    const nFrames = data.n_frames;

    const cx = data.center[0] || 0;
    const cy = data.center[1] || 1;
    const cz = data.center[2] || 0;

    const container = document.getElementById('viewport');
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b0f19);

    const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.05, 500);
    camera.position.set(cx + 1.8, cy + 0.6, cz + 2.0);

    const renderer = new THREE.WebGLRenderer({{ antialias: true }});
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.target.set(cx, cy, cz);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    // Floor grid & subtle coordinate lights
    const grid = new THREE.GridHelper(3.0, 15, 0x334155, 0x1e293b);
    grid.position.set(cx, 0, cz);
    scene.add(grid);
    scene.add(new THREE.AmbientLight(0xffffff, 1.0));

    // Construct Segment Mesh Lines & Joint Spheres
    const lineObjects = [];
    const jointSpheres = [];

    data.segments.forEach(seg => {{
      const geom = new THREE.BufferGeometry();
      const posArr = new Float32Array(seg.markers.length * 3);
      geom.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
      const mat = new THREE.LineBasicMaterial({{ color: seg.color, linewidth: 3 }});
      const line = new THREE.Line(geom, mat);
      scene.add(line);
      lineObjects.push({{ line, markers: seg.markers }});
    }});

    const sphereGeom = new THREE.SphereGeometry(0.015, 12, 12);
    const sphereMat = new THREE.MeshBasicMaterial({{ color: 0xffffff }});
    Object.keys(data.markers).forEach(m => {{
      const mesh = new THREE.Mesh(sphereGeom, sphereMat);
      scene.add(mesh);
      jointSpheres.push({{ mesh, marker: m }});
    }});

    function updatePose(fIdx) {{
      // Update Skeleton Lines
      lineObjects.forEach(obj => {{
        const posAttr = obj.line.geometry.attributes.position;
        const arr = posAttr.array;
        let ptr = 0;
        obj.markers.forEach(m => {{
          const pt = data.markers[m] ? data.markers[m][fIdx] : null;
          if (pt && pt[0] !== null) {{
            arr[ptr++] = pt[0];
            arr[ptr++] = pt[2];
            arr[ptr++] = -pt[1];
          }} else {{
            arr[ptr++] = cx; arr[ptr++] = cy; arr[ptr++] = cz;
          }}
        }});
        posAttr.needsUpdate = true;
      }});

      // Update Joint Points
      jointSpheres.forEach(obj => {{
        const pt = data.markers[obj.marker] ? data.markers[obj.marker][fIdx] : null;
        if (pt && pt[0] !== null) {{
          obj.mesh.position.set(pt[0], pt[2], -pt[1]);
          obj.mesh.visible = true;
        }} else {{
          obj.mesh.visible = false;
        }}
      }});

      document.getElementById('scrubber').value = fIdx;
      const t = data.times[fIdx] !== undefined ? data.times[fIdx].toFixed(2) : "0.00";
      document.getElementById('readout').textContent = `${{t}}s | ${{fIdx}}`;
    }}

    // Preset helper (preserves target center)
    window.setPreset = function(type) {{
      const dist = 2.4;
      if (type === 'side') camera.position.set(cx + dist, cy, cz);
      else if (type === 'front') camera.position.set(cx, cy, cz + dist);
      else if (type === 'top') camera.position.set(cx + 0.001, cy + dist, cz);
      else if (type === 'iso') camera.position.set(cx + 1.6, cy + 1.0, cz + 1.6);
      controls.target.set(cx, cy, cz);
    }};

    // Playback loop
    let isPlaying = false;
    let currentFrameFloat = 0;
    let lastTime = performance.now();

    const playBtn = document.getElementById('playBtn');
    const scrubber = document.getElementById('scrubber');
    const speedSelect = document.getElementById('speed');

    playBtn.addEventListener('click', () => {{
      isPlaying = !isPlaying;
      playBtn.textContent = isPlaying ? 'Pause' : 'Play';
      lastTime = performance.now();
    }});

    scrubber.addEventListener('input', (e) => {{
      isPlaying = false;
      playBtn.textContent = 'Play';
      currentFrameFloat = parseInt(e.target.value);
      updatePose(currentFrameFloat);
    }});

    function loop(now) {{
      requestAnimationFrame(loop);
      controls.update();

      if (isPlaying && nFrames > 0) {{
        const dt = (now - lastTime) / 1000;
        lastTime = now;
        const spd = parseFloat(speedSelect.value);
        currentFrameFloat = (currentFrameFloat + (dt * 60 * spd)) % nFrames;
        updatePose(Math.floor(currentFrameFloat));
      }} else {{
        lastTime = now;
      }}

      renderer.render(scene, camera);
    }}

    updatePose(0);
    requestAnimationFrame(loop);

    window.addEventListener('resize', () => {{
      camera.aspect = container.clientWidth / container.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, container.clientHeight);
    }});
  </script>
</body>
</html>"""
    return html_code


def run_segment_kinematics(c3d_file_path: str, mass_total: float, height_total: float):
    markers = ktk.read_c3d(c3d_file_path)["Points"]
    markers = ktk.filters.butter(markers, fc=6)


    raw_analogs = ktk.read_c3d(c3d_file_path)["Analogs"]  # Completely raw, un-debiased
    
    fplate = ktk.read_c3d(c3d_file_path)["ForcePlatforms"]
    fplate.resample(120, kind="linear", in_place=True)
    fplate = ktk.filters.median(fplate, window_length=5)

    frames = ktk.TimeSeries(time=markers.time)
    joint_positions = ktk.TimeSeries(time=markers.time)

    pelvis_origin = markers.data["SACR"]
    y_vec = (0.5 * (markers.data["RASI"] + markers.data["LASI"])) - pelvis_origin
    xy_vec = markers.data["RASI"] - pelvis_origin
    frames.data["Pelvis"] = ktk.geometry.create_transform_series(positions=pelvis_origin, y=y_vec, xy=xy_vec)
    joint_positions.data["Pelvis"] = pelvis_origin

    LtHip_Origin = 0.5 * (markers.data["LGTR"] + markers.data["LASI"])
    z_vec_lt = LtHip_Origin - (0.5 * (markers.data["LLEP"] + markers.data["LMEP"]))
    xz_vec_lt = markers.data["LMEP"] - markers.data["LLEP"]
    frames.data["ThighL"] = ktk.geometry.create_transform_series(positions=LtHip_Origin, z=z_vec_lt, xz=xz_vec_lt)
    joint_positions.data["HipL"] = LtHip_Origin

    RtHip_Origin = 0.5 * (markers.data["RGTR"] + markers.data["RASI"])
    z_vec_rt = RtHip_Origin - (0.5 * (markers.data["RLEP"] + markers.data["RMEP"]))
    xz_vec_rt = markers.data["RLEP"] - markers.data["RMEP"]
    frames.data["ThighR"] = ktk.geometry.create_transform_series(positions=RtHip_Origin, z=z_vec_rt, xz=xz_vec_rt)
    joint_positions.data["HipR"] = RtHip_Origin

    LtShk_origin = 0.5 * (markers.data["LLEP"] + markers.data["LMEP"])
    z_vec_lshk = LtShk_origin - 0.5 * (markers.data["LMML"] + markers.data["LLML"])
    yz_vec_lshk = markers.data["LSH2"] - markers.data["LSH3"]
    frames.data["ShankL"] = ktk.geometry.create_transform_series(positions=LtShk_origin, z=z_vec_lshk, yz=yz_vec_lshk)
    joint_positions.data["KneeL"] = LtShk_origin

    RtShk_origin = 0.5 * (markers.data["RLEP"] + markers.data["RMEP"])
    z_vec_rshk = RtShk_origin - 0.5 * (markers.data["RMML"] + markers.data["RLML"])
    yz_vec_rshk = markers.data["RSH3"] - markers.data["RSH2"]
    frames.data["ShankR"] = ktk.geometry.create_transform_series(positions=RtShk_origin, z=z_vec_rshk, yz=yz_vec_rshk)
    joint_positions.data["KneeR"] = RtShk_origin

    LtFoot_origin = 0.5 * (markers.data["LMML"] + markers.data["LLML"])
    y_vec_lft = markers.data["L5TH"] - LtFoot_origin
    xy_vec_lft = markers.data["LMML"] - LtFoot_origin
    frames.data["FootL"] = ktk.geometry.create_transform_series(positions=LtFoot_origin, y=y_vec_lft, xy=xy_vec_lft)
    joint_positions.data["AnkleL"] = LtFoot_origin

    RtFoot_origin = 0.5 * (markers.data["RMML"] + markers.data["RLML"])
    y_vec_rft = markers.data["R5TH"] - RtFoot_origin
    xy_vec_rft = markers.data["RLML"] - RtFoot_origin
    frames.data["FootR"] = ktk.geometry.create_transform_series(positions=RtFoot_origin, y=y_vec_rft, xy=xy_vec_rft)
    joint_positions.data["AnkleR"] = RtFoot_origin

    Pelvis_to_ThighL_LCS = ktk.geometry.get_local_coordinates(frames.data["ThighL"], frames.data["Pelvis"])
    Pelvis_to_ThighR_LCS = ktk.geometry.get_local_coordinates(frames.data["ThighR"], frames.data["Pelvis"])
    ThighL_to_ShankL_LCS = ktk.geometry.get_local_coordinates(frames.data["ShankL"], frames.data["ThighL"])
    ThighR_to_ShankR_LCS = ktk.geometry.get_local_coordinates(frames.data["ShankR"], frames.data["ThighR"])
    ShankL_to_FootL_LCS = ktk.geometry.get_local_coordinates(frames.data["FootL"], frames.data["ShankL"])
    ShankR_to_FootR_LCS = ktk.geometry.get_local_coordinates(frames.data["FootR"], frames.data["ShankR"])

    angles = ktk.TimeSeries(time=markers.time)
    angles.data["HipL"] = ktk.geometry.get_angles(Pelvis_to_ThighL_LCS, "XZY", degrees=True)
    angles.data["HipR"] = ktk.geometry.get_angles(Pelvis_to_ThighR_LCS, "XZY", degrees=True)
    angles.data["KneeL"] = ktk.geometry.get_angles(ThighL_to_ShankL_LCS, "XZY", degrees=True)
    angles.data["KneeR"] = ktk.geometry.get_angles(ThighR_to_ShankR_LCS, "XZY", degrees=True)
    angles.data["AnkleL"] = ktk.geometry.get_angles(ShankL_to_FootL_LCS, "XYZ", degrees=True)
    angles.data["AnkleR"] = ktk.geometry.get_angles(ShankR_to_FootR_LCS, "XYZ", degrees=True)

    com_positions = ktk.TimeSeries(time=markers.time)
    com_positions.data["Pelvis_CoM"] = LtHip_Origin + 0.5 * (RtHip_Origin - LtHip_Origin)
    com_positions.data["ThighL_CoM"] = LtHip_Origin + (0.433 * (LtShk_origin - LtHip_Origin))
    com_positions.data["ThighR_CoM"] = RtHip_Origin - (0.433 * (RtShk_origin - RtHip_Origin))
    com_positions.data["ShankL_CoM"] = LtShk_origin + (0.433 * (LtFoot_origin - LtShk_origin))
    com_positions.data["ShankR_CoM"] = RtShk_origin + (0.433 * (RtFoot_origin - RtShk_origin))
    com_positions.data["FootL_CoM"] = LtFoot_origin + (0.5 * (markers.data["L5TH"] - LtFoot_origin))
    com_positions.data["FootR_CoM"] = RtFoot_origin + (0.5 * (markers.data["R5TH"] - RtFoot_origin))

    com_velocities = ktk.filters.deriv(com_positions)
    com_accelerations = ktk.filters.deriv(com_velocities)

    omega_raw = ktk.filters.deriv(angles)
    omega = transform_to_omega(angles, omega_raw)
    omega_filt = ktk.filters.butter(omega, fc=5)
    alpha = ktk.filters.deriv(omega_filt)

    FP1,FP2, FP1_filtered, FP2_filtered = process_cop(c3d_file_path)
    fp1 = FP1_filtered.copy()
    fp2 = FP2_filtered.copy()
    fp1.resample(120, kind="linear", in_place=True)
    fp2.resample(120, kind="linear", in_place=True)

    results = compute_inverse_dynamics(
        omega, alpha, com_accelerations, joint_positions, com_positions, fplate,
        mass_total, height_total, FP1_filtered=fp1, FP2_filtered=fp2
    )
    return markers, angles, results, FP1, FP2, raw_analogs


# =======================================================
# STREAMLIT UI
# =======================================================
st.set_page_config(page_title="Gait analysis app", layout="wide")
st.title("Gait analysis app")

st.sidebar.header("Subject Parameters")
mass = st.sidebar.number_input("Total Mass (kg)", value=77.0, step=0.5)
height = st.sidebar.number_input("Total Height (m)", value=1.78, step=0.01)

st.sidebar.header("Data Selection")
uploaded_file = st.sidebar.file_uploader("Upload a C3D file", type=["c3d"])

selected_file_path = None

if uploaded_file is not None:
   with tempfile.NamedTemporaryFile(delete=False, suffix=".c3d") as tmp:
       tmp.write(uploaded_file.read())
       selected_file_path = tmp.name

if st.button("Process Data", type="primary"):
    if not selected_file_path:
        st.error("Select or upload a .c3d file first.")
    else:
        with st.spinner("Processing file..."):
            try:
                markers, angles, results, FP1, FP2, raw_analogs = run_segment_kinematics(selected_file_path, mass, height)
                st.session_state["markers"] = markers
                st.session_state["angles"] = angles
                st.session_state["results"] = results
                st.session_state["FP1_debiased"] = FP1
                st.session_state["FP2_debiased"] = FP2
                st.session_state["raw_analogs"] = raw_analogs

                # Default states for student decisions
                st.session_state["cycle_locked"] = False
                st.session_state["filter_locked"] = False
                st.session_state["t_start"] = float(angles.time[0])
                st.session_state["t_end"] = float(min(angles.time[0] + 1.2, angles.time[-1]))
                st.session_state["chosen_fc"] = 100  # Default initial high cutoff
                st.session_state["apply_debias"] = True
            except Exception as e:
                st.error(f"Error: {e}")

# Initialize persistent student workflow states
for key, default in [
    ("cycle_locked", False),
    ("filter_locked", False),
    ("t_start", 0.0),
    ("t_end", 2.0),
    ("chosen_plate", "FP1"),
    ("chosen_filter", "Butterworth Low-pass"),
    ("chosen_fc", 20),
]:
  if key not in st.session_state:
    st.session_state[key] = default

if (
    "angles" in st.session_state
    and "markers" in st.session_state
    and "FP1_debiased" in st.session_state
):
  st.markdown("---")

  # Define dynamic tabs that unlock sequentially
  tab_labels = ["1. Kinematics & Cycle Selection"]
  if st.session_state["cycle_locked"]:
    tab_labels.append("2. GRF Filter Tuning")
  if st.session_state["filter_locked"]:
    tab_labels.extend(
        ["3. COP Analysis", "4. Joint Kinetics", "5. 3D Animation"]
    )

  active_tabs = st.tabs(tab_labels)

# =========================================================================
  # STEP 1: KINEMATICS & GAIT CYCLE IDENTIFICATION
  # =========================================================================
  with active_tabs[0]:
    st.subheader("Step 1: Identify and Isolate One Gait Cycle")
    st.caption(
        "Inspect sagittal joint kinematics to identify consecutive heel"
        " strikes (0% to 100% of a single cycle)."
    )

    angles_ts = st.session_state["angles"]
    t_min = float(angles_ts.time[0])
    t_max = float(angles_ts.time[-1])

    # Let students inspect the curve with Ankle joints included
    joint_col, _ = st.columns([1, 2])
    with joint_col:
      chosen_joint = st.selectbox(
          "Inspection Joint:",
          [
              "KneeR",
              "KneeL",
              "HipR",
              "HipL",
              "AnkleR",
              "AnkleL",
          ],  # Added ankle joints
      )

    import plotly.graph_objects as go

    fig_kin = go.Figure()
    fig_kin.add_trace(
        go.Scatter(
            x=angles_ts.time,
            y=angles_ts.data[chosen_joint][:, 0],
            mode="lines",
            name="Flexion / Extension",
            line=dict(color="#3b82f6", width=2),
        )
    )

    # Visual highlight of selected window
    t_s = st.session_state["t_start"]
    t_e = st.session_state["t_end"]
    fig_kin.add_vrect(
        x0=t_s,
        x1=t_e,
        fillcolor="rgba(34, 197, 94, 0.15)",
        line_width=1,
        line_dash="dash",
        line_color="#22c55e",
        annotation_text="Selected Cycle",
    )
    fig_kin.update_layout(
        title=f"{chosen_joint} Sagittal Angle (Flexion/Extension)",
        xaxis_title="Time (s)",
        yaxis_title="Angle (deg)",
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig_kin, use_container_width=True)

# 1. Define callback function that runs BEFORE widgets re-render
    if "step1_version" not in st.session_state:
        st.session_state["step1_version"] = 0
        
# Set default values for initial render
    default_start = t_min
    default_end = float(min(t_min + 1.2, t_max))

    # Active key suffix tied to reset counter
    v = st.session_state.get("step1_version", 0)
    k_start = f"input_t_start_{v}"
    k_end = f"input_t_end_{v}"

    st.markdown("#### 🎯 Student Decision: Set Cycle Bounds")
    c_col1, c_col2, c_col3, c_col4 = st.columns([2, 2, 1.2, 1])

    with c_col1:
      val_s = st.number_input(
          "Cycle Initial Contact (s):",
          min_value=t_min,
          max_value=t_max,
          value=float(st.session_state["t_start"]),
          step=0.01,
          format="%.3f",
          key=k_start,
      )
    with c_col2:
      val_e = st.number_input(
          "Next Initial Contact (s):",
          min_value=t_min,
          max_value=t_max,
          value=float(st.session_state["t_end"]),
          step=0.01,
          format="%.3f",
          key=k_end,
      )
    with c_col3:
      st.write("")
      st.write("")
      if st.button("Lock In Gait Cycle", type="primary"):
        if val_e > val_s:
          st.session_state["t_start"] = val_s
          st.session_state["t_end"] = val_e
          st.session_state["cycle_locked"] = True
          st.rerun()
        else:
          st.error("End time must be greater than start time.")
    with c_col4:
      st.write("")
      st.write("")
      if st.button("Reset Selection"):
        # Reset locks
        st.session_state["cycle_locked"] = False
        st.session_state["filter_locked"] = False

        # Reset times to defaults
        st.session_state["t_start"] = default_start
        st.session_state["t_end"] = default_end

        # Bump version counter to force Streamlit to wipe the widget cache
        st.session_state["step1_version"] = v + 1
        st.rerun()

    if st.session_state["cycle_locked"]:
      st.success(
          f"Cycle locked: {st.session_state['t_start']:.3f}s to"
          f" {st.session_state['t_end']:.3f}s (Duration:"
          f" {st.session_state['t_end'] - st.session_state['t_start']:.3f}s)."
          " Proceed to Step 2."
      )
  # =========================================================================
  # STEP 2: FORCE SIGNAL FILTERING DECISION
  # =========================================================================
if st.session_state["cycle_locked"]:
    with active_tabs[1]:
      st.subheader("Step 2: Ground Reaction Force (GRF) Processing Decisions")
      st.caption(
          "Inspect raw forces, optionally refine your gait cycle window,"
          " perform baseline zeroing (de-biasing), and evaluate filter cutoffs."
      )

      # ---------------------------------------------------------
      # Optional Window Refinement
      # ---------------------------------------------------------
      with st.expander(
          "🔍 Optional: Refine Gait Cycle Window Timing", expanded=False
      ):
        w_col1, w_col2, w_col3 = st.columns([2, 2, 1])
        with w_col1:
          ref_t_start = st.number_input(
              "Adjust Cycle Start (s):",
              value=st.session_state["t_start"],
              step=0.005,
              format="%.3f",
          )
        with w_col2:
          ref_t_end = st.number_input(
              "Adjust Cycle End (s):",
              value=st.session_state["t_end"],
              step=0.005,
              format="%.3f",
          )
        with w_col3:
          st.write("")
          st.write("")
          if st.button("Update Window"):
            if ref_t_end > ref_t_start:
              st.session_state["t_start"] = ref_t_start
              st.session_state["t_end"] = ref_t_end
              st.success("Window bounds updated.")
              st.rerun()
            else:
              st.error("End time must be greater than start time.")

      # ---------------------------------------------------------
      # Signal Decision Controls
      # ---------------------------------------------------------
      st.markdown("#### 1. Force Plate & Baseline Zeroing Decisions")
      c_col1, c_col2, c_col3 = st.columns(3)

      with c_col1:
        plate_choice = st.radio(
            "Select Force Plate:",
            ["FP1", "FP2"],
            horizontal=True,
            key="grf_plate_sel",
        )

      with c_col2:
        debias_choice = st.checkbox(
            "Apply Baseline Zeroing (De-bias)",
            value=st.session_state.get("apply_debias", False),
            help=(
                "Subtracts baseline offset measured during quiescent phase"
                " before footstrike."
            ),
        )

      with c_col3:
        if debias_choice:
          # Define quiescent baseline intervals
          default_interval = (
              (6.6, 6.9) if plate_choice == "FP1" else (14.0, 14.25)
          )
          b_start, b_end = st.slider(
              "Baseline Sampling Interval (s):",
              min_value=0.0,
              max_value=float(st.session_state["raw_analogs"].time[-1]),
              value=default_interval,
              step=0.05,
          )

      st.markdown("#### 2. Filtering Decisions")
      f_col1, f_col2 = st.columns(2)
      with f_col1:
        filter_mode = st.selectbox(
            "Filter Algorithm:",
            ["Butterworth Low-pass", "None (Raw)"],
            index=0,
        )

      with f_col2:
        if filter_mode == "Butterworth Low-pass":
          cutoff_fc = st.slider(
              "Cutoff Frequency Fc (Hz):",
              min_value=5,
              max_value=200,
              value=st.session_state.get("chosen_fc", 100),  # Defaults to 100 Hz
              step=5,
              help=(
                  "Starts at 100 Hz (minimal attenuation). Lower this value to"
                  " eliminate high-frequency vibration/impact artifacts."
              ),
          )
        else:
          cutoff_fc = None

      # ---------------------------------------------------------
      # Process the Signals Based on Student Decisions
      # ---------------------------------------------------------
      raw_analogs = st.session_state["raw_analogs"]
      channels = (
          ["F1X", "F1Y", "F1Z", "M1X", "M1Y", "M1Z"]
          if plate_choice == "FP1"
          else ["F2X", "F2Y", "F2Z", "M2X", "M2Y", "M2Z"]
      )
      fp_working = raw_analogs.get_subset(channels)

      # Scale forces to Newtons (x -1000)
      for k in [f"{plate_choice[0:2]}X", f"{plate_choice[0:2]}Y", f"{plate_choice[0:2]}Z"]:
        if k in fp_working.data:
          fp_working.data[k] = fp_working.data[k] * -1000.0

      # Optional student debiasing
      if debias_choice:
        bias_ts = fp_working.get_ts_between_times(
            b_start, b_end, inclusive=False
        )
        for k in fp_working.data.keys():
          fp_working.data[k] -= np.mean(bias_ts.data[k])

      # Trim down to the student's chosen gait cycle window
      cycle_fp_raw = fp_working.get_ts_between_times(
          st.session_state["t_start"], st.session_state["t_end"], inclusive=True
      )

      # Apply student filter choice
      cycle_fp_filt = None
      if filter_mode == "Butterworth Low-pass":
        cycle_fp_filt = ktk.filters.butter(cycle_fp_raw, fc=cutoff_fc)

      # ---------------------------------------------------------
      # Render Interactive Visual Comparison
      # ---------------------------------------------------------
      import plotly.graph_objects as go

      p = "1" if plate_choice == "FP1" else "2"
      axes = [
          ("X (M-L)", f"F{p}X", "#ef4444"),
          ("Y (A-P)", f"F{p}Y", "#22c55e"),
          ("Z (Vertical)", f"F{p}Z", "#3b82f6"),
      ]

      fig_grf = go.Figure()
      for label, ch, color in axes:
        # Initial Raw/Unfiltered trace
        fig_grf.add_trace(
            go.Scatter(
                x=cycle_fp_raw.time,
                y=cycle_fp_raw.data[ch],
                mode="lines",
                line=dict(
                    color=color,
                    dash="dot" if cycle_fp_filt is not None else "solid",
                    width=1.5 if cycle_fp_filt is not None else 2.5,
                ),
                opacity=0.5 if cycle_fp_filt is not None else 1.0,
                name=f"{label} Raw{' (Debiased)' if debias_choice else ''}",
            )
        )
        # Filtered trace overlay
        if cycle_fp_filt is not None:
          fig_grf.add_trace(
              go.Scatter(
                  x=cycle_fp_filt.time,
                  y=cycle_fp_filt.data[ch],
                  mode="lines",
                  line=dict(color=color, width=2.5),
                  name=f"{label} Filtered ({cutoff_fc} Hz)",
              )
          )

      status_text = (
          f"Zeroed (Debiased) | Filter: {cutoff_fc} Hz"
          if debias_choice and cycle_fp_filt is not None
          else (
              "Raw (Non-Zeroed) | Filtered"
              if cycle_fp_filt is not None
              else "Raw Unfiltered"
          )
      )
      fig_grf.update_layout(
          title=f"{plate_choice} Force Traces — [{status_text}]",
          xaxis_title="Time (s)",
          yaxis_title="Force (N)",
          template="plotly_dark",
          hovermode="x unified",
          margin=dict(l=20, r=20, t=40, b=20),
      )
      st.plotly_chart(fig_grf, use_container_width=True)

      # Lock-in button to progress to Step 3
      st.markdown("#### 🎯 Confirm Decisions")
      if st.button("Accept Force Processing Decisions", type="primary"):
        st.session_state["chosen_plate"] = plate_choice
        st.session_state["apply_debias"] = debias_choice
        st.session_state["chosen_filter"] = filter_mode
        st.session_state["chosen_fc"] = cutoff_fc if cutoff_fc else 100
        # Store processed TimeSeries for downstream COP calculation
        st.session_state["cycle_fp_processed"] = (
            cycle_fp_filt if cycle_fp_filt is not None else cycle_fp_raw
        )
        st.session_state["filter_locked"] = True
        st.success("Decisions saved! Proceeding to Step 3.")
        st.rerun()
        
  # =========================================================================
  # STEP 3: CENTER OF PRESSURE (COP) & PLANAR BUTTERFLY PLOT
  # =========================================================================
        if st.session_state["filter_locked"]:
          with active_tabs[2]:
            st.subheader("Step 3: Center of Pressure (COP) Analysis")
            fc_val = st.session_state.get('chosen_fc', 100)
            filt_name = st.session_state.get('chosen_filter', 'None (Raw)')
            filt_suffix = f" ({fc_val} Hz)" if filt_name == "Butterworth Low-pass" else ""
            debias_status = "Zeroed (Debiased)" if st.session_state.get('apply_debias', False) else "Raw (Non-Zeroed)"

            st.caption(
                f"Calculated directly from your confirmed parameters: **{st.session_state['chosen_plate']}** | "
                f"Baseline: **{debias_status}** | "
                f"Filter: **{filt_name}{filt_suffix}**"
            )
      
            # Extract the confirmed processed TimeSeries from Step 2
            final_fp = st.session_state["cycle_fp_processed"]
            plate_prefix = "1" if st.session_state["chosen_plate"] == "FP1" else "2"
      
            fz_key = f"F{plate_prefix}Z"
            mx_key = f"M{plate_prefix}X"
            my_key = f"M{plate_prefix}Y"
      
            fz = final_fp.data[fz_key]
            mx = final_fp.data[mx_key]
            my = final_fp.data[my_key]
      
            # Controls for threshold gate and display options
            cop_ctrl1, cop_ctrl2 = st.columns([2, 2])
            with cop_ctrl1:
              fz_threshold = st.slider(
                  "Vertical Force Gate (|Fz| ≥ N):",
                  min_value=10.0,
                  max_value=200.0,
                  value=50.0,
                  step=5.0,
                  help=(
                      "Masks calculations when the foot is off the plate to prevent"
                      " division-by-zero asymptotes during swing phase."
                  ),
              )
            with cop_ctrl2:
              show_arrows = st.checkbox(
                  "Show Direction of Progression Markers", value=True
              )
      
            # Standard Biomechanical COP Equations:
            # COPx = -My / Fz  (Medio-Lateral)
            # COPy = +Mx / Fz  (Antero-Posterior)
            EPSILON = 1e-6
            contact_mask = np.abs(fz) >= fz_threshold
      
            cop_x = np.where(contact_mask, -my / (fz + EPSILON), np.nan)
            cop_y = np.where(contact_mask, mx / (fz + EPSILON), np.nan)
      
            import plotly.graph_objects as go
      
            col_butterfly, col_timeseries = st.columns(2)
      
            # --- Left Panel: 2D Planar Butterfly Plot (COPx vs COPy) ---
            with col_butterfly:
              valid_indices = np.where(contact_mask)[0]
              fig_butterfly = go.Figure()
      
              if len(valid_indices) == 0:
                st.warning(
                    f"No frames found where |Fz| ≥ {fz_threshold} N. Lower the"
                    " vertical force gate slider above."
                )
              else:
                x_valid = cop_x[valid_indices]
                y_valid = cop_y[valid_indices]
                t_valid = final_fp.time[valid_indices]
      
                # Continuous line colored by progression of time
                fig_butterfly.add_trace(
                    go.Scatter(
                        x=x_valid,
                        y=y_valid,
                        mode="lines+markers" if show_arrows else "lines",
                        line=dict(color="rgba(255, 255, 255, 0.7)", width=2),
                        marker=dict(
                            size=5 if show_arrows else 2,
                            color=t_valid,
                            colorscale="Viridis",
                            showscale=True,
                            colorbar=dict(title="Cycle Time (s)", thickness=12),
                        ),
                        hovertext=[
                            f"Time: {t:.3f}s<br>COPx: {x:.3f}m<br>COPy: {y:.3f}m"
                            for t, x, y in zip(t_valid, x_valid, y_valid)
                        ],
                        hoverinfo="text",
                        name="COP Path",
                    )
                )
      
                # Add initial contact (Heel Strike) and push-off (Toe Off) markers
                fig_butterfly.add_trace(
                    go.Scatter(
                        x=[x_valid[0]],
                        y=[y_valid[0]],
                        mode="markers+text",
                        marker=dict(size=12, color="#22c55e", symbol="star"),
                        text=["Start (Initial Contact)"],
                        textposition="top center",
                        name="Initial Contact",
                    )
                )
                fig_butterfly.add_trace(
                    go.Scatter(
                        x=[x_valid[-1]],
                        y=[y_valid[-1]],
                        mode="markers+text",
                        marker=dict(size=12, color="#ef4444", symbol="x"),
                        text=["End (Toe Off)"],
                        textposition="bottom center",
                        name="Toe Off",
                    )
                )
      
              fig_butterfly.update_layout(
                  title="Planar Butterfly Path (COPx vs. COPy)",
                  xaxis_title="Medio-Lateral Displacement COPx (m)",
                  yaxis_title="Antero-Posterior Displacement COPy (m)",
                  template="plotly_dark",
                  yaxis=dict(scaleanchor="x", scaleratio=1),  # Preserve true aspect ratio
                  margin=dict(l=20, r=20, t=40, b=20),
                  legend=dict(
                      orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                  ),
              )
              st.plotly_chart(fig_butterfly, use_container_width=True)
      
            # --- Right Panel: COP Coordinates vs Time ---
            with col_timeseries:
              fig_ts = go.Figure()
              fig_ts.add_trace(
                  go.Scatter(
                      x=final_fp.time,
                      y=cop_x,
                      mode="lines",
                      line=dict(color="#3b82f6", width=2),
                      name="COPx (Medio-Lateral)",
                  )
              )
              fig_ts.add_trace(
                  go.Scatter(
                      x=final_fp.time,
                      y=cop_y,
                      mode="lines",
                      line=dict(color="#ef4444", width=2),
                      name="COPy (Antero-Posterior)",
                  )
              )
      
              fig_ts.update_layout(
                  title="COP Displacement Components vs. Time",
                  xaxis_title="Time (s)",
                  yaxis_title="Displacement (m)",
                  template="plotly_dark",
                  hovermode="x unified",
                  margin=dict(l=20, r=20, t=40, b=20),
                  legend=dict(
                      orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                  ),
              )
              st.plotly_chart(fig_ts, use_container_width=True)
