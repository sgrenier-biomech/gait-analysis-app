#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 15:10:00 2026

@author: sgrenier
"""
import copy
import io
import json
import os
from pathlib import Path
import tempfile
from inverse_dynamics_final2 import compute_inverse_dynamics
import kineticstoolkit.lab as ktk
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R
import streamlit as st
import streamlit.components.v1 as components

# from COP_final2 import FP1_filtered, FP2_filtered


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
  c3d_data = ktk.read_c3d(str(c3d_file_path))
  markers = c3d_data["Points"]
  force = c3d_data["Analogs"]

  # Sampling frequency
  ForceSF = 1200  # Hz
  num_samples = len(
      force.data["F1X"]
  )  # Assuming all channels have the same length
  time = np.arange(
      0, num_samples / ForceSF, 1 / ForceSF
  )  # Creates time values at 1200Hz
  force.data["Time"] = time

  # Scale all numerical data in the TimeSeries by -1000 to Newtons
  force_to_scale = ["F1X", "F1Y", "F1Z", "F2X", "F2Y", "F2Z"]
  force.data = {
      key: (value * -1000 if key in force_to_scale else value)
      for key, value in force.data.items()
  }

  # Extract only the selected channels as a new TimeSeries
  FP1 = force.get_subset(["F1X", "F1Y", "F1Z", "M1X", "M1Y", "M1Z"])
  FP1_bias = FP1.get_ts_between_times(6.6, 6.9, inclusive=False)

  FP2 = force.get_subset(["F2X", "F2Y", "F2Z", "M2X", "M2Y", "M2Z"])
  FP2_bias = FP2.get_ts_between_times(14.0, 14.25, inclusive=False)

  # De-bias each channel in the force data Force plate 1
  for channel in FP1.data.keys():
    FP1.data[channel] -= np.mean(FP1_bias.data[channel])

  # De-bias each channel in the force data Force plate 2
  for channel in FP2.data.keys():
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
  times = markers.time[::step].tolist()
  n_frames = len(times)

  all_pts = []
  for m in markers.data.values():
    all_pts.append(m[::step, :3])
  stacked = np.concatenate(all_pts, axis=0)
  scale = 0.001 if float(np.nanmax(np.abs(stacked))) > 50.0 else 1.0

  center_x = float(np.nanmean(stacked[:, 0]) * scale)
  center_y = float(np.nanmean(stacked[:, 1]) * scale)
  center_z = float(np.nanmean(stacked[:, 2]) * scale)

  segments = []
  for group_name, info in interconnections.items():
    color = info.get("Color", "#00ffff")
    for link in info["Links"]:
      segments.append({"color": color, "markers": link})

  marker_dict = {}
  for m_name, m_data in markers.data.items():
    downsampled = m_data[::step, :3] * scale
    cleaned = np.where(np.isnan(downsampled), None, np.round(downsampled, 4))
    marker_dict[m_name] = cleaned.tolist()

  data_payload = json.dumps({
      "times": times,
      "n_frames": n_frames,
      "segments": segments,
      "markers": marker_dict,
      "center": [center_x, center_z, -center_y],
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

    const grid = new THREE.GridHelper(3.0, 15, 0x334155, 0x1e293b);
    grid.position.set(cx, 0, cz);
    scene.add(grid);
    scene.add(new THREE.AmbientLight(0xffffff, 1.0));

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

    window.setPreset = function(type) {{
      const dist = 2.4;
      if (type === 'side') camera.position.set(cx + dist, cy, cz);
      else if (type === 'front') camera.position.set(cx, cy, cz + dist);
      else if (type === 'top') camera.position.set(cx + 0.001, cy + dist, cz);
      else if (type === 'iso') camera.position.set(cx + 1.6, cy + 1.0, cz + 1.6);
      controls.target.set(cx, cy, cz);
    }};

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


def run_segment_kinematics(
    c3d_file_path: str, mass_total: float, height_total: float
):
  markers = ktk.read_c3d(c3d_file_path)["Points"]
  markers = ktk.filters.butter(markers, fc=6)

  raw_analogs = ktk.read_c3d(c3d_file_path)["Analogs"]

  t0_kinematics = float(markers.time[0])
  t0_analog = float(raw_analogs.time[0])
  time_offset = t0_kinematics - t0_analog

  fplate = ktk.read_c3d(c3d_file_path)["ForcePlatforms"]
  fplate.resample(120, kind="linear", in_place=True)
  fplate = ktk.filters.median(fplate, window_length=5)

  frames = ktk.TimeSeries(time=markers.time)
  joint_positions = ktk.TimeSeries(time=markers.time)

  pelvis_origin = markers.data["SACR"]
  y_vec = (0.5 * (markers.data["RASI"] + markers.data["LASI"])) - pelvis_origin
  xy_vec = markers.data["RASI"] - pelvis_origin
  frames.data["Pelvis"] = ktk.geometry.create_transform_series(
      positions=pelvis_origin, y=y_vec, xy=xy_vec
  )
  joint_positions.data["Pelvis"] = pelvis_origin

  LtHip_Origin = 0.5 * (markers.data["LGTR"] + markers.data["LASI"])
  z_vec_lt = LtHip_Origin - (0.5 * (markers.data["LLEP"] + markers.data["LMEP"]))
  xz_vec_lt = markers.data["LMEP"] - markers.data["LLEP"]
  frames.data["ThighL"] = ktk.geometry.create_transform_series(
      positions=LtHip_Origin, z=z_vec_lt, xz=xz_vec_lt
  )
  joint_positions.data["HipL"] = LtHip_Origin

  RtHip_Origin = 0.5 * (markers.data["RGTR"] + markers.data["RASI"])
  z_vec_rt = RtHip_Origin - (0.5 * (markers.data["RLEP"] + markers.data["RMEP"]))
  xz_vec_rt = markers.data["RLEP"] - markers.data["RMEP"]
  frames.data["ThighR"] = ktk.geometry.create_transform_series(
      positions=RtHip_Origin, z=z_vec_rt, xz=xz_vec_rt
  )
  joint_positions.data["HipR"] = RtHip_Origin

  LtShk_origin = 0.5 * (markers.data["LLEP"] + markers.data["LMEP"])
  z_vec_lshk = LtShk_origin - 0.5 * (
      markers.data["LMML"] + markers.data["LLML"]
  )
  yz_vec_lshk = markers.data["LSH2"] - markers.data["LSH3"]
  frames.data["ShankL"] = ktk.geometry.create_transform_series(
      positions=LtShk_origin, z=z_vec_lshk, yz=yz_vec_lshk
  )
  joint_positions.data["KneeL"] = LtShk_origin

  RtShk_origin = 0.5 * (markers.data["RLEP"] + markers.data["RMEP"])
  z_vec_rshk = RtShk_origin - 0.5 * (
      markers.data["RMML"] + markers.data["RLML"]
  )
  yz_vec_rshk = markers.data["RSH3"] - markers.data["RSH2"]
  frames.data["ShankR"] = ktk.geometry.create_transform_series(
      positions=RtShk_origin, z=z_vec_rshk, yz=yz_vec_rshk
  )
  joint_positions.data["KneeR"] = RtShk_origin

  LtFoot_origin = 0.5 * (markers.data["LMML"] + markers.data["LLML"])
  y_vec_lft = markers.data["L5TH"] - LtFoot_origin
  xy_vec_lft = markers.data["LMML"] - LtFoot_origin
  frames.data["FootL"] = ktk.geometry.create_transform_series(
      positions=LtFoot_origin, y=y_vec_lft, xy=xy_vec_lft
  )
  joint_positions.data["AnkleL"] = LtFoot_origin

  RtFoot_origin = 0.5 * (markers.data["RMML"] + markers.data["RLML"])
  y_vec_rft = markers.data["R5TH"] - RtFoot_origin
  xy_vec_rft = markers.data["RLML"] - RtFoot_origin
  frames.data["FootR"] = ktk.geometry.create_transform_series(
      positions=RtFoot_origin, y=y_vec_rft, xy=xy_vec_rft
  )
  joint_positions.data["AnkleR"] = RtFoot_origin

  Pelvis_to_ThighL_LCS = ktk.geometry.get_local_coordinates(
      frames.data["ThighL"], frames.data["Pelvis"]
  )
  Pelvis_to_ThighR_LCS = ktk.geometry.get_local_coordinates(
      frames.data["ThighR"], frames.data["Pelvis"]
  )
  ThighL_to_ShankL_LCS = ktk.geometry.get_local_coordinates(
      frames.data["ShankL"], frames.data["ThighL"]
  )
  ThighR_to_ShankR_LCS = ktk.geometry.get_local_coordinates(
      frames.data["ShankR"], frames.data["ThighR"]
  )
  ShankL_to_FootL_LCS = ktk.geometry.get_local_coordinates(
      frames.data["FootL"], frames.data["ShankL"]
  )
  ShankR_to_FootR_LCS = ktk.geometry.get_local_coordinates(
      frames.data["FootR"], frames.data["ShankR"]
  )

  angles = ktk.TimeSeries(time=markers.time)
  angles.data["HipL"] = ktk.geometry.get_angles(
      Pelvis_to_ThighL_LCS, "XZY", degrees=True
  )
  angles.data["HipR"] = ktk.geometry.get_angles(
      Pelvis_to_ThighR_LCS, "XZY", degrees=True
  )
  angles.data["KneeL"] = ktk.geometry.get_angles(
      ThighL_to_ShankL_LCS, "XZY", degrees=True
  )
  angles.data["KneeR"] = ktk.geometry.get_angles(
      ThighR_to_ShankR_LCS, "XZY", degrees=True
  )
  angles.data["AnkleL"] = ktk.geometry.get_angles(
      ShankL_to_FootL_LCS, "XYZ", degrees=True
  )
  angles.data["AnkleR"] = ktk.geometry.get_angles(
      ShankR_to_FootR_LCS, "XYZ", degrees=True
  )

  com_positions = ktk.TimeSeries(time=markers.time)
  com_positions.data["Pelvis_CoM"] = LtHip_Origin + 0.5 * (
      RtHip_Origin - LtHip_Origin
  )
  com_positions.data["ThighL_CoM"] = LtHip_Origin + (
      0.433 * (LtShk_origin - LtHip_Origin)
  )
  com_positions.data["ThighR_CoM"] = RtHip_Origin - (
      0.433 * (RtShk_origin - RtHip_Origin)
  )
  com_positions.data["ShankL_CoM"] = LtShk_origin + (
      0.433 * (LtFoot_origin - LtShk_origin)
  )
  com_positions.data["ShankR_CoM"] = RtShk_origin + (
      0.433 * (RtFoot_origin - RtShk_origin)
  )
  com_positions.data["FootL_CoM"] = LtFoot_origin + (
      0.5 * (markers.data["L5TH"] - LtFoot_origin)
  )
  com_positions.data["FootR_CoM"] = RtFoot_origin + (
      0.5 * (markers.data["R5TH"] - RtFoot_origin)
  )

  com_velocities = ktk.filters.deriv(com_positions)
  com_accelerations = ktk.filters.deriv(com_velocities)

  omega_raw = ktk.filters.deriv(angles)
  omega = transform_to_omega(angles, omega_raw)
  omega_filt = ktk.filters.butter(omega, fc=5)
  alpha = ktk.filters.deriv(omega_filt)

  FP1, FP2, FP1_filtered, FP2_filtered = process_cop(c3d_file_path)
  fp1 = FP1_filtered.copy()
  fp2 = FP2_filtered.copy()
  fp1.resample(120, kind="linear", in_place=True)
  fp2.resample(120, kind="linear", in_place=True)

  # Shift time vectors so ALL analog/force objects match marker time
  t0_target = float(markers.time[0])
  for ts_obj in [raw_analogs, fplate, FP1, fp1, FP2, fp2]:
    if ts_obj is not None:
      shift = t0_target - float(ts_obj.time[0])
      if abs(shift) > 1e-4:
        ts_obj.time = ts_obj.time + shift

  results = compute_inverse_dynamics(
      omega,
      alpha,
      com_accelerations,
      joint_positions,
      com_positions,
      fplate,
      mass_total,
      height_total,
      FP1_filtered=fp1,
      FP2_filtered=fp2,
  )

  # Re-verify alignment right before returning
  for ts_obj in [raw_analogs, fplate, FP1, fp1, FP2, fp2]:
    if ts_obj is not None:
      shift = t0_target - float(ts_obj.time[0])
      if abs(shift) > 1e-4:
        ts_obj.time = ts_obj.time + shift

  return markers, angles, results, FP1, fp1, FP2, fp2, raw_analogs


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
    st.session_state["c3d_file_path"] = selected_file_path

if st.button("Process Data", type="primary"):
  if not selected_file_path:
    st.error("Select or upload a .c3d file first.")
  else:
    with st.spinner("Processing file..."):
      try:
        (
            markers,
            angles,
            results,
            FP1_raw,
            fp1_filt,
            FP2_raw,
            fp2_filt,
            raw_analogs,
        ) = run_segment_kinematics(selected_file_path, mass, height)

        st.session_state["markers"] = markers
        st.session_state["angles"] = angles
        st.session_state["results"] = results

        # Store both raw and default-filtered versions
        st.session_state["FP1_raw"] = FP1_raw
        st.session_state["FP2_raw"] = FP2_raw
        st.session_state["FP1_default_filt"] = fp1_filt
        st.session_state["FP2_default_filt"] = fp2_filt
        st.session_state["raw_analogs"] = raw_analogs

        # Workflow and decision states
        st.session_state["cycle_locked"] = False
        st.session_state["filter_locked"] = False
        st.session_state["t_start"] = float(angles.time[0])
        st.session_state["t_end"] = float(
            min(angles.time[0] + 1.2, angles.time[-1])
        )
        st.session_state["chosen_fc"] = 100
        st.session_state["apply_debias"] = False
        st.session_state["step1_version"] = 0

        st.rerun()

      except Exception as e:
        st.error(f"Error: {e}")

# Initialize persistent student workflow states if not already set
for key, default in [
    ("cycle_locked", False),
    ("filter_locked", False),
    ("t_start", 0.0),
    ("t_end", 2.0),
    ("chosen_plate", "FP1"),
    ("chosen_filter", "None (Raw)"),
    ("chosen_fc", 100),
    ("step1_version", 0),
]:
  if key not in st.session_state:
    st.session_state[key] = default

# Render tabs when data is present in session state
if "angles" in st.session_state and "FP1_raw" in st.session_state:
  st.success("Analysis completed! Begin with Assignment 1 below.")
  st.markdown("---")

# Define dynamic tabs that unlock sequentially
  tab_labels = [
      "3D animation"
      "Assignment 1: Raw Signals",
      "Assignment 2: CoM & GRF",
      "Assignment 3: 6-DOF GRF Analysis",
      "Assignment 4: COP",
      "placeholder1",
      "placeholder2"
  ]
  
  if st.session_state.get("cycle_locked", False):
    tab_labels.append("Step 2: GRF Decisions")
  if st.session_state.get("filter_locked", False):
    tab_labels.extend(
        ["Step 3: COP Analysis", "4. Joint Kinetics", "5. 3D Animation"]
    )

  active_tabs = st.tabs(tab_labels)
  
  # =========================================================================
  # 3D animation Tab
  # =========================================================================


  with active_tabs[0]:
    st.subheader("3D Gait Animation")

    markers_ts = st.session_state.get("markers")

    if markers_ts is not None and hasattr(markers_ts, "data"):
        import numpy as np
        import plotly.graph_objects as go

        # 1. Anatomical interconnections definition
        interconnections = dict()

        interconnections["Pelvis"] = {
            "Color": (1, 0.5, 1),
            "Links": [
                ["SACR", "LASI", "RASI", "SACR"],
                ["LASI", "LGTR", "RGTR", "RASI"],
            ],
        }
        interconnections["Left Leg"] = {
            "Color": (1, 0.5, 0),
            "Links": [
                ["LGTR", "LLEP", "LLML"],
                ["LLML", "LCAL", "L5TH", "LLML"],
            ],
        }
        interconnections["Right Leg"] = {
            "Color": (0, 0.5, 1),
            "Links": [
                ["RGTR", "RLEP", "RLML"],
                ["RLML", "RCAL", "R5TH", "RLML"],
            ],
        }

        # Case-insensitive lookup for marker labels
        available_keys = list(markers_ts.data.keys())
        key_lookup = {k.upper(): k for k in available_keys}

        # 2. Downsample frames for smooth browser animation playback
        total_frames = len(markers_ts.time)
        step = max(1, total_frames // 200)  # ~120 keyframes for fluid streaming
        frame_indices = list(range(0, total_frames, step))

        # Calculate bounding box to keep 3D aspect ratio locked during playback
        all_pts = []
        for segment in interconnections.values():
            for link in segment["Links"]:
                for m in link:
                    m_key = key_lookup.get(m.upper())
                    if m_key and m_key in markers_ts.data:
                        all_pts.append(markers_ts.data[m_key][frame_indices, :3])

        if all_pts:
            pts_concat = np.concatenate(all_pts, axis=0)
            valid = pts_concat[~np.isnan(pts_concat).any(axis=1)]
            if len(valid) > 0:
                x_min, x_max = float(valid[:, 0].min()), float(valid[:, 0].max())
                y_min, y_max = float(valid[:, 1].min()), float(valid[:, 1].max())
                z_min, z_max = float(valid[:, 2].min()), float(valid[:, 2].max())
            else:
                x_min, x_max, y_min, y_max, z_min, z_max = -0.5, 0.5, -0.5, 0.5, 0, 1.5
        else:
            x_min, x_max, y_min, y_max, z_min, z_max = -0.5, 0.5, -0.5, 0.5, 0, 1.5

        def build_frame_data(f_idx):
            data_traces = []

            # Trace 0: Marker nodes
            mx, my, mz, mlabels = [], [], [], []
            for m_upper, actual_key in key_lookup.items():
                pt = markers_ts.data[actual_key][f_idx]
                if not np.isnan(pt[:3]).any():
                    mx.append(pt[0])
                    my.append(pt[1])
                    mz.append(pt[2])
                    mlabels.append(actual_key)

            data_traces.append(
                go.Scatter3d(
                    x=mx, y=my, z=mz,
                    mode="markers",
                    marker=dict(size=3.5, color="#e2e8f0"),
                    text=mlabels,
                    hoverinfo="text",
                    name="Markers"
                )
            )

            # Traces 1+: One line trace per interconnected anatomical group
            for group_name, group_info in interconnections.items():
                r, g, b = [int(c * 255) for c in group_info["Color"]]
                color_str = f"rgb({r},{g},{b})"

                lx, ly, lz = [], [], []
                for link in group_info["Links"]:
                    for i in range(len(link) - 1):
                        m1 = key_lookup.get(link[i].upper())
                        m2 = key_lookup.get(link[i + 1].upper())
                        if m1 and m2 and m1 in markers_ts.data and m2 in markers_ts.data:
                            p1 = markers_ts.data[m1][f_idx]
                            p2 = markers_ts.data[m2][f_idx]
                            if not np.isnan(p1[:3]).any() and not np.isnan(p2[:3]).any():
                                lx.extend([p1[0], p2[0], None])
                                ly.extend([p1[1], p2[1], None])
                                lz.extend([p1[2], p2[2], None])

                data_traces.append(
                    go.Scatter3d(
                        x=lx, y=ly, z=lz,
                        mode="lines",
                        line=dict(color=color_str, width=5),
                        name=group_name,
                        hoverinfo="none"
                    )
                )

            return data_traces

        # Initial baseline frame
        initial_traces = build_frame_data(frame_indices[0])

        # Animation keyframes
        frames = [
            go.Frame(data=build_frame_data(idx), name=f"f_{idx}")
            for idx in frame_indices
        ]

        # Slider scrubber
        sliders = [{
            "steps": [
                {
                    "method": "animate",
                    "args": [[f.name], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}],
                    "label": f"{markers_ts.time[idx]:.2f}s"
                }
                for idx, f in zip(frame_indices, frames)
            ],
            "currentvalue": {"prefix": "Time: ", "visible": True},
            "pad": {"t": 30}
        }]

        fig_stick = go.Figure(data=initial_traces, frames=frames)

        fig_stick.update_layout(
            scene=dict(
                xaxis=dict(range=[x_min, x_max], title="X (Mediolateral)"),
                yaxis=dict(range=[y_min, y_max], title="Y (Anteroposterior)"),
                zaxis=dict(range=[z_min, z_max], title="Z (Vertical)"),
                aspectmode="data"
            ),
            updatemenus=[{
                "type": "buttons",
                "showactive": False,
                "x": 0.05,
                "y": 1.15,
                "buttons": [
                    {
                        "label": "▶ Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 30, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}]
                    },
                    {
                        "label": "⏸ Pause",
                        "method": "animate",
                        "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}]
                    }
                ]
            }],
            sliders=sliders,
            height=620,
            margin=dict(l=0, r=0, t=30, b=0)
        )

        st.plotly_chart(fig_stick, use_container_width=True)

    else:
        st.info("Upload and process a .c3d file first to preview the 3D stick-figure animation.")
        
  # =========================================================================
  # ASSIGNMENT 1: RAW SIGNALS (POINTS & ANALOGS)
  # =========================================================================
  with active_tabs[1]:
        # --- Assignment 1: Raw Signals ---
        st.subheader("Assignment 1: Raw Signals")

        # 1. Retrieve data structures from session_state
        markers_dict = st.session_state.get("markers")
        analogs_ts = st.session_state.get("raw_analogs")
        available_analogs = (
            list(analogs_ts.data.keys())
            if (analogs_ts is not None and hasattr(analogs_ts, "data"))
            else []
        )

        import plotly.graph_objects as go

        # --- Part A: Raw Markers ---
        if markers_dict is not None and hasattr(markers_dict, "data"):
            marker_names = list(markers_dict.data.keys())

            selected_marker = st.selectbox(
                "Select a Marker to Inspect:",
                options=marker_names,
                index=None,
                placeholder="Choose a marker...",
                key="asgt1_marker_select",
            )

            if selected_marker is None:
                st.info("Please select a marker above to inspect raw marker trajectories.")
            else:
                marker_data = markers_dict.data[selected_marker]
                time_series = markers_dict.time

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=time_series,
                        y=marker_data[:, 0],
                        mode="lines",
                        name=f"{selected_marker} X",
                        line=dict(color="#ef4444", width=1.5),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=time_series,
                        y=marker_data[:, 1],
                        mode="lines",
                        name=f"{selected_marker} Y",
                        line=dict(color="#10b981", width=1.5),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=time_series,
                        y=marker_data[:, 2],
                        mode="lines",
                        name=f"{selected_marker} Z",
                        line=dict(color="#3b82f6", width=1.5),
                    )
                )

                fig.update_layout(
                    title=f"Raw 3D Trajectory: {selected_marker}",
                    xaxis_title="Time (s)",
                    yaxis_title="Position (m or mm)",
                    legend_title="Coordinate",
                    hovermode="x unified",
                    height=450,
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No marker data available. Please process a .c3d file first.")

        with st.expander("💡 Assignment Helper: Part A Questions"):
            st.markdown(
                """
                * **What points have you graphed?** 
                  Identify the anatomical landmarks corresponding to these acronyms based on the paper's description.
                * **What does each line represent?** 
                  * **X (Red):** M-L (Mediolateral)
                  * **Y (Green):** A-P (Anteroposterior)
                  * **Z (Blue):** Vertical
                """
            )

        st.markdown("---")

        # --- Part B: Analogs ---
        st.markdown("#### Part B: Raw Vertical Force Component from `Analogs` (Not Points)")
        st.caption(
            "Zoom in so that only a single foot strike is displayed. Select the raw"
            " vertical analog channel to inspect before calibration or zeroing."
        )

        if available_analogs:
            sine_candidates = [k for k in available_analogs if "sin" in k.lower()]
            vertical_candidates = [
                k
                for k in available_analogs
                if "fz" in k.lower() or "f1z" in k.lower() or "force" in k.lower()
            ]

            if sine_candidates:
                default_channel = sine_candidates[0]
            elif vertical_candidates:
                default_channel = vertical_candidates[0]
            else:
                default_channel = available_analogs[0]

            def_analog_idx = (
                available_analogs.index(default_channel)
                if default_channel in available_analogs
                else 0
            )

            c_a1, c_a2 = st.columns([2, 1])
            with c_a1:
                chosen_analog = st.selectbox(
                    "Select Raw Analog Channel:",
                    available_analogs,
                    index=def_analog_idx,
                    key="as1_analog_ch",
                )
            with c_a2:
                st.caption(f"Analog channels recorded: {len(available_analogs)} total channels.")

            fig_as1_analog = go.Figure()
            if chosen_analog in analogs_ts.data:
                fig_as1_analog.add_trace(
                    go.Scatter(
                        x=analogs_ts.time,
                        y=analogs_ts.data[chosen_analog],
                        mode="lines",
                        line=dict(color="#f59e0b", width=1.5),
                        name=chosen_analog,
                    )
                )

            fig_as1_analog.update_layout(
                title=f"Raw Analog Signal: {chosen_analog}",
                xaxis_title="Time (s)",
                yaxis_title="Raw ADC Voltage / Bits (Uncalibrated)",
                template="plotly_dark",
                hovermode="x unified",
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_as1_analog, use_container_width=True)

            with st.expander("💡 Assignment Helper: Part B Questions"):
                st.markdown(
                    f"""
                    * **Explain what you have plotted:** 
                      You plotted `{chosen_analog}` directly from `c3d["Analogs"]`. This is the raw transducer electrical signal (voltage or digital counts) recorded directly from the force plate amplifier before calibration matrices, scale factors, or baseline offsets are applied.
                    * **Does it make sense?**
                      * Is there a steady non-zero baseline during unloaded intervals?
                      * Are there noticeable deflection spikes during foot strikes?
                      * Is high-frequency electrical noise visible along the baseline?
                    """
                )
        else:
            st.info("No raw analog channels found in the loaded trial.")

        st.markdown("---")
        
  # =========================================================================
  # ASSIGNMENT 2: CENTRE OF MASS & GROUND REACTION FORCES
  # =========================================================================
  with active_tabs[2]:
    st.subheader("Assignment 2: Centre of Mass & Ground Reaction Forces")
    st.markdown(
        """
        Complete the three deliverables below to inspect the estimated whole-body Centre of Mass (CoM) 
        and evaluate the calibrated Ground Reaction Force across the trial and zoomed in on a single foot strike.
        """
    )

    markers_ts = st.session_state["markers"]


# --- Part 1: Centre of Mass Trajectory (4 Lines) ---
    st.markdown("#### Part 1: Best Estimate of Centre of Mass (CoM)")
    st.caption(
        "Evaluate the available raw markers in your dataset and select which single marker "
        "provides the best surrogate estimate of whole-body CoM."
    )

    available_marker_list = list(markers_ts.data.keys())
    com_marker_options = ["None (Select a marker)"] + available_marker_list

    c_opt1, _ = st.columns([2, 1])
    with c_opt1:
      chosen_marker = st.selectbox(
          "Student Decision: Select Raw Marker as Best CoM Estimate:",
          com_marker_options,
          index=0,  # Defaults to "None (Select a marker)"
          key="as2_raw_com_marker",
      )

    fig_as2_com = go.Figure()

    if chosen_marker != "None (Select a marker)":
      # Extract raw marker trajectory directly (N x 4)
      selected_com = markers_ts.data[chosen_marker]

      # Ensure homogeneous coordinates (N, 4) with W = 1.0
      if selected_com.shape[-1] == 3:
        ones_col = np.ones((len(selected_com), 1))
        selected_com = np.hstack([selected_com, ones_col])
      elif selected_com.shape[-1] == 4:
        selected_com = np.copy(selected_com)
        selected_com[:, 3] = 1.0

      com_lines_info = [
          ("Line 1: X (Medio-Lateral)", "#ef4444", "solid"),
          ("Line 2: Y (Antero-Posterior)", "#22c55e", "solid"),
          ("Line 3: Z (Vertical)", "#3b82f6", "solid"),
          ("Line 4: Homogeneous Coordinate (W = 1.0)", "#a855f7", "dot"),
      ]

      for col_idx, (name, color, dash) in enumerate(com_lines_info):
        fig_as2_com.add_trace(
            go.Scatter(
                x=markers_ts.time,
                y=selected_com[:, col_idx],
                mode="lines",
                name=name,
                line=dict(color=color, dash=dash, width=2),
            )
        )

      fig_as2_com.update_layout(
          title=f"CoM Estimate: Raw Marker [{chosen_marker}] - 4-Line Trajectory",
          xaxis_title="Time (s)",
          yaxis_title="Position (m) / Homogeneous Unit",
          template="plotly_dark",
          hovermode="x unified",
          margin=dict(l=20, r=20, t=40, b=20),
      )
    else:
      fig_as2_com.update_layout(
          title="Centre of Mass - Please select a marker from above",
          xaxis_title="Time (s)",
          yaxis_title="Position (m)",
          template="plotly_dark",
          margin=dict(l=20, r=20, t=40, b=20),
      )

    st.plotly_chart(fig_as2_com, use_container_width=True)

    with st.expander("💡 Lab Question: Explain Your Selection & The 4 Lines"):
      st.markdown("""
        * **Why did you select this marker as your best estimate?**
          * Reflect on where the whole-body center of mass lies during upright human locomotion.
          * Compare how a single surface marker behaves relative to an average midpoint of two anatomical landmarks.
        * **What does each line on your plot represent?**
          * **Line 1 (X, Red):**M-L
          * **Line 2 (Y, Green):** A-P 
          * **Line 3 (Z, Blue):** Vertical
          * **Line 4 (Purple, Dot):** Constant flat line at **1.0**. In Kineticstoolkit, point positions are stored as homogeneous 4-element vectors $[x, y, z, 1]^T$ to allow standard $4 \\times 4$ transformation matrices to handle translations and rotations.
        """)


    # --- Part 2: Full-Trial Ground Reaction Force ---
    st.markdown("#### Part 2: Full-Trial Ground Reaction Forces")
    st.caption("Select a force plate to view its calibrated forces in Newtons.")

    col_fp_choice, _ = st.columns([1, 2])
    with col_fp_choice:
      as2_plate = st.radio(
          "Select Force Plate to Inspect:",
          ["FP1", "FP2"],
          horizontal=True,
          key="as2_fp_choice_radio",
      )

    raw_fp_ts = (
        st.session_state["FP1_raw"]
        if as2_plate == "FP1"
        else st.session_state["FP2_raw"]
    )
    p_num = "1" if as2_plate == "FP1" else "2"

    grf_channel_meta = [
        (f"F{p_num}X", f"F{p_num}X", "#ef4444"),
        (
            f"F{p_num}Y",
            f"F{p_num}Y",
            "#22c55e",
        ),
        (f"F{p_num}Z", f"F{p_num}Z", "#3b82f6"),
    ]

    fig_full_grf = go.Figure()
    for label, key, color in grf_channel_meta:
      if key in raw_fp_ts.data:
        fig_full_grf.add_trace(
            go.Scatter(
                x=raw_fp_ts.time,
                y=raw_fp_ts.data[key],
                mode="lines",
                name=label,
                line=dict(color=color, width=1.8),
            )
        )

    fig_full_grf.update_layout(
        title=f"Full Trial Ground Reaction Force: {as2_plate}",
        xaxis_title="Time (s)",
        yaxis_title="Force (N)",
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig_full_grf, use_container_width=True)

    with st.expander("💡 Lab Question: What Have You Plotted in Part 2?"):
      st.markdown(
          f"""
            * **Explanation:** You have plotted the calibrated tri-axial Ground Reaction Force components. Explain what each component is.
            * **Signal Behavior:** The signal hovers near 0 N when no subject is on the plate and displays prominent deflections during the foot contact phase.
            """
      )

    st.markdown("---")

    # --- Part 3: Zoomed-In Single Foot Strike ---
    st.markdown("#### Part 3: Zoomed-In Single Foot Strike")
    st.caption("Adjust the window below to isolate a single stance phase.")

    t_fp_start = float(raw_fp_ts.time[0])
    t_fp_end = float(raw_fp_ts.time[-1])

    # Automatically identify stance bounds where |Fz| > 50 N
    fz_trace = (
        raw_fp_ts.data[f"F{p_num}Z"]
        if f"F{p_num}Z" in raw_fp_ts.data
        else np.array([])
    )
    contact_pts = np.where(np.abs(fz_trace) > 50.0)[0]
    if len(contact_pts) > 0:
      auto_start = float(raw_fp_ts.time[contact_pts[0]]) - 0.1
      auto_end = float(raw_fp_ts.time[contact_pts[-1]]) + 0.1
    else:
      auto_start = t_fp_start + 0.5
      auto_end = auto_start + 0.8

    col_z1, col_z2 = st.columns(2)
    with col_z1:
      strike_zoom_s = st.number_input(
          "Foot Strike Window Start (s):",
          min_value=t_fp_start,
          max_value=t_fp_end,
          value=round(max(t_fp_start, auto_start), 3),
          step=0.01,
          format="%.3f",
          key="as2_zoom_start",
      )
    with col_z2:
      strike_zoom_e = st.number_input(
          "Foot Strike Window End (s):",
          min_value=t_fp_start,
          max_value=t_fp_end,
          value=round(min(t_fp_end, auto_end), 3),
          step=0.01,
          format="%.3f",
          key="as2_zoom_end",
      )

    fig_zoom_grf = go.Figure()
    for label, key, color in grf_channel_meta:
      if key in raw_fp_ts.data:
        fig_zoom_grf.add_trace(
            go.Scatter(
                x=raw_fp_ts.time,
                y=raw_fp_ts.data[key],
                mode="lines",
                name=label,
                line=dict(color=color, width=2.5),
            )
        )

    fig_zoom_grf.update_layout(
        title=(
            f"Zoomed Single Foot Strike: {as2_plate} ({strike_zoom_s:.3f}s to"
            f" {strike_zoom_e:.3f}s)"
        ),
        xaxis_title="Time (s)",
        yaxis_title="Force (N)",
        template="plotly_dark",
        hovermode="x unified",
        xaxis=dict(range=[strike_zoom_s, strike_zoom_e], autorange=False),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig_zoom_grf, use_container_width=True)

    with st.expander("💡 Lab Question: Comparison & Noise Discussion"):
      st.markdown(
          """
            * **Comparison with Assignment 1:**
              * In Assignment 1, you plotted uncalibrated raw voltage/counts directly from `c3d["Analogs"]`.
              * In Assignment 2, calibration matrices and amplifier gain factors have transformed those electrical signals into calibrated forces in Newtons (N).
            * **Noise Content Analysis:**
              * **Baseline Noise:** The unloaded baseline has slight fluctuations (~5–10 N) compared to the raw ADC voltage.
              * **Impact Transients vs. Noise:** Notice the sharp, high-frequency oscillations during the initial 50 ms of heel strike (the heel impact transient). This is **not purely electronic noise**; it reflects physical mechanical shock propagation through the leg skeleton and the natural resonance/vibration frequency of the force plate mounting structure.
            """
      )

    st.markdown("---")

# =========================================================================
  # ASSIGNMENT 3: 6-COMPONENT GROUND REACTION FORCES & MOMENTS
  # =========================================================================
  with active_tabs[3]:
    st.subheader(
        "Assignment 3: 6-DOF Ground Reaction Force & Moment Processing"
    )
    st.markdown(
        """
        Follow the sequential steps below to prepare, calibrate, zero, and filter your force platform data. 
        You will then inspect all 6 kinetic components ($F_x, F_y, F_z, M_x, M_y, M_z$) across exactly two footstrikes.
        """
    )

    angles_ts = st.session_state["angles"]
    k_t_start = float(angles_ts.time[0])
    k_t_end = float(angles_ts.time[-1])
    kin_duration = k_t_end - k_t_start

    raw_fp1 = st.session_state["FP1_raw"]
    raw_fp2 = st.session_state["FP2_raw"]
    n_total_fp = len(raw_fp1.time)

    from plotly.subplots import make_subplots

    # -------------------------------------------------------------
    # State Initializations for Assignment 3
    # -------------------------------------------------------------
    for p_id, def_s, def_e in [
        ("FP1", k_t_start + 0.1, k_t_start + 0.4),
        ("FP2", k_t_start + 0.8, k_t_start + 1.1),
    ]:
      if f"as3_{p_id}_debias_applied" not in st.session_state:
        st.session_state[f"as3_{p_id}_debias_applied"] = False
      if (
          f"as3_{p_id}_base_s" not in st.session_state
          or st.session_state[f"as3_{p_id}_base_s"] < k_t_start
      ):
        st.session_state[f"as3_{p_id}_base_s"] = round(def_s, 3)
      if (
          f"as3_{p_id}_base_e" not in st.session_state
          or st.session_state[f"as3_{p_id}_base_e"]
          <= st.session_state[f"as3_{p_id}_base_s"]
      ):
        st.session_state[f"as3_{p_id}_base_e"] = round(def_e, 3)
      if f"as3_bs_{p_id}" not in st.session_state:
        st.session_state[f"as3_bs_{p_id}"] = float(
            st.session_state[f"as3_{p_id}_base_s"]
        )
      if f"as3_be_{p_id}" not in st.session_state:
        st.session_state[f"as3_be_{p_id}"] = float(
            st.session_state[f"as3_{p_id}_base_e"]
        )
      if f"as3_view_ver_{p_id}" not in st.session_state:
        st.session_state[f"as3_view_ver_{p_id}"] = 0

    if (
        "as3_win_s" not in st.session_state
        or st.session_state["as3_win_s"] < k_t_start
    ):
      st.session_state["as3_win_s"] = round(k_t_start + 0.5, 3)
    if (
        "as3_win_e" not in st.session_state
        or st.session_state["as3_win_e"] <= st.session_state["as3_win_s"]
    ):
      st.session_state["as3_win_e"] = round(
          min(k_t_end, st.session_state["as3_win_s"] + 1.8), 3
      )
    if "as3_num_t_s" not in st.session_state:
      st.session_state["as3_num_t_s"] = float(st.session_state["as3_win_s"])
    if "as3_num_t_e" not in st.session_state:
      st.session_state["as3_num_t_e"] = float(st.session_state["as3_win_e"])

    # =============================================================
    # STEP 1: UNIT SCALING & SIGN CONVENTION
    # =============================================================
    st.markdown("### Step 1: Unit Scaling & Sign Convention")
    st.caption(
        "Decide whether to convert raw analog units into SI units (Newtons and"
        " Newton-meters) and align polarities with laboratory axes."
    )

    c_sc1, c_sc2 = st.columns(2)
    with c_sc1:
      apply_scale = st.checkbox(
          "Scale & Invert Force Components (Upward/Forward +ve)",
          value=False,  # Unchecked by default
          help="Converts voltages to Newtons and aligns polarities.",
          key="as3_scale_cb",
      )
    with c_sc2:
      apply_moment_scale = st.checkbox(
          "Scale Moment Components (N·m Conversion)",
          value=False,  # Unchecked by default
          help="Converts raw moment signals into Newton-meters.",
          key="as3_mscale_cb",
      )

    st.markdown("---")

# =============================================================
    # STEP 2A: BASELINE ZEROING FOR FORCE PLATFORM 1
    # =============================================================
    st.markdown("### Step 2a: Baseline Zeroing (De-bias) — Force Platform 1")
    st.info(
        "👉 **Instruction for FP1:** While in **'Zoom View'**, drag a box over"
        " a quiet unloaded section to zoom in. Switch to **'Select Debias"
        " Range'** to highlight the exact baseline and click **'Apply Zeroing"
        " to FP1'**.",
        icon="ℹ️",
    )

    if "as3_FP1_zoom_range" not in st.session_state:
      st.session_state["as3_FP1_zoom_range"] = [k_t_start, k_t_end]

    v1 = st.session_state["as3_view_ver_FP1"]
    col_fp1_ctrl, col_fp1_rst = st.columns([3, 1])
    with col_fp1_ctrl:
      fp1_mode = st.radio(
          "FP1 Interaction Tool Mode:",
          ["Zoom View", "Select Debias Range"],
          horizontal=True,
          key=f"as3_fp1_mode_{v1}",
      )
    with col_fp1_rst:
      st.write("")
      if st.button("Reset FP1 Graph & Values", key="as3_fp1_rst_btn"):
        st.session_state["as3_FP1_zoom_range"] = [k_t_start, k_t_end]
        st.session_state["as3_FP1_base_s"] = round(k_t_start + 0.1, 3)
        st.session_state["as3_FP1_base_e"] = round(k_t_start + 0.4, 3)
        st.session_state.pop(f"as3_bs_FP1_{v1}", None)
        st.session_state.pop(f"as3_be_FP1_{v1}", None)
        st.session_state["as3_view_ver_FP1"] += 1
        st.rerun()

    v1 = st.session_state["as3_view_ver_FP1"]
    fig_fp1 = go.Figure()
    if "F1Z" in raw_fp1.data:
      fig_fp1.add_trace(
          go.Scatter(
              x=raw_fp1.time,
              y=raw_fp1.data["F1Z"],
              mode="lines",
              line=dict(color="#3b82f6", width=1.5),
              name="FP1 Fz",
          )
      )

    fp1_bs = float(st.session_state["as3_FP1_base_s"])
    fp1_be = float(st.session_state["as3_FP1_base_e"])
    fig_fp1.add_vrect(
        x0=fp1_bs,
        x1=fp1_be,
        fillcolor="rgba(6, 182, 212, 0.25)",
        line_width=2,
        line_dash="dot",
        line_color="#06b6d4",
        annotation_text="FP1 Baseline Range",
        annotation_position="top left",
    )

    fp1_curr_zoom = st.session_state["as3_FP1_zoom_range"]

    fig_fp1.update_layout(
        title="Force Platform 1: Vertical Force (F1Z)",
        template="plotly_dark",
        height=320,
        dragmode="select",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=35, b=20),
        xaxis=dict(
            title="Time (s)",
            range=fp1_curr_zoom,
            autorange=False,
        ),
        yaxis=dict(title="Force (N)"),
        uirevision=f"fp1_rev_{v1}",
    )

    chart_fp1_event = st.plotly_chart(
        fig_fp1,
        use_container_width=True,
        on_select="rerun",
        selection_mode=["box"],
        key=f"as3_chart_fp1_{v1}",
    )

    if chart_fp1_event and "selection" in chart_fp1_event:
      boxes = chart_fp1_event["selection"].get("box", [])
      if boxes and len(boxes) > 0 and "x" in boxes[0]:
        x_pts = boxes[0]["x"]
        n_s = round(float(min(x_pts)), 3)
        n_e = round(float(max(x_pts)), 3)

        if fp1_mode == "Zoom View":
          if (
              abs(n_s - fp1_curr_zoom[0]) > 0.005
              or abs(n_e - fp1_curr_zoom[1]) > 0.005
          ):
            st.session_state["as3_FP1_zoom_range"] = [n_s, n_e]
            st.rerun()
        else:
          if (
              abs(n_s - st.session_state["as3_FP1_base_s"]) > 0.005
              or abs(n_e - st.session_state["as3_FP1_base_e"]) > 0.005
          ):
            st.session_state["as3_FP1_base_s"] = n_s
            st.session_state["as3_FP1_base_e"] = n_e
            st.session_state[f"as3_bs_FP1_{v1}"] = n_s
            st.session_state[f"as3_be_FP1_{v1}"] = n_e
            st.rerun()

    c1_b1, c1_b2, c1_b3 = st.columns([1.5, 1.5, 1.5])
    with c1_b1:
      as3_b_start_fp1 = st.number_input(
          "FP1 Baseline Start (s):",
          min_value=k_t_start,
          max_value=k_t_end,
          value=float(st.session_state["as3_FP1_base_s"]),
          step=0.01,
          format="%.3f",
          key=f"as3_bs_FP1_{v1}",
      )
      st.session_state["as3_FP1_base_s"] = as3_b_start_fp1
    with c1_b2:
      as3_b_end_fp1 = st.number_input(
          "FP1 Baseline End (s):",
          min_value=k_t_start,
          max_value=k_t_end,
          value=float(st.session_state["as3_FP1_base_e"]),
          step=0.01,
          format="%.3f",
          key=f"as3_be_FP1_{v1}",
      )
      st.session_state["as3_FP1_base_e"] = as3_b_end_fp1
    with c1_b3:
      st.write("")
      st.write("")
      if st.button("Apply Zeroing to FP1", type="primary", key="as3_btn_app_fp1"):
        st.session_state["as3_FP1_debias_applied"] = True
        st.success("FP1 baseline offset zeroed!")
        st.rerun()

    st.markdown("---")

    # =============================================================
    # STEP 2B: BASELINE ZEROING FOR FORCE PLATFORM 2
    # =============================================================
    st.markdown("### Step 2b: Baseline Zeroing (De-bias) — Force Platform 2")
    st.info(
        "👉 **Instruction for FP2:** While in **'Zoom View'**, drag a box over"
        " a quiet unloaded section to zoom in. Switch to **'Select Debias"
        " Range'** to highlight the exact baseline and click **'Apply Zeroing"
        " to FP2'**.",
        icon="ℹ️",
    )

    if "as3_FP2_zoom_range" not in st.session_state:
      st.session_state["as3_FP2_zoom_range"] = [k_t_start, k_t_end]

    v2 = st.session_state["as3_view_ver_FP2"]
    col_fp2_ctrl, col_fp2_rst = st.columns([3, 1])
    with col_fp2_ctrl:
      fp2_mode = st.radio(
          "FP2 Interaction Tool Mode:",
          ["Zoom View", "Select Debias Range"],
          horizontal=True,
          key=f"as3_fp2_mode_{v2}",
      )
    with col_fp2_rst:
      st.write("")
      if st.button("Reset FP2 Graph & Values", key="as3_fp2_rst_btn"):
        st.session_state["as3_FP2_zoom_range"] = [k_t_start, k_t_end]
        st.session_state["as3_FP2_base_s"] = round(k_t_start + 0.8, 3)
        st.session_state["as3_FP2_base_e"] = round(k_t_start + 1.1, 3)
        st.session_state.pop(f"as3_bs_FP2_{v2}", None)
        st.session_state.pop(f"as3_be_FP2_{v2}", None)
        st.session_state["as3_view_ver_FP2"] += 1
        st.rerun()

    v2 = st.session_state["as3_view_ver_FP2"]
    fig_fp2 = go.Figure()
    if "F2Z" in raw_fp2.data:
      fig_fp2.add_trace(
          go.Scatter(
              x=raw_fp2.time,
              y=raw_fp2.data["F2Z"],
              mode="lines",
              line=dict(color="#22c55e", width=1.5),
              name="FP2 Fz",
          )
      )

    fp2_bs = float(st.session_state["as3_FP2_base_s"])
    fp2_be = float(st.session_state["as3_FP2_base_e"])
    fig_fp2.add_vrect(
        x0=fp2_bs,
        x1=fp2_be,
        fillcolor="rgba(217, 70, 239, 0.25)",
        line_width=2,
        line_dash="dot",
        line_color="#d946ef",
        annotation_text="FP2 Baseline Range",
        annotation_position="top left",
    )

    fp2_curr_zoom = st.session_state["as3_FP2_zoom_range"]

    fig_fp2.update_layout(
        title="Force Platform 2: Vertical Force (F2Z)",
        template="plotly_dark",
        height=320,
        dragmode="select",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=35, b=20),
        xaxis=dict(
            title="Time (s)",
            range=fp2_curr_zoom,
            autorange=False,
        ),
        yaxis=dict(title="Force (N)"),
        uirevision=f"fp2_rev_{v2}",
    )

    chart_fp2_event = st.plotly_chart(
        fig_fp2,
        use_container_width=True,
        on_select="rerun",
        selection_mode=["box"],
        key=f"as3_chart_fp2_{v2}",
    )

    if chart_fp2_event and "selection" in chart_fp2_event:
      boxes = chart_fp2_event["selection"].get("box", [])
      if boxes and len(boxes) > 0 and "x" in boxes[0]:
        x_pts = boxes[0]["x"]
        n_s = round(float(min(x_pts)), 3)
        n_e = round(float(max(x_pts)), 3)

        if fp2_mode == "Zoom View":
          if (
              abs(n_s - fp2_curr_zoom[0]) > 0.005
              or abs(n_e - fp2_curr_zoom[1]) > 0.005
          ):
            st.session_state["as3_FP2_zoom_range"] = [n_s, n_e]
            st.rerun()
        else:
          if (
              abs(n_s - st.session_state["as3_FP2_base_s"]) > 0.005
              or abs(n_e - st.session_state["as3_FP2_base_e"]) > 0.005
          ):
            st.session_state["as3_FP2_base_s"] = n_s
            st.session_state["as3_FP2_base_e"] = n_e
            st.session_state[f"as3_bs_FP2_{v2}"] = n_s
            st.session_state[f"as3_be_FP2_{v2}"] = n_e
            st.rerun()

    c2_b1, c2_b2, c2_b3 = st.columns([1.5, 1.5, 1.5])
    with c2_b1:
      as3_b_start_fp2 = st.number_input(
          "FP2 Baseline Start (s):",
          min_value=k_t_start,
          max_value=k_t_end,
          value=float(st.session_state["as3_FP2_base_s"]),
          step=0.01,
          format="%.3f",
          key=f"as3_bs_FP2_{v2}",
      )
      st.session_state["as3_FP2_base_s"] = as3_b_start_fp2
    with c2_b2:
      as3_b_end_fp2 = st.number_input(
          "FP2 Baseline End (s):",
          min_value=k_t_start,
          max_value=k_t_end,
          value=float(st.session_state["as3_FP2_base_e"]),
          step=0.01,
          format="%.3f",
          key=f"as3_be_FP2_{v2}",
      )
      st.session_state["as3_FP2_base_e"] = as3_b_end_fp2
    with c2_b3:
      st.write("")
      st.write("")
      if st.button("Apply Zeroing to FP2", type="primary", key="as3_btn_app_fp2"):
        st.session_state["as3_FP2_debias_applied"] = True
        st.success("FP2 baseline offset zeroed!")
        st.rerun()

    st.markdown("---")

# =============================================================
    # STEP 3: SELECT TWO FOOTSTRIKES FROM FULL SIGNAL
    # =============================================================
    st.markdown("### Step 3: Select Two Footstrikes from Full Signal")
    st.caption(
        "Use **'Zoom View'** to get a closer look at the steps, or switch to"
        " **'Select Footstrikes Window'** to isolate two footstrikes. The"
        " graph automatically updates to focus on your selection."
    )

    if "as3_fs_view_ver" not in st.session_state:
      st.session_state["as3_fs_view_ver"] = 0
    if "as3_fs_zoom_range" not in st.session_state:
      st.session_state["as3_fs_zoom_range"] = [k_t_start, k_t_end]

    v_fs = st.session_state["as3_fs_view_ver"]

    col_fs_ctrl, col_fs_rst = st.columns([3, 1])
    with col_fs_ctrl:
      fs_tool_mode = st.radio(
          "Footstrike Selection Tool Mode:",
          ["Zoom View", "Select Footstrikes Window"],
          horizontal=True,
          key=f"as3_fs_tool_mode_{v_fs}",
      )
    with col_fs_rst:
      st.write("")
      if st.button("Reset Footstrikes View", key="as3_fs_rst_btn"):
        st.session_state["as3_fs_zoom_range"] = [k_t_start, k_t_end]
        st.session_state["as3_win_s"] = round(k_t_start + 0.5, 3)
        st.session_state["as3_win_e"] = round(min(k_t_end, k_t_start + 2.3), 3)
        st.session_state["as3_fs_view_ver"] += 1
        st.rerun()

    v_fs = st.session_state["as3_fs_view_ver"]
    fig_fs = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "FP1: Vertical Force (F1Z)",
            "FP2: Vertical Force (F2Z)",
        ),
    )
    if "F1Z" in raw_fp1.data:
      fig_fs.add_trace(
          go.Scatter(
              x=raw_fp1.time,
              y=raw_fp1.data["F1Z"],
              mode="lines",
              line=dict(color="#3b82f6", width=1.5),
              name="FP1 Fz",
          ),
          row=1,
          col=1,
      )
    if "F2Z" in raw_fp2.data:
      fig_fs.add_trace(
          go.Scatter(
              x=raw_fp2.time,
              y=raw_fp2.data["F2Z"],
              mode="lines",
              line=dict(color="#22c55e", width=1.5),
              name="FP2 Fz",
          ),
          row=2,
          col=1,
      )

    curr_win_s = float(st.session_state["as3_win_s"])
    curr_win_e = float(st.session_state["as3_win_e"])

    for r in [1, 2]:
      fig_fs.add_vrect(
          x0=curr_win_s,
          x1=curr_win_e,
          fillcolor="rgba(234, 179, 8, 0.25)",
          line_width=2,
          line_dash="dash",
          line_color="#eab308",
          annotation_text="Two Footstrikes Window" if r == 1 else "",
          annotation_position="top left",
          row=r,
          col=1,
      )

    fs_curr_zoom = st.session_state["as3_fs_zoom_range"]

    fig_fs.update_layout(
        template="plotly_dark",
        height=400,
        dragmode="select",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=35, b=20),
        xaxis=dict(range=fs_curr_zoom, autorange=False),
        xaxis2=dict(title="Time (s)", range=fs_curr_zoom, autorange=False),
        yaxis=dict(title="Force (N)"),
        yaxis2=dict(title="Force (N)"),
        uirevision=f"fs_rev_{v_fs}",
    )

    chart_fs_event = st.plotly_chart(
        fig_fs,
        use_container_width=True,
        on_select="rerun",
        selection_mode=["box"],
        key=f"as3_fs_chart_{v_fs}",
    )

    # -------------------------------------------------------------
    # Capture Mouse Selection & Update State + Version Counter
    # -------------------------------------------------------------
    if chart_fs_event and "selection" in chart_fs_event:
      fs_boxes = chart_fs_event["selection"].get("box", [])
      if fs_boxes and len(fs_boxes) > 0 and "x" in fs_boxes[0]:
        x_pts = fs_boxes[0]["x"]
        n_s = round(float(min(x_pts)), 3)
        n_e = round(float(max(x_pts)), 3)

        if fs_tool_mode == "Zoom View":
          if (
              abs(n_s - fs_curr_zoom[0]) > 0.005
              or abs(n_e - fs_curr_zoom[1]) > 0.005
          ):
            st.session_state["as3_fs_zoom_range"] = [n_s, n_e]
            st.session_state["as3_fs_view_ver"] += 1
            st.rerun()
        else:
          # Automatically update window bounds, zoom to selection, and increment version
          if (
              abs(n_s - st.session_state["as3_win_s"]) > 0.005
              or abs(n_e - st.session_state["as3_win_e"]) > 0.005
          ):
            st.session_state["as3_win_s"] = n_s
            st.session_state["as3_win_e"] = n_e
            st.session_state["as3_fs_zoom_range"] = [
                max(k_t_start, n_s - 0.05),
                min(k_t_end, n_e + 0.05),
            ]
            st.session_state["as3_fs_view_ver"] += 1
            st.rerun()

    # -------------------------------------------------------------
    # Number Inputs Tied to the Current Version
    # -------------------------------------------------------------
    v_fs = st.session_state["as3_fs_view_ver"]
    c_fs_plat, c_fs_s, c_fs_e = st.columns([1.5, 2, 2])
    with c_fs_plat:
      as3_plate = st.radio(
          "Target Platform for 6-Component Plotting:",
          ["FP1", "FP2"],
          horizontal=True,
          key=f"as3_fs_plate_choice_{v_fs}",
      )
    with c_fs_s:
      as3_t_start = st.number_input(
          "Window Start (s):",
          min_value=k_t_start,
          max_value=k_t_end,
          value=float(st.session_state["as3_win_s"]),
          step=0.01,
          format="%.3f",
          key=f"as3_num_t_s_{v_fs}",
      )
      st.session_state["as3_win_s"] = as3_t_start
    with c_fs_e:
      as3_t_end = st.number_input(
          "Window End (s):",
          min_value=k_t_start,
          max_value=k_t_end,
          value=float(st.session_state["as3_win_e"]),
          step=0.01,
          format="%.3f",
          key=f"as3_num_t_e_{v_fs}",
      )
      st.session_state["as3_win_e"] = as3_t_end

    st.markdown("---")    

# =============================================================
    # STEP 4: APPLY FILTERING & SIGNAL CONDITIONING
    # =============================================================
    st.markdown("### Step 4: Apply Filtering Algorithm & Parameters")
    st.caption(
        "Select your filtering or smoothing algorithm to condition the isolated footstrike signals."
    )

    f_col1, f_col2 = st.columns(2)
    with f_col1:
      as3_filter_mode = st.selectbox(
          "Filter Algorithm:",
          [
              "Butterworth Low-pass",
              "Butterworth High-pass",
              "Moving Average Smoothing",
              "None (Raw)",
          ],
          index=0,
          key="as3_filt_sel_step4",
      )

    with f_col2:
      as3_fc = None
      as3_smooth_window = None

      if as3_filter_mode == "Butterworth Low-pass":
        as3_fc = st.slider(
            "Low-pass Cutoff Frequency Fc (Hz):",
            min_value=5,
            max_value=200,
            value=100,
            step=5,
            key="as3_fc_slider_lp",
            help="Attenuates frequencies higher than Fc (mains hum, plate vibrations).",
        )
      elif as3_filter_mode == "Butterworth High-pass":
        as3_fc = st.slider(
            "High-pass Cutoff Frequency Fc (Hz):",
            min_value=1,
            max_value=50,
            value=10,
            step=1,
            key="as3_fc_slider_hp",
            help="Attenuates frequencies lower than Fc (low-frequency baseline drift).",
        )
      elif as3_filter_mode == "Moving Average Smoothing":
        as3_smooth_window = st.slider(
            "Smoothing Window Length (Samples):",
            min_value=3,
            max_value=101,
            value=11,
            step=2,
            key="as3_smooth_slider",
            help="Number of samples across the moving average kernel (must be odd).",
        )
      else:
        st.caption("Displaying unfiltered raw data.")

    # -------------------------------------------------------------
    # Compute Pipeline Transformations
    # -------------------------------------------------------------
    raw_base_ts = raw_fp1 if as3_plate == "FP1" else raw_fp2
    p_num = "1" if as3_plate == "FP1" else "2"
    fp_work = copy.deepcopy(raw_base_ts)

    # 1. Scaling
    if apply_scale:
      for f_key in [f"F{p_num}X", f"F{p_num}Y", f"F{p_num}Z"]:
        if f_key in fp_work.data:
          fp_work.data[f_key] = fp_work.data[f_key] * 1.0
    if apply_moment_scale:
      for m_key in [f"M{p_num}X", f"M{p_num}Y", f"M{p_num}Z"]:
        if m_key in fp_work.data:
          fp_work.data[m_key] = fp_work.data[m_key] * 1.0

    # 2. Debiasing using plate-specific baseline interval
    is_debiased_applied = st.session_state[f"as3_{as3_plate}_debias_applied"]
    plat_b_start = st.session_state[f"as3_{as3_plate}_base_s"]
    plat_b_end = st.session_state[f"as3_{as3_plate}_base_e"]

    if is_debiased_applied and plat_b_end > plat_b_start:
      b_p_s = max(0.0, min(1.0, (plat_b_start - k_t_start) / kin_duration))
      b_p_e = max(0.0, min(1.0, (plat_b_end - k_t_start) / kin_duration))
      b_i_s = int(b_p_s * n_total_fp)
      b_i_e = max(b_i_s + 1, int(b_p_e * n_total_fp))
      for k in fp_work.data.keys():
        bias_val = np.nanmean(fp_work.data[k][b_i_s:b_i_e])
        fp_work.data[k] -= bias_val

    # 3. Slicing Footstrikes Window
    w_p_s = max(0.0, min(1.0, (as3_t_start - k_t_start) / kin_duration))
    w_p_e = max(0.0, min(1.0, (as3_t_end - k_t_start) / kin_duration))
    w_i_s = int(w_p_s * n_total_fp)
    w_i_e = max(w_i_s + 2, int(w_p_e * n_total_fp))

    n_samples_window = w_i_e - w_i_s
    time_window = np.linspace(as3_t_start, as3_t_end, n_samples_window)

    window_raw_ts = ktk.TimeSeries(time=time_window)
    for k in fp_work.data.keys():
      window_raw_ts.data[k] = np.copy(raw_base_ts.data[k][w_i_s:w_i_e])

    window_processed_ts = ktk.TimeSeries(time=time_window)
    for k in fp_work.data.keys():
      window_processed_ts.data[k] = np.copy(fp_work.data[k][w_i_s:w_i_e])

# 4. Filtering / Smoothing Application
    filter_legend_label = "Processed"
    window_filtered_ts = copy.deepcopy(window_processed_ts)

    try:
      if as3_filter_mode == "Butterworth Low-pass" and as3_fc is not None:
        window_filtered_ts = ktk.filters.butter(window_processed_ts, fc=as3_fc)
        window_filtered_ts.time = time_window
        filter_legend_label = f"Low-pass ({as3_fc} Hz)"

      elif as3_filter_mode == "Butterworth High-pass" and as3_fc is not None:
        # High-pass: Raw minus Low-pass baseline
        lp_baseline = ktk.filters.butter(window_processed_ts, fc=as3_fc)
        for k in window_filtered_ts.data.keys():
          window_filtered_ts.data[k] = (
              window_processed_ts.data[k] - lp_baseline.data[k]
          )
        window_filtered_ts.time = time_window
        filter_legend_label = f"High-pass ({as3_fc} Hz)"

      elif (
          as3_filter_mode == "Moving Average Smoothing"
          and as3_smooth_window is not None
      ):
        window_filtered_ts = ktk.filters.smooth(
            window_processed_ts, window_length=as3_smooth_window
        )
        window_filtered_ts.time = time_window
        filter_legend_label = f"Smoothed ({as3_smooth_window} pts)"

      else:
        window_filtered_ts = copy.deepcopy(window_processed_ts)
        filter_legend_label = "Processed (Unfiltered)"
    except Exception as e:
      st.warning(
          f"Filter could not be applied ({e}). Falling back to unfiltered data."
      )
      window_filtered_ts = copy.deepcopy(window_processed_ts)
      filter_legend_label = "Processed (Unfiltered)"
      
      
    st.markdown("---")

    # -------------------------------------------------------------
    # 6-Component Output Display
    # -------------------------------------------------------------
    st.markdown(f"#### Six-Component Kinetic Plots ({as3_plate}: 2 Footstrikes)")

    component_specs = [
        (
            f"F{p_num}X",
            "Fx: Medio-Lateral Force (Side-to-Side Shear)",
            "Force (N)",
            "#ef4444",
        ),
        (
            f"F{p_num}Y",
            "Fy: Antero-Posterior Force (Braking & Propulsion)",
            "Force (N)",
            "#22c55e",
        ),
        (
            f"F{p_num}Z",
            "Fz: Vertical Ground Reaction Force (Weight Bearing)",
            "Force (N)",
            "#3b82f6",
        ),
        (
            f"M{p_num}X",
            "Mx: Moment about Medio-Lateral Axis",
            "Moment (N·m)",
            "#f97316",
        ),
        (
            f"M{p_num}Y",
            "My: Moment about Antero-Posterior Axis",
            "Moment (N·m)",
            "#eab308",
        ),
        (
            f"M{p_num}Z",
            "Mz: Free Moment about Vertical Axis",
            "Moment (N·m)",
            "#a855f7",
        ),
    ]

    col_g1, col_g2 = st.columns(2)

    for i, (ch_key, comp_title, y_label, comp_color) in enumerate(
        component_specs
    ):
      target_col = col_g1 if (i % 2 == 0) else col_g2
      with target_col:
        fig_comp = go.Figure()

        if ch_key in window_raw_ts.data:
          y_raw = np.asarray(window_raw_ts.data[ch_key]).squeeze()
          fig_comp.add_trace(
              go.Scatter(
                  x=window_raw_ts.time,
                  y=y_raw,
                  mode="lines",
                  name="Unprocessed (Raw)",
                  line=dict(color="#94a3b8", dash="dot", width=1.5),
                  opacity=0.6,
              )
          )

        if (
            window_filtered_ts is not None
            and ch_key in window_filtered_ts.data
        ):
          y_filt = np.asarray(window_filtered_ts.data[ch_key]).squeeze()
          fig_comp.add_trace(
              go.Scatter(
                  x=window_filtered_ts.time,
                  y=y_filt,
                  mode="lines",
                  name=filter_legend_label,
                  line=dict(color=comp_color, width=2.5),
              )
          )

        fig_comp.update_layout(
            title=comp_title,
            xaxis_title="Time (s)",
            yaxis_title=y_label,
            template="plotly_dark",
            hovermode="x unified",
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(range=[as3_t_start, as3_t_end], autorange=False),
        )
        st.plotly_chart(fig_comp, use_container_width=True)
        
    # -------------------------------------------------------------
    # 5. Assignment Helper & Theory Explanations
    # -------------------------------------------------------------
    with st.expander("💡 Lab Report Guide: Answers to Assignment 3 Questions"):
      st.markdown(
          r"""
        ### 1. What Each of the 6 Components Represents:
        * **$F_x$ (Medio-Lateral Force):** Lateral and medial shear forces exerted on the platform.
        * **$F_y$ (Antero-Posterior Force):** The anterior/posterior shear force.
        * **$F_z$ (Vertical Force):** Total vertical load bearing. 
        * **$M_x$ (Frontal Moment):** Moment about the X axis, produced primarily by the vertical load acting at an offset distance along the Y axis ($F_z \times d_y$).
        * **$M_y$ (Sagittal Moment):** Moment about the Y axis, produced by the vertical load offset laterally along the X axis ($F_z \times d_x$).
        * **$M_z$ (Free Vertical Moment / Torque):** 

        ---

        ### 2. Explanation of Signal Conditioning Steps:

        #### **A. Unit Scaling & Sign Conventions**
        * **What is scaling?** Transducers output raw analog signals in millivolts ($\text{mV}$) or ADC binary counts. 
        * **How the app does it:** 
        * **Why it is important:** 

        #### **B. Quiescent Baseline Debiasing (Zeroing)**
        * **What debiasing?** Removing DC electrical offset voltages.
        * **How the app does it:** 
          $$x_{\text{debiased}}(t) = x(t) - \mu_{\text{baseline}}$$
        * **Why it is important:** 

        #### **C. Low-Pass Filtering**
        * **What is low pass filtering?** 
        * **How the app does it:** 
        * **Why it is important:** 
        """
      )
  # =========================================================================
  # STEP 1: KINEMATICS & GAIT CYCLE IDENTIFICATION
  # =========================================================================
  with active_tabs[4]:
    st.subheader("Step 1: Identify and Isolate One Gait Cycle")
    st.caption(
        "Inspect sagittal kinematics. Drag a box across one gait cycle (heel"
        " strike to heel strike) or type timestamps below."
    )

    angles_ts = st.session_state["angles"]
    t_min = float(angles_ts.time[0])
    t_max = float(angles_ts.time[-1])

    v = st.session_state.get("step1_version", 0)
    k_start = f"input_t_start_{v}"
    k_end = f"input_t_end_{v}"

    if "t_start" not in st.session_state or st.session_state["t_start"] < t_min:
      st.session_state["t_start"] = t_min
    if (
        "t_end" not in st.session_state
        or st.session_state["t_end"] <= st.session_state["t_start"]
    ):
      st.session_state["t_end"] = min(t_min + 1.2, t_max)

    if k_start not in st.session_state:
      st.session_state[k_start] = float(st.session_state["t_start"])
    if k_end not in st.session_state:
      st.session_state[k_end] = float(st.session_state["t_end"])

    available_keys = list(angles_ts.data.keys())
    preferred_order = ["AnkleR", "AnkleL", "KneeR", "KneeL", "HipR", "HipL"]
    joint_options = [j for j in preferred_order if j in available_keys]
    if not joint_options:
      joint_options = available_keys

    c_sel1, c_sel2 = st.columns([2, 1])
    with c_sel1:
      chosen_joint = st.selectbox("Inspection Joint:", joint_options)
    with c_sel2:
      zoom_to_window = st.checkbox(
          "🔍 Zoom graph to selected cycle",
          value=st.session_state.get("cycle_locked", False),
      )

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

    t_curr_s = float(st.session_state["t_start"])
    t_curr_e = float(st.session_state["t_end"])

    fig_kin.add_vrect(
        x0=t_curr_s,
        x1=t_curr_e,
        fillcolor="rgba(34, 197, 94, 0.25)",
        line_width=2,
        line_dash="dash",
        line_color="#22c55e",
        annotation_text="Selected Window",
        annotation_position="top left",
    )

    x_axis_range = (
        [t_curr_s - 0.05, t_curr_e + 0.05]
        if zoom_to_window
        else [t_min, t_max]
    )

    fig_kin.update_layout(
        title=f"{chosen_joint} Sagittal Angle (Flexion/Extension)",
        xaxis_title="Time (s)",
        yaxis_title="Angle (deg)",
        template="plotly_dark",
        dragmode="select",
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(range=x_axis_range, autorange=False),
        uirevision=f"rev_{v}_{zoom_to_window}",
    )

    chart_event = st.plotly_chart(
        fig_kin,
        use_container_width=True,
        on_select="rerun",
        selection_mode=["box"],
        key=f"kinematics_chart_{v}",
    )

    if chart_event and "selection" in chart_event:
      selection_dict = chart_event["selection"]
      x_vals = None
      if "box" in selection_dict and len(selection_dict["box"]) > 0:
        box = selection_dict["box"][0]
        if "x" in box and len(box["x"]) >= 2:
          x_vals = box["x"]
      elif "points" in selection_dict and len(selection_dict["points"]) > 1:
        pts_x = [p["x"] for p in selection_dict["points"] if "x" in p]
        if pts_x:
          x_vals = [min(pts_x), max(pts_x)]

      if x_vals:
        new_s = round(float(min(x_vals)), 3)
        new_e = round(float(max(x_vals)), 3)
        if (
            abs(new_s - st.session_state["t_start"]) > 0.005
            or abs(new_e - st.session_state["t_end"]) > 0.005
        ):
          st.session_state["t_start"] = new_s
          st.session_state["t_end"] = new_e
          st.session_state[k_start] = new_s
          st.session_state[k_end] = new_e
          st.rerun()

    st.markdown("#### 🎯 Student Decision: Set Cycle Bounds")
    c_col1, c_col2, c_col3, c_col4 = st.columns([2, 2, 1.2, 1])

    with c_col1:
      val_s = st.number_input(
          "Cycle Initial Contact (s):",
          min_value=t_min,
          max_value=t_max,
          step=0.01,
          format="%.3f",
          key=k_start,
      )
    with c_col2:
      val_e = st.number_input(
          "Next Initial Contact (s):",
          min_value=t_min,
          max_value=t_max,
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
      if st.button("Reset View / Selection"):
        st.session_state["cycle_locked"] = False
        st.session_state["filter_locked"] = False
        st.session_state["t_start"] = t_min
        st.session_state["t_end"] = min(t_min + 1.2, t_max)
        st.session_state["step1_version"] = v + 1
        st.rerun()

    if st.session_state.get("cycle_locked", False):
      st.success(
          f"Cycle locked: {st.session_state['t_start']:.3f}s to"
          f" {st.session_state['t_end']:.3f}s. Proceed to Step 2 above."
      )

  # =========================================================================
  # STEP 2: GRF DECISIONS (INDEPENDENT FP1 & FP2 ZEROING)
  # =========================================================================
  if st.session_state.get("cycle_locked", False) and len(active_tabs) > 3:
    with active_tabs[5]:
      st.subheader("Step 2: Ground Reaction Force (GRF) Processing Decisions")
      st.caption(
          "Configure baseline zeroing and filtering for each force plate"
          " independently."
      )

      angles_ts = st.session_state["angles"]
      k_t_start = float(angles_ts.time[0])
      k_t_end = float(angles_ts.time[-1])
      kin_duration = k_t_end - k_t_start

      for p_name, def_start, def_end in [
          ("FP1", k_t_start + 0.2, k_t_start + 0.6),
          ("FP2", k_t_start + 1.0, k_t_start + 1.4),
      ]:
        if f"{p_name}_debias" not in st.session_state:
          st.session_state[f"{p_name}_debias"] = False
        if f"{p_name}_b_start" not in st.session_state:
          st.session_state[f"{p_name}_b_start"] = round(float(def_start), 3)
        if f"{p_name}_b_end" not in st.session_state:
          st.session_state[f"{p_name}_b_end"] = round(float(def_end), 3)

      plate_choice = st.radio(
          "Active Force Plate to Inspect & Configure:",
          ["FP1", "FP2"],
          horizontal=True,
          key="grf_plate_sel",
      )

      st.markdown(f"#### 1. Baseline Zeroing for **{plate_choice}**")
      c_col1, c_col2, c_col3 = st.columns([1.5, 2, 2])

      with c_col1:
        st.session_state[f"{plate_choice}_debias"] = st.checkbox(
            f"Zero Baseline for {plate_choice}",
            value=st.session_state[f"{plate_choice}_debias"],
            key=f"cb_debias_{plate_choice}",
        )

      with c_col2:
        st.session_state[f"{plate_choice}_b_start"] = st.number_input(
            f"{plate_choice} Quiescent Start (s):",
            min_value=k_t_start,
            max_value=k_t_end,
            value=float(st.session_state[f"{plate_choice}_b_start"]),
            step=0.01,
            format="%.3f",
            disabled=not st.session_state[f"{plate_choice}_debias"],
            key=f"num_b_start_{plate_choice}",
        )

      with c_col3:
        st.session_state[f"{plate_choice}_b_end"] = st.number_input(
            f"{plate_choice} Quiescent End (s):",
            min_value=k_t_start,
            max_value=k_t_end,
            value=float(st.session_state[f"{plate_choice}_b_end"]),
            step=0.01,
            format="%.3f",
            disabled=not st.session_state[f"{plate_choice}_debias"],
            key=f"num_b_end_{plate_choice}",
        )

      st.markdown("#### 2. Filtering Decisions")
      f_col1, f_col2 = st.columns(2)
      with f_col1:
        filter_mode = st.selectbox(
            "Filter Algorithm:",
            ["None (Raw)", "Butterworth Low-pass"],
            index=0,
            key="filter_mode_sel",
        )
      with f_col2:
        if filter_mode == "Butterworth Low-pass":
          cutoff_fc = st.slider(
              "Cutoff Frequency Fc (Hz):",
              min_value=5,
              max_value=200,
              value=st.session_state.get("chosen_fc", 100),
              step=5,
          )
        else:
          cutoff_fc = None
          st.caption("Displaying unfiltered raw signals.")

      raw_plate_ts = (
          st.session_state["FP1_raw"]
          if plate_choice == "FP1"
          else st.session_state["FP2_raw"]
      )
      fp_working = copy.deepcopy(raw_plate_ts)
      n_total_fp = len(fp_working.time)

      if st.session_state[f"{plate_choice}_debias"]:
        b_s = st.session_state[f"{plate_choice}_b_start"]
        b_e = st.session_state[f"{plate_choice}_b_end"]

        b_pct_start = max(0.0, min(1.0, (b_s - k_t_start) / kin_duration))
        b_pct_end = max(0.0, min(1.0, (b_e - k_t_start) / kin_duration))
        b_idx_start = int(b_pct_start * n_total_fp)
        b_idx_end = max(b_idx_start + 1, int(b_pct_end * n_total_fp))

        for k in fp_working.data.keys():
          baseline_offset = np.nanmean(
              fp_working.data[k][b_idx_start:b_idx_end]
          )
          fp_working.data[k] -= baseline_offset

      user_t_start = float(st.session_state["t_start"])
      user_t_end = float(st.session_state["t_end"])

      c_pct_start = max(
          0.0, min(1.0, (user_t_start - k_t_start) / kin_duration)
      )
      c_pct_end = max(0.0, min(1.0, (user_t_end - k_t_start) / kin_duration))
      c_idx_start = int(c_pct_start * n_total_fp)
      c_idx_end = max(c_idx_start + 2, int(c_pct_end * n_total_fp))

      cycle_fp_raw = ktk.TimeSeries()
      n_cycle_samples = c_idx_end - c_idx_start
      cycle_fp_raw.time = np.linspace(
          user_t_start, user_t_end, n_cycle_samples
      )

      for k in fp_working.data.keys():
        cycle_fp_raw.data[k] = np.copy(
            fp_working.data[k][c_idx_start:c_idx_end]
        )

      cycle_fp_filt = None
      if filter_mode == "Butterworth Low-pass" and cutoff_fc is not None:
        cycle_fp_filt = ktk.filters.butter(cycle_fp_raw, fc=cutoff_fc)
        cycle_fp_filt.time = np.copy(cycle_fp_raw.time)

      p = "1" if plate_choice == "FP1" else "2"
      axes = [
          ("X (M-L)", f"F{p}X", "#ef4444"),
          ("Y (A-P)", f"F{p}Y", "#22c55e"),
          ("Z (Vertical)", f"F{p}Z", "#3b82f6"),
      ]

      fig_grf = go.Figure()
      is_debiased = st.session_state[f"{plate_choice}_debias"]

      for label, ch, color in axes:
        if ch in cycle_fp_raw.data:
          if cycle_fp_filt is None:
            fig_grf.add_trace(
                go.Scatter(
                    x=cycle_fp_raw.time,
                    y=cycle_fp_raw.data[ch],
                    mode="lines",
                    line=dict(color=color, width=2.5),
                    name=f"{label} {'(Zeroed)' if is_debiased else '(Raw)'}",
                )
            )
          else:
            fig_grf.add_trace(
                go.Scatter(
                    x=cycle_fp_raw.time,
                    y=cycle_fp_raw.data[ch],
                    mode="lines",
                    line=dict(color=color, dash="dot", width=1.5),
                    opacity=0.45,
                    name=f"{label} Raw",
                )
            )
            fig_grf.add_trace(
                go.Scatter(
                    x=cycle_fp_filt.time,
                    y=cycle_fp_filt.data[ch],
                    mode="lines",
                    line=dict(color=color, width=2.5),
                    name=f"{label} Filtered ({cutoff_fc} Hz)",
                )
            )

      fig_grf.update_layout(
          title=f"{plate_choice} Force Traces",
          xaxis_title="Time (s)",
          yaxis_title="Force (N)",
          template="plotly_dark",
          hovermode="x unified",
          margin=dict(l=20, r=20, t=40, b=20),
          xaxis=dict(
              range=[user_t_start, user_t_end],
              autorange=False,
          ),
      )
      st.plotly_chart(fig_grf, use_container_width=True)

      st.markdown("#### 🎯 Confirm Decisions")
      if st.button("Accept Force Processing Decisions", type="primary"):
        st.session_state["chosen_plate"] = plate_choice
        st.session_state["chosen_filter"] = filter_mode
        st.session_state["chosen_fc"] = cutoff_fc if cutoff_fc else 100
        st.session_state["cycle_fp_processed"] = (
            cycle_fp_filt if cycle_fp_filt is not None else cycle_fp_raw
        )
        st.session_state["filter_locked"] = True
        st.success("Decisions saved! Step 3 (COP Analysis) is now unlocked.")
        st.rerun()

  # =========================================================================
  # STEP 3: COP & BUTTERFLY PLOT
  # =========================================================================
  if st.session_state.get("filter_locked", False) and len(active_tabs) > 4:
    with active_tabs[6]:
      st.subheader("Step 3: Center of Pressure (COP) Analysis")
      chosen_plate = st.session_state.get("chosen_plate", "FP1")
      fc_val = st.session_state.get("chosen_fc", 100)
      filt_name = st.session_state.get("chosen_filter", "None (Raw)")
      filt_suffix = (
          f" ({fc_val} Hz)" if filt_name == "Butterworth Low-pass" else ""
      )
      debias_status = (
          "Zeroed (Debiased)"
          if st.session_state.get(f"{chosen_plate}_debias", False)
          else "Raw (Non-Zeroed)"
      )

      st.caption(
          f"Parameters: **{chosen_plate}** | Baseline: **{debias_status}** |"
          f" Filter: **{filt_name}{filt_suffix}**"
      )

      final_fp = st.session_state["cycle_fp_processed"]
      plate_prefix = "1" if chosen_plate == "FP1" else "2"
      fz = final_fp.data[f"F{plate_prefix}Z"]
      mx = final_fp.data[f"M{plate_prefix}X"]
      my = final_fp.data[f"M{plate_prefix}Y"]

      max_fz = float(np.nanmax(np.abs(fz))) if len(fz) > 0 else 500.0
      default_gate = min(40.0, max_fz * 0.15)

      cop_ctrl1, cop_ctrl2 = st.columns([2, 2])
      with cop_ctrl1:
        fz_threshold = st.slider(
            "Vertical Force Gate (|Fz| ≥ N):",
            min_value=5.0,
            max_value=max(100.0, float(np.ceil(max_fz))),
            value=float(default_gate),
            step=5.0,
        )
      with cop_ctrl2:
        show_arrows = st.checkbox("Show Progression Markers", value=True)

      EPSILON = 1e-6
      contact_mask = np.abs(fz) >= fz_threshold
      cop_x = np.where(contact_mask, -my / (fz + EPSILON), np.nan)
      cop_y = np.where(contact_mask, mx / (fz + EPSILON), np.nan)

      col_butterfly, col_timeseries = st.columns(2)
      with col_butterfly:
        valid_indices = np.where(contact_mask)[0]
        fig_butterfly = go.Figure()
        if len(valid_indices) == 0:
          st.warning(f"No samples have |Fz| ≥ {fz_threshold:.1f} N.")
        else:
          x_valid = cop_x[valid_indices]
          y_valid = cop_y[valid_indices]
          t_valid = final_fp.time[valid_indices]
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
                      colorbar=dict(title="Time (s)", thickness=12),
                  ),
                  name="COP Path",
              )
          )
          fig_butterfly.add_trace(
              go.Scatter(
                  x=[x_valid[0]],
                  y=[y_valid[0]],
                  mode="markers+text",
                  marker=dict(size=12, color="#22c55e", symbol="star"),
                  text=["Heel Strike"],
                  textposition="top center",
                  name="Heel Strike",
              )
          )
          fig_butterfly.add_trace(
              go.Scatter(
                  x=[x_valid[-1]],
                  y=[y_valid[-1]],
                  mode="markers+text",
                  marker=dict(size=12, color="#ef4444", symbol="x"),
                  text=["Toe Off"],
                  textposition="bottom center",
                  name="Toe Off",
              )
          )

        fig_butterfly.update_layout(
            title="Planar Butterfly Path (COPx vs. COPy)",
            xaxis_title="Medio-Lateral COPx (m)",
            yaxis_title="Antero-Posterior COPy (m)",
            template="plotly_dark",
            yaxis=dict(scaleanchor="x", scaleratio=1),
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_butterfly, use_container_width=True)

      with col_timeseries:
        fig_ts = go.Figure()
        fig_ts.add_trace(
            go.Scatter(
                x=final_fp.time,
                y=cop_x,
                mode="lines",
                line=dict(color="#3b82f6", width=2),
                name="COPx (M-L)",
            )
        )
        fig_ts.add_trace(
            go.Scatter(
                x=final_fp.time,
                y=cop_y,
                mode="lines",
                line=dict(color="#ef4444", width=2),
                name="COPy (A-P)",
            )
        )
        fig_ts.update_layout(
            title="COP Displacement vs. Time",
            xaxis_title="Time (s)",
            yaxis_title="Displacement (m)",
            template="plotly_dark",
            hovermode="x unified",
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_ts, use_container_width=True)