#!/usr/bin/env python3
"""
param_tuner.py
--------------
A local web server for real-time parameter tuning of the LiDAR pipeline.

Runs alongside your ROS2 nodes and sends 'ros2 param set' commands
to update DBSCAN and PCL parameters without restarting anything.
Also saves tuned values back to YAML config files.

Usage:
    pipenv run python3 param_tuner.py

Then open: http://localhost:5000

Requirements (add to pipenv if needed):
    pipenv install flask
"""

import json
import os
import subprocess
import yaml
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
WS         = Path.home() / "lidar_ws"
CONFIG_DIR = WS / "src" / "lidar_bringup" / "config"

FUSION_YAML       = CONFIG_DIR / "fusion.yaml"
DBSCAN_FUSED_YAML = CONFIG_DIR / "dbscan_fused.yaml"
DETECTOR_CPP_YAML = CONFIG_DIR / "detector_cpp.yaml"

# ── Parameter definitions ──────────────────────────────────────────────────
# Each entry: node_name, param_name, label, min, max, step, description
PARAMS = {
    "fusion": [
        {
            "node":  "cloud_fusion",
            "param": "voxel_size",
            "label": "Voxel Size (m)",
            "min":   0.01, "max": 0.20, "step": 0.01,
            "desc":  "Grid cell size for deduplication. Larger = fewer points, faster. Smaller = more detail."
        },
        {
            "node":  "cloud_fusion",
            "param": "max_age_sec",
            "label": "Max Age (s)",
            "min":   0.1, "max": 5.0, "step": 0.1,
            "desc":  "Discard sensor messages older than this. Increase if fusion rate drops on live sensors."
        },
        {
            "node":  "cloud_fusion",
            "param": "publish_rate_hz",
            "label": "Publish Rate (Hz)",
            "min":   1.0, "max": 20.0, "step": 1.0,
            "desc":  "How often the fused cloud is published. 10Hz recommended."
        },
    ],
    "dbscan": [
        {
            "node":  "dbscan_fused",
            "param": "cluster_eps",
            "label": "Epsilon (m)",
            "min":   0.1, "max": 2.0, "step": 0.05,
            "desc":  "Max distance between two points to be in the same cluster. Larger = merges nearby objects."
        },
        {
            "node":  "dbscan_fused",
            "param": "cluster_min_pts",
            "label": "Min Points",
            "min":   5, "max": 100, "step": 1,
            "desc":  "Min points to form a cluster. Higher = ignores small noise clusters."
        },
        {
            "node":  "dbscan_fused",
            "param": "voxel_size",
            "label": "Voxel Size (m)",
            "min":   0.01, "max": 0.20, "step": 0.01,
            "desc":  "Downsample before clustering. Larger = faster DBSCAN but less precise."
        },
        {
            "node":  "dbscan_fused",
            "param": "min_range",
            "label": "Min Range (m)",
            "min":   0.1, "max": 5.0, "step": 0.1,
            "desc":  "Ignore points closer than this (removes sensor self-returns)."
        },
        {
            "node":  "dbscan_fused",
            "param": "max_range",
            "label": "Max Range (m)",
            "min":   1.0, "max": 80.0, "step": 1.0,
            "desc":  "Ignore points further than this."
        },
        {
            "node":  "dbscan_fused",
            "param": "min_z",
            "label": "Min Z (m)",
            "min":   -2.0, "max": 0.0, "step": 0.1,
            "desc":  "Remove points below this height (ground filter)."
        },
        {
            "node":  "dbscan_fused",
            "param": "max_z",
            "label": "Max Z (m)",
            "min":   0.5, "max": 5.0, "step": 0.1,
            "desc":  "Remove points above this height (ceiling filter)."
        },
    ],
    "pcl": [
        {
            "node":  "detector_cpp_fused",
            "param": "voxel_leaf",
            "label": "Voxel Leaf (m)",
            "min":   0.02, "max": 0.30, "step": 0.01,
            "desc":  "PCL voxel downsample leaf size. Directly controls processing speed."
        },
        {
            "node":  "detector_cpp_fused",
            "param": "cluster_tolerance",
            "label": "Cluster Tolerance (m)",
            "min":   0.05, "max": 2.0, "step": 0.05,
            "desc":  "Euclidean clustering distance threshold. Larger = merges nearby objects."
        },
        {
            "node":  "detector_cpp_fused",
            "param": "cluster_min_size",
            "label": "Min Cluster Size",
            "min":   5, "max": 200, "step": 5,
            "desc":  "Minimum points per cluster. Higher = ignores small detections."
        },
        {
            "node":  "detector_cpp_fused",
            "param": "cluster_max_size",
            "label": "Max Cluster Size",
            "min":   100, "max": 500000, "step": 1000,
            "desc":  "Maximum points per cluster. Lower = splits large merged blobs."
        },
        {
            "node":  "detector_cpp_fused",
            "param": "plane_dist_thresh",
            "label": "Ground Plane Thresh (m)",
            "min":   0.01, "max": 0.20, "step": 0.01,
            "desc":  "RANSAC inlier threshold for ground plane removal. Larger = removes more ground."
        },
        {
            "node":  "detector_cpp_fused",
            "param": "plane_max_iter",
            "label": "RANSAC Iterations",
            "min":   50, "max": 500, "step": 10,
            "desc":  "More iterations = better ground plane fit but slower processing."
        },
        {
            "node":  "detector_cpp_fused",
            "param": "sor_mean_k",
            "label": "SOR Mean K",
            "min":   5, "max": 50, "step": 1,
            "desc":  "Statistical outlier removal neighbours. More = stricter noise removal."
        },
        {
            "node":  "detector_cpp_fused",
            "param": "sor_stddev",
            "label": "SOR Std Dev",
            "min":   0.1, "max": 3.0, "step": 0.1,
            "desc":  "SOR threshold multiplier. Lower = removes more outliers (may remove real points)."
        },
        {
            "node":  "detector_cpp_fused",
            "param": "min_z",
            "label": "Min Z (m)",
            "min":   -2.0, "max": 0.0, "step": 0.1,
            "desc":  "Z passthrough lower bound."
        },
        {
            "node":  "detector_cpp_fused",
            "param": "max_z",
            "label": "Max Z (m)",
            "min":   0.5, "max": 5.0, "step": 0.1,
            "desc":  "Z passthrough upper bound."
        },
    ],
}

# ── ROS2 helpers ───────────────────────────────────────────────────────────
def ros2_param_set(node: str, param: str, value) -> dict:
    """Call ros2 param set and return result."""
    cmd = ["ros2", "param", "set", f"/{node}", param, str(value)]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3,
            env={**os.environ, "ROS_DOMAIN_ID": os.environ.get("ROS_DOMAIN_ID", "0")}
        )
        if result.returncode == 0:
            return {"ok": True, "msg": f"Set /{node} {param} = {value}"}
        else:
            return {"ok": False, "msg": result.stderr.strip() or result.stdout.strip()}
    except subprocess.TimeoutExpired:
        return {"ok": False, "msg": "ros2 param set timed out — is the node running?"}
    except FileNotFoundError:
        return {"ok": False, "msg": "ros2 not found — source your ROS2 workspace first"}


def ros2_param_get(node: str, param: str):
    """Get current value of a ROS2 parameter."""
    cmd = ["ros2", "param", "get", f"/{node}", param]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
        if result.returncode == 0:
            # Parse output like: "Double value is: 0.05"
            line = result.stdout.strip()
            for prefix in ["Double value is: ", "Integer value is: ", "Boolean value is: ", "String value is: "]:
                if prefix in line:
                    raw = line.split(prefix)[-1].strip()
                    try:
                        return float(raw)
                    except ValueError:
                        return raw
        return None
    except Exception:
        return None


def get_all_current_values() -> dict:
    """Fetch current values for all parameters from running nodes."""
    values = {}
    for group, params in PARAMS.items():
        for p in params:
            key = f"{p['node']}__{p['param']}"
            val = ros2_param_get(p["node"], p["param"])
            values[key] = val
    return values


def save_to_yaml(group: str, node: str, param: str, value) -> bool:
    """Persist a parameter change back to the appropriate YAML file."""
    yaml_map = {
        "cloud_fusion":       FUSION_YAML,
        "dbscan_fused":       DBSCAN_FUSED_YAML,
        "detector_cpp_fused": DETECTOR_CPP_YAML,
    }
    yaml_file = yaml_map.get(node)
    if not yaml_file or not yaml_file.exists():
        return False

    try:
        with open(yaml_file) as f:
            data = yaml.safe_load(f) or {}

        # Find the node key in the YAML
        for top_key in data:
            if "ros__parameters" in data[top_key]:
                data[top_key]["ros__parameters"][param] = value
                break

        with open(yaml_file, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
        return True
    except Exception:
        return False


# ── HTML Template ──────────────────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>LiDAR Parameter Tuner</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet"/>
<style>
  :root {
    --bg:       #0a0e17;
    --surface:  #111827;
    --border:   #1e2d45;
    --accent1:  #00e5ff;
    --accent2:  #ff6b35;
    --accent3:  #7cfc5e;
    --text:     #e2eaf4;
    --muted:    #5a7a9a;
    --ok:       #7cfc5e;
    --err:      #ff4d6d;
    --mono:     'JetBrains Mono', monospace;
    --sans:     'DM Sans', sans-serif;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    min-height: 100vh;
    padding: 0 0 60px 0;
  }

  /* Header */
  header {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 20px 40px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 100;
  }

  .logo {
    font-family: var(--mono);
    font-size: 13px;
    font-weight: 700;
    color: var(--accent1);
    letter-spacing: 0.15em;
    text-transform: uppercase;
  }

  .logo span { color: var(--muted); font-weight: 400; }

  #status-bar {
    font-family: var(--mono);
    font-size: 11px;
    padding: 6px 14px;
    border-radius: 4px;
    background: #0d1f0d;
    border: 1px solid #1a3d1a;
    color: var(--ok);
    min-width: 300px;
    text-align: center;
    transition: all 0.3s;
  }

  #status-bar.error {
    background: #1f0d0d;
    border-color: #3d1a1a;
    color: var(--err);
  }

  #status-bar.idle {
    color: var(--muted);
    background: var(--bg);
    border-color: var(--border);
  }

  /* Main layout */
  main {
    max-width: 1400px;
    margin: 0 auto;
    padding: 40px 40px 0;
  }

  /* Top info bar */
  .info-bar {
    display: flex;
    gap: 16px;
    margin-bottom: 36px;
    flex-wrap: wrap;
  }

  .info-pill {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
    padding: 6px 14px;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: var(--surface);
  }

  .info-pill b { color: var(--accent1); }

  /* Section tabs */
  .tabs {
    display: flex;
    gap: 4px;
    margin-bottom: 28px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0;
  }

  .tab-btn {
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 10px 22px;
    background: none;
    border: none;
    color: var(--muted);
    cursor: pointer;
    border-bottom: 2px solid transparent;
    margin-bottom: -1px;
    transition: all 0.15s;
  }

  .tab-btn:hover  { color: var(--text); }

  .tab-btn.active {
    color: var(--accent1);
    border-bottom-color: var(--accent1);
  }

  .tab-btn.active.pcl-tab    { color: var(--accent2); border-bottom-color: var(--accent2); }
  .tab-btn.active.fusion-tab { color: var(--accent3); border-bottom-color: var(--accent3); }

  /* Tab panels */
  .tab-panel { display: none; }
  .tab-panel.active { display: block; }

  /* Parameter grid */
  .param-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
    gap: 16px;
  }

  /* Parameter card */
  .param-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px 22px;
    transition: border-color 0.2s;
    position: relative;
    overflow: hidden;
  }

  .param-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent1);
    opacity: 0;
    transition: opacity 0.2s;
  }

  .param-card.pcl-card::before    { background: var(--accent2); }
  .param-card.fusion-card::before { background: var(--accent3); }

  .param-card:hover { border-color: #2a3f5f; }
  .param-card:hover::before { opacity: 1; }
  .param-card.changed { border-color: var(--accent1); }
  .param-card.changed::before { opacity: 1; }
  .param-card.changed.pcl-card { border-color: var(--accent2); }
  .param-card.changed.fusion-card { border-color: var(--accent3); }

  .param-label {
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 700;
    color: var(--accent1);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 4px;
  }

  .param-card.pcl-card    .param-label { color: var(--accent2); }
  .param-card.fusion-card .param-label { color: var(--accent3); }

  .param-desc {
    font-size: 12px;
    color: var(--muted);
    line-height: 1.5;
    margin-bottom: 16px;
  }

  .param-controls {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  input[type="range"] {
    flex: 1;
    -webkit-appearance: none;
    height: 4px;
    border-radius: 2px;
    background: var(--border);
    outline: none;
    cursor: pointer;
  }

  input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: var(--accent1);
    cursor: pointer;
    border: 2px solid var(--bg);
    box-shadow: 0 0 6px rgba(0,229,255,0.4);
    transition: transform 0.1s;
  }

  input[type="range"]::-webkit-slider-thumb:hover {
    transform: scale(1.2);
  }

  .pcl-card   input[type="range"]::-webkit-slider-thumb { background: var(--accent2); box-shadow: 0 0 6px rgba(255,107,53,0.4); }
  .fusion-card input[type="range"]::-webkit-slider-thumb { background: var(--accent3); box-shadow: 0 0 6px rgba(124,252,94,0.4); }

  input[type="number"] {
    width: 80px;
    background: var(--bg);
    border: 1px solid var(--border);
    color: var(--text);
    font-family: var(--mono);
    font-size: 13px;
    font-weight: 600;
    padding: 6px 10px;
    border-radius: 4px;
    text-align: right;
    outline: none;
    transition: border-color 0.15s;
  }

  input[type="number"]:focus { border-color: var(--accent1); }
  .pcl-card   input[type="number"]:focus { border-color: var(--accent2); }
  .fusion-card input[type="number"]:focus { border-color: var(--accent3); }

  /* Apply button per card */
  .apply-btn {
    font-family: var(--mono);
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 6px 14px;
    border-radius: 4px;
    border: none;
    cursor: pointer;
    background: rgba(0,229,255,0.1);
    color: var(--accent1);
    border: 1px solid rgba(0,229,255,0.2);
    transition: all 0.15s;
    margin-top: 12px;
    width: 100%;
  }

  .apply-btn:hover {
    background: rgba(0,229,255,0.2);
    border-color: var(--accent1);
  }

  .apply-btn:active { transform: scale(0.98); }

  .pcl-card .apply-btn {
    background: rgba(255,107,53,0.1);
    color: var(--accent2);
    border-color: rgba(255,107,53,0.2);
  }
  .pcl-card .apply-btn:hover {
    background: rgba(255,107,53,0.2);
    border-color: var(--accent2);
  }

  .fusion-card .apply-btn {
    background: rgba(124,252,94,0.1);
    color: var(--accent3);
    border-color: rgba(124,252,94,0.2);
  }
  .fusion-card .apply-btn:hover {
    background: rgba(124,252,94,0.2);
    border-color: var(--accent3);
  }

  /* Footer actions */
  .footer-actions {
    position: fixed;
    bottom: 0; left: 0; right: 0;
    background: var(--surface);
    border-top: 1px solid var(--border);
    padding: 14px 40px;
    display: flex;
    gap: 12px;
    align-items: center;
    z-index: 100;
  }

  .footer-actions span {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
    margin-right: 8px;
  }

  .action-btn {
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 8px 20px;
    border-radius: 4px;
    border: none;
    cursor: pointer;
    transition: all 0.15s;
  }

  .btn-apply-all {
    background: var(--accent1);
    color: var(--bg);
  }
  .btn-apply-all:hover { filter: brightness(1.1); }

  .btn-save {
    background: none;
    color: var(--accent3);
    border: 1px solid var(--accent3);
  }
  .btn-save:hover { background: rgba(124,252,94,0.1); }

  .btn-refresh {
    background: none;
    color: var(--muted);
    border: 1px solid var(--border);
  }
  .btn-refresh:hover { color: var(--text); border-color: var(--text); }

  /* Node status indicators */
  .node-status {
    display: flex;
    gap: 10px;
    align-items: center;
    margin-left: auto;
  }

  .node-dot {
    display: flex;
    align-items: center;
    gap: 6px;
    font-family: var(--mono);
    font-size: 10px;
    color: var(--muted);
  }

  .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--err);
  }

  .dot.online { background: var(--ok); box-shadow: 0 0 4px var(--ok); }

  /* Current value badge */
  .current-val {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--muted);
    margin-top: 8px;
  }

  .current-val span { color: var(--text); }
</style>
</head>
<body>

<header>
  <div class="logo">NOVA // LiDAR <span>Parameter Tuner</span></div>
  <div id="status-bar" class="idle">Ready — adjust a parameter to begin</div>
</header>

<main>
  <div class="info-bar">
    <div class="info-pill">ROS2 Jazzy</div>
    <div class="info-pill">Target: <b>base_link</b></div>
    <div class="info-pill">Fusion: <b>/nova/cloud_fused</b></div>
    <div class="info-pill">Changes apply live — no restart needed</div>
  </div>

  <div class="tabs">
    <button class="tab-btn active" onclick="switchTab('dbscan', this)">DBSCAN Detector</button>
    <button class="tab-btn pcl-tab" onclick="switchTab('pcl', this)">PCL Detector</button>
    <button class="tab-btn fusion-tab" onclick="switchTab('fusion', this)">Fusion</button>
  </div>

  <div id="tab-dbscan" class="tab-panel active">
    <div class="param-grid" id="grid-dbscan"></div>
  </div>
  <div id="tab-pcl" class="tab-panel">
    <div class="param-grid" id="grid-pcl"></div>
  </div>
  <div id="tab-fusion" class="tab-panel">
    <div class="param-grid" id="grid-fusion"></div>
  </div>
</main>

<div class="footer-actions">
  <span>Actions:</span>
  <button class="action-btn btn-apply-all" onclick="applyAll()">Apply All Changes</button>
  <button class="action-btn btn-save" onclick="saveAll()">Save to YAML</button>
  <button class="action-btn btn-refresh" onclick="refreshValues()">Refresh from Nodes</button>
  <div class="node-status" id="node-status"></div>
</div>

<script>
const PARAMS = {{ params_json|safe }};
const pendingChanges = {};

function setStatus(msg, type='ok') {
  const el = document.getElementById('status-bar');
  el.textContent = msg;
  el.className = type === 'error' ? 'error' : type === 'idle' ? 'idle' : '';
}

function buildGrid(group) {
  const grid = document.getElementById('grid-' + group);
  const params = PARAMS[group];
  grid.innerHTML = '';

  params.forEach(p => {
    const key = p.node + '__' + p.param;
    const cardClass = group === 'pcl' ? 'pcl-card' : group === 'fusion' ? 'fusion-card' : '';

    const card = document.createElement('div');
    card.className = 'param-card ' + cardClass;
    card.id = 'card-' + key;

    card.innerHTML = `
      <div class="param-label">${p.label}</div>
      <div class="param-desc">${p.desc}</div>
      <div class="param-controls">
        <input type="range" id="slider-${key}"
               min="${p.min}" max="${p.max}" step="${p.step}"
               value="${p.min}"
               oninput="onSlider('${key}', '${p.node}', '${p.param}', this.value, ${p.step})"/>
        <input type="number" id="num-${key}"
               min="${p.min}" max="${p.max}" step="${p.step}"
               value="${p.min}"
               oninput="onNumber('${key}', '${p.node}', '${p.param}', this.value)"/>
      </div>
      <div class="current-val" id="cur-${key}">Current: <span>—</span></div>
      <button class="apply-btn" onclick="applySingle('${p.node}', '${p.param}', '${key}')">
        Apply ↗
      </button>
    `;
    grid.appendChild(card);
  });
}

function onSlider(key, node, param, val, step) {
  const v = step < 1 ? parseFloat(parseFloat(val).toFixed(3)) : parseInt(val);
  document.getElementById('num-' + key).value = v;
  pendingChanges[key] = { node, param, value: v };
  document.getElementById('card-' + key).classList.add('changed');
}

function onNumber(key, node, param, val) {
  const v = parseFloat(val);
  if (isNaN(v)) return;
  document.getElementById('slider-' + key).value = v;
  pendingChanges[key] = { node, param, value: v };
  document.getElementById('card-' + key).classList.add('changed');
}

async function applySingle(node, param, key) {
  const change = pendingChanges[key];
  if (!change) { setStatus('No change pending for ' + param, 'idle'); return; }

  setStatus('Setting /' + node + ' ' + param + ' = ' + change.value + ' ...', 'idle');

  const res = await fetch('/set_param', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ node: change.node, param: change.param, value: change.value })
  });
  const data = await res.json();

  if (data.ok) {
    setStatus('✓ ' + data.msg);
    document.getElementById('card-' + key).classList.remove('changed');
    document.getElementById('cur-' + key).innerHTML = 'Current: <span>' + change.value + '</span>';
    delete pendingChanges[key];
  } else {
    setStatus('✗ ' + data.msg, 'error');
  }
}

async function applyAll() {
  const keys = Object.keys(pendingChanges);
  if (keys.length === 0) { setStatus('No pending changes', 'idle'); return; }

  setStatus('Applying ' + keys.length + ' changes...', 'idle');
  let ok = 0, fail = 0;

  for (const key of keys) {
    const c = pendingChanges[key];
    const res = await fetch('/set_param', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ node: c.node, param: c.param, value: c.value })
    });
    const data = await res.json();
    if (data.ok) {
      ok++;
      document.getElementById('card-' + key).classList.remove('changed');
      document.getElementById('cur-' + key).innerHTML = 'Current: <span>' + c.value + '</span>';
      delete pendingChanges[key];
    } else {
      fail++;
    }
  }
  setStatus('✓ Applied ' + ok + ' params' + (fail ? ' — ' + fail + ' failed' : ''), fail ? 'error' : 'ok');
}

async function saveAll() {
  setStatus('Saving to YAML files...', 'idle');
  const res = await fetch('/save_yaml', { method: 'POST' });
  const data = await res.json();
  setStatus(data.ok ? '✓ Saved to YAML — rebuild to make permanent' : '✗ ' + data.msg, data.ok ? 'ok' : 'error');
}

async function refreshValues() {
  setStatus('Fetching current values from nodes...', 'idle');
  const res = await fetch('/get_values');
  const data = await res.json();

  let found = 0;
  for (const [key, val] of Object.entries(data)) {
    if (val !== null && val !== undefined) {
      const slider = document.getElementById('slider-' + key);
      const num    = document.getElementById('num-' + key);
      const cur    = document.getElementById('cur-' + key);
      if (slider) { slider.value = val; num.value = val; }
      if (cur) cur.innerHTML = 'Current: <span>' + val + '</span>';
      found++;
    }
  }
  setStatus(found > 0 ? '✓ Loaded ' + found + ' values from running nodes' : 'No running nodes found — start your launch file first', found > 0 ? 'ok' : 'error');
  updateNodeStatus(data);
}

function updateNodeStatus(data) {
  const nodes = ['cloud_fusion', 'dbscan_fused', 'detector_cpp_fused'];
  const labels = { cloud_fusion: 'Fusion', dbscan_fused: 'DBSCAN', detector_cpp_fused: 'PCL' };
  const bar = document.getElementById('node-status');
  bar.innerHTML = nodes.map(n => {
    const online = Object.entries(data).some(([k, v]) => k.startsWith(n + '__') && v !== null);
    return `<div class="node-dot"><div class="dot ${online ? 'online' : ''}"></div>${labels[n]}</div>`;
  }).join('');
}

function switchTab(group, btn) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + group).classList.add('active');
  btn.classList.add('active');
}

// Build all grids on load
buildGrid('dbscan');
buildGrid('pcl');
buildGrid('fusion');
refreshValues();
</script>

</body>
</html>
"""


# ── Flask routes ───────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML, params_json=json.dumps(PARAMS))


@app.route("/set_param", methods=["POST"])
def set_param():
    data  = request.get_json()
    node  = data.get("node", "")
    param = data.get("param", "")
    value = data.get("value")

    if not node or not param or value is None:
        return jsonify({"ok": False, "msg": "Missing node/param/value"})

    result = ros2_param_set(node, param, value)
    return jsonify(result)


@app.route("/get_values")
def get_values():
    values = get_all_current_values()
    return jsonify(values)


@app.route("/save_yaml", methods=["POST"])
def save_yaml():
    """Save all current slider values back to YAML files."""
    saved = 0
    for group, params in PARAMS.items():
        for p in params:
            val = ros2_param_get(p["node"], p["param"])
            if val is not None:
                if save_to_yaml(group, p["node"], p["param"], val):
                    saved += 1

    if saved > 0:
        return jsonify({"ok": True, "msg": f"Saved {saved} parameters to YAML files"})
    else:
        return jsonify({"ok": False, "msg": "Could not save — check YAML file paths"})


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  LiDAR Parameter Tuner")
    print("="*60)
    print(f"  Config dir : {CONFIG_DIR}")
    print(f"  Open       : http://localhost:5000")
    print("="*60)
    print("  Make sure your launch file is running first!")
    print("  Changes apply live via 'ros2 param set'")
    print("="*60 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
