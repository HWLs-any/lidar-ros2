# Multi-LiDAR Perception Pipeline (ROS 2)

## Main Goal
Build a **reproducible, modular ROS 2 perception pipeline** for a **fixed-mounted multi-LiDAR setup**, enabling:
- **Time/TF-aligned fusion** of multiple LiDAR point clouds into a common frame
- **Object detection (clustering)** on both **per-sensor** and **fused** point clouds
- **Clear RViz visualization** for qualitative evaluation and debugging

---

## What the Code Does (Current Status)

### Input
- Four independent `PointCloud2` topics:
  - `/front/cloud`, `/right/cloud`, `/back/cloud`, `/left/cloud`

### Core Pipeline
1. **TF / Frames**
   - `lidar_description` provides a URDF/Xacro with fixed sensor frames
   - `robot_state_publisher` publishes TF for consistent transforms

2. **Fusion (Python) — `lidar_py_pkg/cloud_fusion`**
   - Subscribes to all four point cloud topics
   - Transforms each cloud into a shared `target_frame` (default: `base_link`)
   - Applies voxel-based de-duplication
   - Publishes the fused cloud:
     - `/nova/cloud_fused`

3. **Object Detection / Clustering (two algorithms)**
   - **C++ / PCL — `lidar_cpp_pkg/object_detector_pcl`**
     - Voxel downsampling
     - Statistical outlier removal
     - Plane segmentation (RANSAC) for ground removal
     - Euclidean clustering
     - Publishes `MarkerArray` bounding boxes for RViz
     - Supports **metrics output** (latency, detections count, etc.)
   - **Python / DBSCAN — `lidar_py_pkg/cloud_detector`**
     - Optional voxel downsample
     - DBSCAN clustering
     - Publishes `MarkerArray` bounding boxes for RViz

### Visualization
- RViz shows:
  - Per-sensor point clouds
  - Fused cloud
  - Bounding-box markers from C++ and Python detectors
  - TF frames

---

## Repository Structure
```

lidar_ws/
├── Pipfile
├── Pipfile.lock
├── src/
│   ├── lidar_py_pkg/          # Python ROS 2 nodes (fusion, DBSCAN detector, utilities)
│   ├── lidar_cpp_pkg/         # C++ ROS 2 nodes (PCL detector + metrics)
│   ├── lidar_description/     # URDF/Xacro (frames)
│   ├── lidar_bringup/         # XML launch files
│   └── sick_scan_xd/          # (optional) SICK driver submodule for live operation
├── rviz/                      # Saved RViz configs
└── bags/                      # (optional) ros2 bag recordings (not required for build)

````

---

## Requirements
- **Ubuntu 24.04**
- **ROS 2 Jazzy**
- **Python 3.12**
- **colcon**
- **pipenv**
- **PCL (system packages)**
- **sick_scan_xd (for live Lidar data)**


---

## Install / Clone

### 1) Install system dependencies
```bash
sudo apt update
sudo apt install -y \
  python3-pip python3-venv \
  direnv \
  git \
  python3-colcon-common-extensions \
  ros-jazzy-robot-state-publisher \
  ros-jazzy-tf2-ros ros-jazzy-tf2-sensor-msgs \
  ros-jazzy-pcl-conversions \
  libpcl-dev
````
Most of them are already comes with ros-jazzy standard installation

### 2) Install pipenv

```bash
python3 -m pip install --user pipenv
```

(Optional but recommended: keep venv inside the repo)

```bash
echo 'export PIPENV_VENV_IN_PROJECT=1' >> ~/.bashrc
source ~/.bashrc
```

### 3) Clone the repo

```bash
cd ~
git clone git@github.com:HWLs-any/lidar-ros2.git
cd lidar-ros2
```

If you use `sick_scan_xd` as a submodule (for live sensors):

```bash
git submodule update --init --recursive
```

---

## Python Environment (pipenv)

Create / install dependencies from `Pipfile.lock`:

```bash
cd ~/lidar-ros2
pipenv sync --dev
```

---

## Build (ROS 2 + pipenv)

```bash
cd ~/lidar-ros2
source /opt/ros/jazzy/setup.bash
pipenv run colcon build --symlink-install
source install/setup.bash
```

---

## Run with ROS 2 bag playback

After building and sourcing, run:

```bash
#Replay
pipenv run ros2 launch lidar_bringup live_fusion_detection.launch.xml \
  play_bag:=true \
  use_sim_time:=true

#Live
pipenv run ros2 launch lidar_bringup live_fusion_detection.launch.xml \
  play_bag:=false \
  use_sim_time:=false

```

You can change `bag_path` to any other recorded bag location.

