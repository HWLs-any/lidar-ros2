# Multi-LiDAR Perception Pipeline (ROS 2)

## Main Goal
Build a **reproducible, modular ROS 2 perception pipeline** for a **fixed-mounted multi-LiDAR setup**, enabling:
- **Time/TF-aligned fusion** of multiple LiDAR point clouds into a common frame
- **Object detection (clustering)** on both **per-sensor** and **fused** point clouds
- **Clear RViz visualization** for qualitative evaluation and debugging

---

## Requirements
- **Ubuntu 24.04**
- **ROS 2 Jazzy**
- **Python 3.12**
- **colcon**
- **pipx**
- **pipenv**
- **PCL (system packages)**
- **sick_scan_xd (for live Lidar data)**


---

## Install / Clone

```bash
#update + basic tools
sudo apt update
sudo apt upgrade
sudo apt install -y git curl build-essential python3 python3-pip python3-venv python3-dev

#pipx (recommended way to install Python CLI tools globally)
sudo apt install -y pipx
pipx ensurepath
#Close/reopen your terminal OR run:
source ~/.bashrc

#Install pipenv globally via pipx
pipx install pipenv
pipenv --version

#Optional but recommended: direnv
sudo apt install -y direnv
#Enable direnv for bash:
grep -q 'direnv hook bash' ~/.bashrc || echo 'eval "$(direnv hook bash)"' >> ~/.bashrc
source ~/.bashrc
#Make pipenv create .venv inside the project (recommended)
grep -q 'PIPENV_VENV_IN_PROJECT=1' ~/.bashrc || echo 'export PIPENV_VENV_IN_PROJECT=1' >> ~/.bashrc
source ~/.bashrc

#If ROS 2 Jazzy is installed, this should work:
source /opt/ros/jazzy/setup.bash
echo $ROS_DISTRO    #Expected: jazzy
#install common build helpers (safe to do):
sudo apt install -y python3-colcon-common-extensions python3-rosdep
#Initialize rosdep (one-time):
sudo rosdep init || true
rosdep update

#Clone repo WITH submodules
mkdir -p ~/lidar_ws 
cd ~/lidar_ws          

git clone --recurse-submodules git@github.com:HWLs-any/lidar-ros2.git .
cd lidar_ws

#Only use this, if you forgot to clone with submodule
git submodule update --init --recursive

#You have cloned repo sucessfully


#If you use direnv: create/enable .envrc
cd ~/lidar_ws          
cat > .envrc <<'EOF'
# ROS 2 Jazzy
source /opt/ros/jazzy/setup.bash

# Project-local pipenv venv
export PIPENV_VENV_IN_PROJECT=1

# Optional overlay (after first build)
if [ -f install/setup.bash ]; then
  source install/setup.bash
fi
EOF

direnv allow

#Create the pipenv virtualenv (Python 3.12)
pipenv --python 3.12

#Install Python libraries needed by nodes

#These are the ones I’ve been using in code:
pipenv install numpy scikit-learn open3d

#Important for ROS builds inside pipenv (I hit these missing error before):
pipenv install empy catkin_pkg


#Install ROS package dependencies (rosdep)
cd ~/lidar_ws
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -r -y


#Build (always through pipenv)
cd ~/lidar_ws          #always build from here
pipenv run colcon build --symlink-install
source install/setup.bash


#Sanity checks:
pipenv run ros2 pkg list | grep -E "lidar_|sick_scan_xd"
pipenv run ros2 pkg executables lidar_py_pkg
pipenv run ros2 pkg executables lidar_cpp_pkg

#Run (example)
#Example launch (adjust to your actual file names):
pipenv run ros2 launch lidar_bringup live_sensors_rviz.launch.xml

````
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
├── results/                   # Store result metrics
└── bags/                      # (optional) ros2 bag recordings (not required for build)

````
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
