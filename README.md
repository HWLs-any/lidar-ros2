# lidar_ws

## Description
ROS 2 workspace for lidar processing. Includes:

- `lidar_py_pkg`: Python ROS 2 nodes for Open3D-based point cloud processing
- `lidar_description`: Contain URDF/xacro file
- `lidar_bringup`: necessary files to launch multiple nodes
- SICK lidar driver (`sick_scan_xd`) included as a git submodule

This workspace is fully reproducible with pipenv and supports Ubuntu 24.04 + ROS 2 Jazzy + Python 3.12.

---

## Super compact one-block setup script

Copy and paste the following in the terminal to set up the workspace, install dependencies, and build:

```bash
# Clone repository with submodules
mkdir lidar_ws
cd lidar_ws
git clone --recurse-submodules git@github.com:HWLs-any/lidar_ros2.git

# Install pipenv if not present
python3 -m pip install --user pipenv
export PATH="$HOME/.local/bin:$PATH"

# Install Python dependencies
pipenv install

# Source ROS 2 environment
source /opt/ros/jazzy/setup.bash

# Build workspace with symlink install
pipenv run colcon build --symlink-install
source install/setup.bash

# Verify lidar_py_pkg node exists
ros2 pkg executables lidar_py_pkg
```
After running this block, the workspace is ready to run ROS 2 nodes.

Step-by-step setup instructions
1. Clone repository with submodules
```
git clone --recurse-submodules git@github.com:HWLs-any/Lidar_ros2.git
```

If cloned without --recurse-submodules:
```
git submodule update --init --recursive
```

2. Install pipenv
```
cd lidar_ws
python3 -m pip install --user pipenv
export PATH="$HOME/.local/bin:$PATH"
```

3. Install Python dependencies
```
pipenv install numpy scikit-learn open3d colcon-common-extensions
```
Installs numpy, scikit-learn, open3d, colcon-common-extensions
Virtual environment is created in ~/.local/share/virtualenvs/...

4. Activate environment and ROS 2
Option A — Using direnv (if .envrc exists)
```
direnv allow
```

Option B — Manual activation
```
source /opt/ros/jazzy/setup.bash
pipenv shell
```
Python interpreter points to the pipenv environment
ROS 2 environment is sourced

5. Build workspace with symlink install
```
cd ~/lidar_ws
pipenv run colcon build --symlink-install
source install/setup.bash
```
--symlink-install ensures Python changes are reflected immediately

6. Verify ROS 2 packages
```
ros2 pkg list | grep lidar_py_pkg
ros2 pkg executables lidar_py_pkg
```
lidar_py_pkg and node plane_remove_o3d should be listed


7. Run a ROS 2 node
```
ros2 run lidar_py_pkg plane_remove_o3d \
  --ros-args -p input_topic:=/nova/cloud_fused \
             -p output_topic:=/nova/cloud_ground_removed \
             -p distance_thresh:=0.05
```
Node subscribes to /nova/cloud_fused and publishes filtered points to /nova/cloud_ground_removed

8. Run SICK lidar driver(test)
```
ros2 run sick_scan_xd sick_generic_node
```
Ensure SICK lidar is connected
Submodules are initialized and updated

9. Update workspace later
```
git pull
git submodule update --init --recursive
pipenv run colcon build --symlink-install
source install/setup.bash
```
Pulls updates from GitHub, updates submodules, rebuilds workspace
