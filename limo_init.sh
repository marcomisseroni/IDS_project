#!/bin/bash

LIMO0=192.168.1.37
LIMO1=192.168.1.35
LIMO2=192.168.1.34


# Check the working directory

REPO_ROOT="$(git rev-parse --show-toplevel)"

cd "$REPO_ROOT" || {
  echo "ERROR: not inside a git repo"
  exit 1
}

cd ros2_ws

# Build the project

colcon build
source install/setup.bash

cd launch

ros2 launch launch_multiple.py &

# Activate camera, odometry and compression for:

#Limo0
ssh agilex@LIMO0 "
ros2 launch orbbec_camera dabai.launch.py camera_name:=$NAMESPACE color_fps:=10 &

ros2 run image_transport republish raw compressed \
  --ros-args \
  -r in:=/$NAMESPACE/color/image_raw \
  -r out/compressed:=/$NAMESPACE/compressed &

namespace_launch &

wait
" &
#Limo1
ssh agilex@LIMO1 "
ros2 launch orbbec_camera dabai.launch.py camera_name:=$NAMESPACE color_fps:=10 &

ros2 run image_transport republish raw compressed \
  --ros-args \
  -r in:=/$NAMESPACE/color/image_raw \
  -r out/compressed:=/$NAMESPACE/compressed &

namespace_launch &

wait
" &
#Limo2
ssh agilex@LIMO2 "
ros2 launch orbbec_camera dabai.launch.py camera_name:=$NAMESPACE color_fps:=10 &

ros2 run image_transport republish raw compressed \
  --ros-args \
  -r in:=/$NAMESPACE/color/image_raw \
  -r out/compressed:=/$NAMESPACE/compressed &

namespace_launch &

wait
" &

wait