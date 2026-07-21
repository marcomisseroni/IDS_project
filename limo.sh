#!/bin/bash

source /opt/ros/foxy/setup.bash
source ~/.bashrc

ros2 launch orbbec_camera dabai.launch.py camera_name:=$NAMESPACE color_fps:=10 &

CAM_PID=$!

ros2 launch motion_utils_ros limo_namespace_start.launch.py namespace:=$NAMESPACE &

NS_PID=$!

cd ros2_ws
source install/setup.bash
cd ..

ros2 run vision vision_node $NAMESPACE_NUMBER &

VISION_PID=$!

trap "kill $CAM_PID $NS_PID $VISION_PID 2>/dev/null" SIGINT

wait
