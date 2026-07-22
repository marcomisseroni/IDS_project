#!/bin/bash

DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")

cd "$DIR/ros2_ws"

source install/setup.bash

trap 'kill -TERM -- -$$ 2>/dev/null; sleep 1; kill -9 -- -$$ 2>/dev/null; exit 1' SIGINT

ros2 launch launch/launch_multiple.py &
ros2 run limo_control measurement_router &

wait