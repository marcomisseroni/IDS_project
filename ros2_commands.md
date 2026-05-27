# Useful commands for **ros2 humble**

## Source the local setup file

For each new terminal always source the setup file:

```
source /opt/ros/humble/setup.bash
```

Otherwise you can add this line to your bashrc file.

## Create a new package

A package is a collection of nodes or other types of file such as **urdf** or **xacro**.
To create a **python** package, navigate to the directory `ros2_ws/src`, then you can use the command:

```
ros2 pkg create --build-type ament_python <package_name>
```

Instead to create a **C++** package you can use the command:

```
ros2 pkg create --build-type ament_cmake <package_name>
```

You can also add before `<package_name>` the flag `--node-name` followed by `<node_name>` in order to create
a new package containing a new node. 

## Create a new node 

To create a new node you can just go to the directory `ros2_ws/src/package_name/package_name` and create
a new file `new_node.py` or `new_node.cpp`.
After creating the new node you have to modify few files.

### Add dependencies

Navigate to `ros2_ws/src/package`.

Open `package.xml` with a text editor.
Fill the fields: `<description>`, `<maintainer>` and `<licence>`. You will see a part like:

```
<description>Examples of minimal publisher/subscriber using rclpy</description>
<maintainer email="you@email.com">Your Name</maintainer>
<license>Apache License 2.0</license>
```

After the lines above we have to add all the dependencies corresponding to your 
node's import statements, for example:

```
<exec_depend>rclpy</exec_depend>
<exec_depend>std_msgs</exec_depend>
```

This declares the packages needs `rclpy` and `std_msgs` when the code is executed.

### Add an entry point

Open the `setup.py` file. Again match the `<maintainer>`, `<maintainer_emai>`, `<description>` and `<licence>` fields
to your `package.xml`.

```
maintainer='YourName',
maintainer_email='you@email.com',
description='Examples of minimal publisher/subscriber using rclpy',
license='Apache License 2.0',
```

Add the following line within the `console_scripts` brackets of the `entry_points` field:

```
entry_points={
        'console_scripts': [
                'executable_name = package_name.module_name:function_name',
        ],
},
```

Where `executable_name` can be chosen, `module_name` is the name of the node and `function_name`
is the function to be executed, usually `main`.

### Check setup.cfg

The content of this file should already be populated correctly like:

```
[develop]
script-dir=$base/lib/package_name
[install]
install-scripts=$base/lib/package_name
```

## Build a package

To build a specific package, navigate to `ros2_ws` then you can use:

```
colcon build --packages-select <package_name>
```

Otherwise to build everything just use:

```
colcon build
```

After building one or more packages don't forget to source: `source install/setup.bash`

## Run a node

Navigate to the workspace folder `ros2_ws` and run: `ros2 run package_name node_name`.

## Run a launch file

Navigate to the launch directory and use `ros2 launch launch_file.py`.
If the launch file is provided by a package then use `ros2 launch <package_name> <launch_file_name>`

Actual launch file command
```
ros2 launch launch/launch_file.py
```

## Useful terminal commands

To read what is published on a topic from the terminal you can use: `ros2 topic echo /topic_name`.

To publish onto a topic: `ros2 topic pub /topic_name data_type data`. Note that `data_type` is a ros2 type such as: `std_msgs/msg/Bool` and `data` must be written in YAML syntax.
An example is:

```
ros2 topic pub /test std_msgs/msg/Bool "data: false"
```

Another example, to start the looping the `EKF_node.py` you can use:

```
ros2 topic pub --once /admin std_msgs/msg/String "{data: 'start_ekf'}"
```
It can be also used to stop such node with: `start_ekf`. Similarly for the `MPC_node.py` you can use: `start_mpc` and `stop_mpc`.

## Commands for LIMO

Connect to the wifi OptiTrack.speed with the password `60A84A244BECD`

To access the LIMO using SSH protocol you can use: `ssh agilex@192.168.1.34` where the last number could be different. The password is: `agx`. 

To work on the same domain of the LIMO with your own computer you can check the *ROS DOMAIN* with: `echo $ROS_DOMAIN_ID` on the LIMO and then set the same *ROS DOMAIN* on your computer using:  `export ROS_DOMAIN_ID=2`.

To launch the camera node on the LIMO you can use: `ros2 launch orbbec_camera dabai.launch.py`.
To launch the odometry node: `ros2 launch limo_bringup limo_start.launch.py`.

To launch the nodes with a namespace: `ros2 launch orbbec_camera dabai.launch.py \ --ros-args -r __ns:=/limo0`

To run the image compression node on the limo: `ros2 run image_transport republish raw compressed --ros-args --remap in:=/camera/color/image_raw`.
`ros2 run image_transport republish raw compressedDepth  --ros-args  -r in:=/camera/depth/image_raw`

It's possible to play a pre-recorded *bag play*, which replay all the messages shared on all the corresponding topics using `ros2 bag play <bag_name>`.

### Firewall deactivation for windows users

In the settings look for `DEACTIVATE WINDOWS FIREWALL FOR PRIVATE NETWORKS` then go to `privacy e sicurezza -> sicurezza di windows -> firewall e protezione di rete -> reti private -> disattiva`.

## Multi limo commands

To launch the camera with the namespace: `ros2 launch orbbec_camera dabai.launch.py camera_name:=limo1`

To launch the camera compression with the namespace: `ros2 run image_transport republish raw compressed --ros-args -r in:=/limo1/color/image_raw -r out/compressed:=/limo1/compressed`

To launch the odometry with namespace: ` `