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
colcon buld --package-select <package_name>
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

## Useful terminal commands

To read what is published on a topic from the terminal you can use: `ros2 topic echo /topic_name`.

To publish onto a topic: `ros2 topic pub /topic_name data`