FROM ros:humble

WORKDIR /IDS_project

RUN apt update && apt install -y python3-pip && \
    rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip3 install -r requirements.txt

WORKDIR /IDS_project/ros2_ws

RUN /bin/bash -c "source /opt/ros/humble/setup.bash && colcon build"

RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "source /IDS_project/ros2_ws/install/setup.bash" >> /root/.bashrc

CMD ["/bin/bash"]