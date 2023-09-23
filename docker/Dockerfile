# syntax=docker/dockerfile:1

FROM ros:humble 

RUN mkdir /microros_ws  && \
    cd /microros_ws && \
    git clone -b $ROS_DISTRO https://github.com/micro-ROS/micro_ros_setup.git src/micro_ros_setup

WORKDIR /microros_ws


RUN . /opt/ros/${ROS_DISTRO}/setup.sh && \
    apt update && \
    apt install -y python3-pip && \
    rosdep update && \
    rosdep install --from-paths src --ignore-src -y && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install  scikit-learn numpy

RUN . /opt/ros/${ROS_DISTRO}/setup.sh && colcon build && . install/local_setup.sh && ros2 run micro_ros_setup create_agent_ws.sh && ros2 run micro_ros_setup build_agent.sh

