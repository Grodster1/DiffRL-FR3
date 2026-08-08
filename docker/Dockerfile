FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# ---- System dependencies ------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-desktop \
    ros-jazzy-ros-gz \
    ros-jazzy-gz-ros2-control \
    ros-jazzy-moveit \
    build-essential cmake git curl wget \
    python3-pip python3-venv python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# ---- Python ML dependencies ---------------------------------
RUN pip3 install --break-system-packages --no-cache-dir \
    torch torchvision

RUN pip3 install --break-system-packages --no-cache-dir \
    stable-baselines3 gymnasium diffusers wandb matplotlib pandas pyyaml scipy

RUN pip3 install --break-system-packages --no-cache-dir "setuptools<81" \
    && python3 -c "import pkg_resources; print('pkg_resources OK')"

# ---- Franka ROS 2 (build from source for Jazzy) -------------
WORKDIR /opt/franka_ws/src

RUN git clone https://github.com/frankarobotics/franka_description.git
RUN git clone https://github.com/frankarobotics/franka_ros2.git -b jazzy

WORKDIR /opt/franka_ws
RUN source /opt/ros/jazzy/setup.bash && \
    apt-get update && rosdep update --rosdistro=jazzy && \
    rosdep install --from-paths src --ignore-src -r -y --skip-keys="Franka libfranka" && \
    rm -rf /var/lib/apt/lists/* && \
    colcon build --symlink-install \
      --cmake-args -DCMAKE_BUILD_TYPE=Release \
      --packages-select franka_description franka_msgs

# ---- Env + auto-source --------------------------------------
ENV GZ_SIM_RESOURCE_PATH=/opt/franka_ws/install/franka_description/share
RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc && \
    echo "source /opt/franka_ws/install/setup.bash" >> /root/.bashrc && \
    echo "[ -f /ws/install/setup.bash ] && source /ws/install/setup.bash" >> /root/.bashrc

# ---- Workspace ----------------------------------------------
# Na razie tylko tworzymy katalog — pakiety montujemy przez volumes.
# Gdy package.xml gotowe, odkomentuj poniższe:
WORKDIR /ws
# COPY src/ src/
# RUN source /opt/ros/jazzy/setup.bash && \
#     source /opt/franka_ws/install/setup.bash && \
#     rosdep install --from-paths src --ignore-src -r -y && \
#     colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

# ---- Entrypoint ----------------------------------------------
COPY docker/ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]