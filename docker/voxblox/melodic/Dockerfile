From osrf/ros:melodic-desktop-full

#LABEL com.nvidia.volumes.needed="nvidia_driver"
#ENV PATH /usr/local/nvidia/bin:${PATH}
#ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}


ENV DEBIAN_FRONTEND noninteractive

RUN apt update && apt-get install python-wstool python-catkin-tools ros-melodic-cmake-modules protobuf-compiler autoconf libtool vim tmux  wget -y

#Run mkdir -p ~/catkin_ws/src && \
#cd ~/catkin_ws && \
#catkin init && \
#catkin config --extend /opt/ros/kinetic && \
#catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release && \
#catkin config --merge-devel

#WORKDIR "~/catkin_ws/src"
#Run wstool init . ./voxblox/voxblox_ssh.rosinstall && \
#wstool update &&\
#catkin build voxblox_ros

