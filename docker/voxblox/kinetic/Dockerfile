From osrf/ros:kinetic-desktop-full

#LABEL com.nvidia.volumes.needed="nvidia_driver"
#ENV PATH /usr/local/nvidia/bin:${PATH}
#ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}


ENV DEBIAN_FRONTEND noninteractive

RUN apt update && apt-get install python-wstool python-catkin-tools ros-kinetic-cmake-modules protobuf-compiler autoconf libtool vim tmux  wget -y
RUN apt-get install python-wstool python-catkin-tools ros-kinetic-cmake-modules protobuf-compiler autoconf libprotobuf-dev protobuf-c-compiler -y

RUN apt update && apt-get install sudo --upgrade 

#RUN wget https://ompl.kavrakilab.org/install-ompl-ubuntu.sh && chmod u+x ./install-ompl-ubuntu.sh &&SUDO=sudo ./install-ompl-ubuntu.sh --app

ADD ./install-ompl-ubuntu.sh /install-ompl-ubuntu.sh
RUN chmod u+x ./install-ompl-ubuntu.sh && ./install-ompl-ubuntu.sh --app

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
RUN ln -s /usr/local/include/ompl-1.5/ompl /usr/include/ && ln -s /usr/local/lib/libompl* /usr/lib/

