# From osrf/ros:kinetic-desktop-full
From nvidia/cuda:10.2-cudnn8-devel-ubuntu16.04

ENV DEBIAN_FRONTEND noninteractive


# install packages
RUN apt-get update && apt-get install -q -y --no-install-recommends \
    dirmngr \
    gnupg2 \
    && rm -rf /var/lib/apt/lists/*

# setup keys
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# setup sources.list
RUN echo "deb http://packages.ros.org/ros/ubuntu xenial main" > /etc/apt/sources.list.d/ros1-latest.list

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ENV ROS_DISTRO kinetic

# install ros packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-kinetic-ros-core=1.3.2-0* \
    && rm -rf /var/lib/apt/lists/*


# install bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    python-rosdep \
    python-rosinstall \
    python-vcstools \
    && rm -rf /var/lib/apt/lists/*

# bootstrap rosdep
RUN rosdep init && \
  rosdep update --rosdistro $ROS_DISTRO

# install ros packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-kinetic-ros-base=1.3.2-0* \
    && rm -rf /var/lib/apt/lists/*

# install ros packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-kinetic-robot=1.3.2-0* \
    && rm -rf /var/lib/apt/lists/*

# install ros packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-kinetic-desktop=1.3.2-0* \
    && rm -rf /var/lib/apt/lists/*

# install ros packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-kinetic-desktop-full=1.3.2-0* \
    && rm -rf /var/lib/apt/lists/*
# setup entrypoint


RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-kinetic-robot=1.3.2-0* \
    && rm -rf /var/lib/apt/lists/*


RUN apt update && apt-get install -y python-wstool python-catkin-tools ros-kinetic-cmake-modules protobuf-compiler autoconf libtool vim tmux wget\
    python-wstool python-catkin-tools ros-kinetic-cmake-modules protobuf-compiler autoconf libprotobuf-dev protobuf-c-compiler \
    sudo

#RUN wget https://ompl.kavrakilab.org/install-ompl-ubuntu.sh && chmod u+x ./install-ompl-ubuntu.sh &&SUDO=sudo ./install-ompl-ubuntu.sh --app
COPY --from=linjun/voxblox_omplapp /usr/local/include/ompl-1.5/ompl /usr/include/ompl
COPY --from=linjun/voxblox_omplapp /usr/local/lib/libompl* /usr/lib/

RUN ln -s /usr/lib/x86_64-linux-gnu/libvtkCommonCore-6.2.so /usr/lib/x86_64-linux-gnu/libvtkproj4-6.2.so.6.2.0


COPY ./ros_entrypoint.sh /

RUN chmod +x /ros_entrypoint.sh
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]

