xhost +local:root
docker run -it \
    --gpus 0 \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume="/home/arclabdl1/Linjun/catkin_ws:/root/catkin_ws" \
    linjun/voxblox_omplapp \
    bash

export containerId=$(docker ps -l -q)
