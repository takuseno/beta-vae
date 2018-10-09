#!/bin/bash

docker run --runtime nvidia -it --rm \
    --shm-size=256m \
    -u app \
    -e QT_X11_NO_MITSHM=1 \
    -e DISPLAY=$DISPLAY \
    -v ${PWD}:/home/app \
    -v /tmp/.X11-unix/:/tmp/.X11-unix \
    takuseno/beta-vae \
    python train.py $*
