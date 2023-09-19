#!/bin/bash

# Run container from image
docker run -it --rm \
    --network host \
    --shm-size 6G \
    -v /home/andrey/Zion/workspaces/SNR-SKF:/workspaces/SNR-SKF \
    -v /dev/*:/dev/* \
    -v /etc/localtime:/etc/localtime:ro \
    --name "vblc-container" \
    --runtime nvidia \
    --gpus all \
    --workdir /workspaces/SNR-SKF snr
