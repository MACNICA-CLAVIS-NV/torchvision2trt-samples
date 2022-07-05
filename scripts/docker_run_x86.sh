#!/usr/bin/env bash

set -eu

IMAGE_TAG="22.06-py3"
if [ ${#} -gt 0 ]
then
    IMAGE_TAG=${1}
fi

sudo docker run \
    -it \
    --rm \
    --net=host \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    torchvision2trt-samples:${IMAGE_TAG}
