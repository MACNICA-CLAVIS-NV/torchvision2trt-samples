#!/usr/bin/env bash

set -eu

source scripts/docker_base.sh

sudo docker run \
    -it \
    --rm \
    --net=host \
    --runtime nvidia \
    torchvision2trt-samples:l4t-r${L4T_VERSION}
