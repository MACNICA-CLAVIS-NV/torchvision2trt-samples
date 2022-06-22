#!/usr/bin/env bash

set -eu

source scripts/docker_base.sh

CONTAINER_NAME="my-torchvision2trt-samples"
if [ ${#} -gt 0 ]
then
    CONTAINER_NAME=${1}
fi
echo "Container \"${CONTAINER_NAME}\" is being committed to image \"torchvision2trt-samples:l4t-r${L4T_VERSION}\"."

sudo docker commit ${CONTAINER_NAME} torchvision2trt-samples:l4t-r${L4T_VERSION}
