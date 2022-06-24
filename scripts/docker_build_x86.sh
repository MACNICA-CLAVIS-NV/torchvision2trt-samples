#!/usr/bin/env bash

set -eu

IMAGE_TAG="22.05-py3"
if [ ${#} -gt 0 ]
then
    IMAGE_TAG=${1}
fi

BASE_IMAGE="nvcr.io/nvidia/pytorch:${IMAGE_TAG}"

echo "Base Image: ${BASE_IMAGE}"

sudo docker build --build-arg BASE_IMAGE=${BASE_IMAGE} -f Dockerfile.x86 -t torchvision2trt-samples:${IMAGE_TAG} .
