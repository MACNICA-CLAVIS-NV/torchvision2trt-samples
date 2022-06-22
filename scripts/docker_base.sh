#!/usr/bin/env bash

source scripts/l4t_version.sh

BASE_IMAGE="nvcr.io/nvidia/l4t-ml:r${L4T_VERSION}-py3"
BASE_DEVEL="nvcr.io/nvidia/nvidia-l4t-ml:r${L4T_VERSION}-py3"

if [ $L4T_RELEASE -eq 32 ]; then
	if [ $L4T_REVISION_MAJOR -eq 5 ]; then
		if [ $L4T_REVISION_MINOR -eq 1 ]; then
			BASE_IMAGE="nvcr.io/nvidia/l4t-ml:r32.5.0-py3"
		elif [ $L4T_REVISION_MINOR -eq 2 ]; then
			BASE_IMAGE="nvcr.io/nvidia/l4t-ml:r32.5.0-py3"
		fi
	fi
fi
	
echo "l4t-base image:  $BASE_IMAGE"