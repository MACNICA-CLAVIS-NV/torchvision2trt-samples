ARG BASE_IMAGE=nvcr.io/nvidia/l4t-ml:r32.6.1-py3
FROM ${BASE_IMAGE}

ARG REPOSITORY_NAME=torchvision2trt-samples

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

WORKDIR /tmp

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libprotobuf* \
        protobuf-compiler \
        graphviz && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install graphviz onnx pydot

WORKDIR /

RUN mkdir /${REPOSITORY_NAME}
COPY ./ /${REPOSITORY_NAME}

RUN cd /${REPOSITORY_NAME}/plugin && \
    protoc --cpp_out=./ --python_out=./ trt_plugin.proto && \
    mv trt_plugin.pb.cc trt_plugin.pb.cpp && \
    rm -rf build && \
	mkdir build && \
	cd build && \
	cmake .. && \
	make

WORKDIR /
