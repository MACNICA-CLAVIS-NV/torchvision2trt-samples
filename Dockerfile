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
        ninja-build \
        graphviz && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install pydotplus graphviz

#RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt && \
RUN git clone https://github.com/chitoku/torch2trt --branch jp4.6_tensorrt8 && \
    cd torch2trt && \
    python3 setup.py install --plugins && \
    cd ../ && \
    rm -rf torch2trt

WORKDIR /

RUN mkdir /${REPOSITORY_NAME}
COPY ./ /${REPOSITORY_NAME}

RUN cd /${REPOSITORY_NAME}/plugin && \
	mkdir build && \
	cd build && \
	cmake .. && \
	make

WORKDIR /
