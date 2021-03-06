FROM nvcr.io/nvidia/l4t-ml:r32.4.3-py3

WORKDIR /tmp

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libprotobuf* \
        protobuf-compiler \
        ninja-build \
        graphviz && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install pydotplus graphviz

RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt && \
    cd torch2trt && \
    python3 setup.py install --plugins && \
    cd ../ && \
    rm -rf torch2trt

WORKDIR /

RUN git clone https://github.com/MACNICA-CLAVIS-NV/torchvision2trt-samples && \
	cd torchvision2trt-samples/plugin && \
	mkdir build && \
	cd build && \
	cmake .. && \
	make

WORKDIR /
