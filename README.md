# torchvision2trt-samples

*Read this in Japanese [日本語](README.jp.md)*

## What does this application do?
- This repository provides a colletion of Jupyer notebooks to demonstrate on how to convert Torchvision pre-trained models to NVIDIA TensorRT.
- You can also understand how to develop TensorRT custom layer with NVIDIA CUDA and NVIDIA CuDNN with a sample TensorRT plugin contained in this repository.

## Jupyter notebooks

1. **PyTorch inference** \([torchvision_normal.ipynb](./torchvision_normal.ipynb)\)  
    This notebook shows how to do inference by GPU in PyTorch.  
    ![](./doc/torchvision-normal.svg)

1. **TensorRT inference with ONNX model** \([torchvision_onnx.ipynb](./torchvision_onnx.ipynb)\)  
    This notebook shows how to convert a pre-trained PyTorch model to a ONNX model first, and also shows how to do inference by TensorRT with the ONNX model.  
    ![](./doc/torchvision-onnx.svg)

1. **TensorRT inference with Torch-TensorRT** \([torchvision_torch_tensorrt.ipynb](./torchvision_torch_tensorrt.ipynb)\)  
    This notebook shows how to import a pre-trained PyTorch model to TensorRT with [Torch-TensorRT](https://github.com/pytorch/TensorRT).  
    ![](./doc/torchvision-torch-tensorrt.svg)
    You need to install Torch-TensorRT in the Docker container separately. Please refer to [\"Install Torch-TensorRT\"](#install-torch-tensorrt) for the details.

1. **TensorRT Inference with TensorRT API** \([torchvision_trtapi.ipynb](./torchvision_trtapi.ipynb)\)  
    This notebook  shows how to import a pre-trained PyTorch model data (weights and bias) with a user-defined network with the TensorRT API. This notebook also shows how to use custom layers with the TensorRT API.  
    ![](./doc/torchvision-trtapi.svg)

## Prerequisites
- NVIDIA Jetson Series Developer Kits
- NVIDIA JetPack 4.4 or later
    - The Torch-TensorRT sample \([torchvision_torch_tensorrt.ipynb](./torchvision_torch_tensorrt.ipynb)\) needs JetPack 4.6 or later.

## Installation

- **This application can be installed with Dockerfile so that you don't need to clone this repository manually.**
- **This application will be built on [Machine Learning for Jetson/L4T](https://ngc.nvidia.com/catalog/containers/nvidia:l4t-ml) which is distributed from NVIDIA NGC.**

### Change docker configuration

1. Set the default docker runtime to **nvidia** as described at [this link](https://github.com/dusty-nv/jetson-containers#docker-default-runtime)

1. Reboot your Jetson

### Increase swap memory **(Only for Jetson Nano)**

The default 2GB swap memory is insufficient. Increse it to 4GB as described at [JetsonHacks - Jetson Nano – Even More Swap](https://www.jetsonhacks.com/2019/11/28/jetson-nano-even-more-swap/)  
You need to restart Jetson after the swap memory expansion.

### Build a docker image locally

1. Clone this repository.
    ```
    $ git clone https://github.com/MACNICA-CLAVIS-NV/torchvision2trt-samples
    ```
1. Build a docker image
    ```
    $ cd torchvision2trt-samples
    
    $ ./scripts/docker_build.sh
    ```

### Install Torch-TensorRT

After the container build, please install Torch-TensorRT with the [install_torch_tensorrt](./install_torch_tensorrt.ipynb) notebook.

1. Launch a named (persistent) container with the docker_run_named.sh script.
    ```
    ./scripts/docker_run_named.sh
    ```

1.  Open [localhost:8888](http://localhost:8888) from Web browser, and input the password **"nvidia"**.

1. You can find \"install_torch_tensorrt\" notebook at the **/torchvision2trt-samples** directory. Please follow the instruction in the notebook. The build process takes about one hour. After the build is completed, exit from Jupyter, then exit from the Docker container. 

1. Committed the container to the image.
    ```
    ./scripts/docker_commit.sh
    ```

1. Now you can remove the container.
    ```
    sudo docker rm my-torchvision2trt-samples
    ```

*Please note that only the Torch-TensorRT sample \([torchvision_torch_tensorrt.ipynb](./torchvision_torch_tensorrt.ipynb)\) requires this installation.*

## Usage

**For Jetson Nano, you sometimes see the low memory warning on Jetson's L4T desktop while you run these notebooks. To run these notebooks on Jetson Nano, logout the desktop, and login to the Jetson Nano from your PC with network access, and open these notebooks in a Web browser of your PC remotely. It seems that this method reduces Jetson Nano's memory usage.**

1. Run a docker container generated from the image built as the above.
    ```
    $ ./scripts/docker_run.sh
    ```
1. Open [localhost:8888](http://localhost:8888) from Web browser, and input the password **"nvidia"**.

1. You can find these samples at the **/torchvision2trt-samples** directory as the following picture.
![Screenshot1](./doc/screenshot.jpg)

## How to rebuild the pooling plugin library

1. Open a terminal (Click the terminal button as shown in the following figure.)  
![Screenshot1](./doc/screenshot2.jpg)

2. Follow the following the following instruction.

    ```
    # cd /torchvision2trt-samples/plugin

    # rm -R build

    # mkdir build

    # cd build

    # cmake ..
    -- The CXX compiler identification is GNU 7.5.0
    -- The CUDA compiler identification is NVIDIA 10.2.89
    -- Check for working CXX compiler: /usr/bin/c++
    -- Check for working CXX compiler: /usr/bin/c++ -- works
    -- Detecting CXX compiler ABI info
    -- Detecting CXX compiler ABI info - done
    -- Detecting CXX compile features
    -- Detecting CXX compile features - done
    -- Check for working CUDA compiler: /usr/local/cuda/bin/nvcc
    -- Check for working CUDA compiler: /usr/local/cuda/bin/nvcc -- works
    -- Detecting CUDA compiler ABI info
    -- Detecting CUDA compiler ABI info - done
    -- Looking for C++ include pthread.h
    -- Looking for C++ include pthread.h - found
    -- Looking for pthread_create
    -- Looking for pthread_create - not found
    -- Looking for pthread_create in pthreads
    -- Looking for pthread_create in pthreads - not found
    -- Looking for pthread_create in pthread
    -- Looking for pthread_create in pthread - found
    -- Found Threads: TRUE  
    -- Found Protobuf: /usr/lib/aarch64-linux-gnu/libprotobuf.so;-lpthread (found version "3.0.0") 
    -- Configurable variable Protobuf_VERSION set to 3.0.0
    -- Configurable variable Protobuf_INCLUDE_DIRS set to /usr/include
    -- Configurable variable Protobuf_LIBRARIES set to /usr/lib/aarch64-linux-gnu/libprotobuf.so;-lpthread
    -- Found CUDA: /usr/local/cuda (found version "10.2") 
    -- Configurable variable CUDA_VERSION set to 10.2
    -- Configurable variable CUDA_INCLUDE_DIRS set to /usr/local/cuda/include
    -- Found CUDNN: /usr/include  
    -- Found cuDNN: v?  (include: /usr/include, library: /usr/lib/aarch64-linux-gnu/libcudnn.so)
    -- Configurable variable CUDNN_VERSION set to ?
    -- Configurable variable CUDNN_INCLUDE_DIRS set to /usr/include
    -- Configurable variable CUDNN_LIBRARIES set to /usr/lib/aarch64-linux-gnu/libcudnn.so
    -- Configurable variable CUDNN_LIBRARY_DIRS set to 
    -- Found TensorRT: /usr/lib/aarch64-linux-gnu/libnvinfer.so (found version "..") 
    -- Configurable variable TensorRT_VERSION_STRING set to ..
    -- Configurable variable TensorRT_INCLUDE_DIRS set to /usr/include/aarch64-linux-gnu
    -- Configurable variable TensorRT_LIBRARIES set to /usr/lib/aarch64-linux-gnu/libnvinfer.so
    -- Configuring done
    -- Generating done
    -- Build files have been written to: /torchvision2trt-samples/plugin/build

    # make
    Scanning dependencies of target PoolingPlugin
    [ 12%] Building CUDA object CMakeFiles/PoolingPlugin.dir/PoolingAlgo.cu.o
    [ 25%] Building CXX object CMakeFiles/PoolingPlugin.dir/CudaPooling.cpp.o
    [ 37%] Building CXX object CMakeFiles/PoolingPlugin.dir/trt_plugin.pb.cpp.o
    [ 50%] Building CXX object CMakeFiles/PoolingPlugin.dir/PoolingPlugin.cpp.o
    [ 62%] Building CXX object CMakeFiles/PoolingPlugin.dir/CuDnnPooling.cpp.o
    [ 75%] Building CXX object CMakeFiles/PoolingPlugin.dir/CopyPlugin.cpp.o
    [ 87%] Linking CUDA device code CMakeFiles/PoolingPlugin.dir/cmake_device_link.o
    [100%] Linking CXX shared module libPoolingPlugin.so
    [100%] Built target PoolingPlugin
    ```

## References

- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt)
- [NVIDIA-AI-IOT/torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt)
- [Developing Real-time Neural Networks for Jetson](https://www.nvidia.com/en-us/gtc/on-demand/?search=22676)
- [Machine Learning for Jetson/L4T](https://ngc.nvidia.com/catalog/containers/nvidia:l4t-ml)
