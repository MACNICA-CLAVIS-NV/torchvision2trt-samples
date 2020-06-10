/*
MIT License

Copyright (c) 2020 MACNICA Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <cstring>
#include <iostream>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include "trt_plugin.pb.h"
#include "CuDnnPooling.h"
#include "CudaPooling.h"

using namespace nvinfer1;

namespace macnica_trt_plugins
{

class PoolingPlugin : public IPluginV2Ext
{
private:
    pooling_Message message;
    macnica::Pooling *poolAlg;

public:

PoolingPlugin(pooling_Message message) : message(message)
{
    if (message.impl() == macnica_trt_plugins::AlgoImpl::CuDNN) {
        poolAlg = new macnica::CuDnnPooling();
    }
    else {
        poolAlg = new macnica::CudaPooling();
    }

    macnica::PoolingType type = macnica::PoolingType::MAXIMUM;
    if (message.mode() == macnica_trt_plugins::PoolingMode::Average) {
        type = macnica::PoolingType::AVERAGE;
    }
    int numBatches = message.dims(0);
    int numChannels = message.dims(1);
    int height = message.dims(2);
    int width = message.dims(3);
    int windowHeight = message.window(0);
    int windowWidth = message.window(1);
    int strideHeight = message.stride(0);
    int strideWidth = message.stride(1);
    poolAlg->configure( \
        type, numBatches, numChannels, height, width, \
        windowHeight, windowWidth, strideHeight, strideWidth);

#if 0
    std::cout << type << std::endl;
    std::cout << numBatches << std::endl;
    std::cout << numChannels << std::endl;
    std::cout << height << std::endl;
    std::cout << width << std::endl;
    std::cout << windowHeight << std::endl;
    std::cout << windowWidth << std::endl;
    std::cout << strideHeight << std::endl;
    std::cout << strideWidth << std::endl;
#endif
}

const char* getPluginType() const override
{
    return ("pooling");
}

const char* getPluginVersion() const override
{
    return ("1");
}

int getNbOutputs() const override
{
    return (1);
}

Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
{
    Dims    dims;
    
    dims.d[0] = inputs->d[0];
    dims.d[1] = poolAlg->heightOut;
    dims.d[2] = poolAlg->widthOut;
    dims.nbDims = 3;
    //std::cout << dims.d[0] << "," << dims.d[1] << "," << dims.d[2] << "," << dims.d[3] << std::endl;

    return (dims);
}

bool supportsFormat(DataType type, PluginFormat format) const override
{
    if (type != DataType::kFLOAT) {
        return (false);
    }
    if (format != PluginFormat::kNCHW) {
        return (false);
    }
    return (true);
}

void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) override
{
}

int initialize() override
{
    return (0);
}

void terminate() override
{
}

size_t getWorkspaceSize(int maxBatchSize) const override
{
    return (0);
}

int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override
{
    //cudaStreamSynchronize(stream);

    std::cout << "Process started." << std::endl;    

    poolAlg->process((float *)inputs[0], (float *)outputs[0], stream);

    std::cout << "Process finished." << std::endl; 

    return (0);
}

size_t getSerializationSize() const override
{
    return (message.SerializeAsString().size());
}

void serialize(void* buffer) const override
{
    message.SerializeToArray(buffer, getSerializationSize());
}

void destroy() override
{
    delete poolAlg;
}

#if 0
IPluginV2* clone() const override
{
    return (new PoolingPlugin(message));
}
#endif

void setPluginNamespace(const char* pluginNamespace) override
{
}

const char* getPluginNamespace() const override
{
    return ("macnica_trt_plugins");
}

nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    return (DataType::kFLOAT);
}

bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    return (false);
}

bool canBroadcastInputAcrossBatch(int inputIndex) const
{
    return (false);
}

void configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast, const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize)
{
}

IPluginV2Ext* clone() const
{
    return (new PoolingPlugin(message));
}

};  // class

class PoolingPluginCreator : public IPluginCreator {

public:

PoolingPluginCreator()
{
}

const char* getPluginName() const override
{
    return ("pooling");
}
    
const char* getPluginVersion() const override
{
    return ("1");
}

const PluginFieldCollection* getFieldNames() override
{
    return (nullptr);
}
  
IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) override
{
    return (nullptr);
}
 
IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override
{
    pooling_Message message;
    message.ParseFromArray(serialData, serialLength);
    return (new PoolingPlugin(message));
}
 
void setPluginNamespace(const char* pluginNamespace) override
{
}
 
const char* getPluginNamespace() const override
{
    return ("macnica_trt_plugins");
}

};  // class

REGISTER_TENSORRT_PLUGIN(PoolingPluginCreator);

}   // namespace
