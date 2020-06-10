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

using namespace nvinfer1;

namespace macnica_trt_plugins
{

class CopyPlugin : public IPluginV2Ext
{
private:
    copy_Message message;

public:

CopyPlugin(copy_Message message) : message(message)
{
    std::cout << "CopyPlugin called" << std::endl;
}

const char* getPluginType() const override
{
    return ("copy");
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
    dims.nbDims = inputs->nbDims;
    for (int i = 0;i < inputs->nbDims;i++) {
        dims.d[i] = inputs->d[i];
    }
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
    std::cout << "configureWithFormat called" << std::endl;
    for (int i = 0;i < inputDims->nbDims;i++) {
        std::cout << inputDims->d[i];
        std::cout << ", ";
    }
    std::cout << std::endl;
    for (int i = 0;i < outputDims->nbDims;i++) {
        std::cout << outputDims->d[i];
        std::cout << ", ";
    }
    std::cout << std::endl;
}

int initialize() override
{
    std::cout << "initialize called" << std::endl;
    return (0);
}

void terminate() override
{
    std::cout << "terminate called" << std::endl;
}

size_t getWorkspaceSize(int maxBatchSize) const override
{
    return (0);
}

int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override
{
    std::cout << "Process started!" << std::endl;    

    int len = 1;
    for (int i = 0; i < message.dims_size(); i++) {
        len *= message.dims(i);
        std::cout << message.dims(i) << std::endl;
    }

    std::cout << "Batch size: " << batchSize << std::endl;
    std::cout << "Transfer elements: " << len << std::endl;

    //memcpy(outputs[0], inputs[0], (size_t)(len * sizeof(float)));
    cudaMemcpyAsync(outputs[0], inputs[0], (size_t)(len * sizeof(float)), cudaMemcpyDeviceToDevice, stream);

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
    std::cout << "destroy called" << std::endl;
}

#if 0
IPluginV2* clone() const override
{
    return (new CopyPlugin(message));
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
    std::cout << "configureWithFormat called" << std::endl;
}

IPluginV2Ext* clone() const
{
    return (new CopyPlugin(message));
}

};  // class

class CopyPluginCreator : public IPluginCreator {

public:

CopyPluginCreator()
{
}

const char* getPluginName() const override
{
    return ("copy");
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
    copy_Message message;
    message.ParseFromArray(serialData, serialLength);
    return (new CopyPlugin(message));
}
 
void setPluginNamespace(const char* pluginNamespace) override
{
}
 
const char* getPluginNamespace() const override
{
    return ("macnica_trt_plugins");
}

};  // class

REGISTER_TENSORRT_PLUGIN(CopyPluginCreator);

}   // namespace
