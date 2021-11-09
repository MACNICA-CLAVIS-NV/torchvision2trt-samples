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

AsciiChar const* getPluginType() const noexcept override
{
    return ("copy");
}

AsciiChar const* getPluginVersion() const noexcept override
{
    return ("1");
}

int32_t getNbOutputs() const noexcept override
{
    return (1);
}

Dims getOutputDimensions(int32_t index, const Dims* inputs, int32_t nbInputDims) noexcept override
{
    Dims    dims;
    dims.nbDims = inputs->nbDims;
    for (int i = 0;i < inputs->nbDims;i++) {
        dims.d[i] = inputs->d[i];
    }
    return (dims);
}

bool supportsFormat(DataType type, PluginFormat format) const noexcept override
{
    if (type != DataType::kFLOAT) {
        return (false);
    }
    //if (format != PluginFormat::kNCHW) {
    if (format != PluginFormat::kLINEAR) {
        return (false);
    }
    return (true);
}

void configureWithFormat(const Dims* inputDims, int32_t nbInputs, const Dims* outputDims, int32_t nbOutputs, DataType type, PluginFormat format, int32_t maxBatchSize) noexcept override
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

int32_t initialize() noexcept override
{
    std::cout << "initialize called" << std::endl;
    return (0);
}

void terminate() noexcept override
{
    std::cout << "terminate called" << std::endl;
}

size_t getWorkspaceSize(int32_t maxBatchSize) const noexcept override
{
    return (0);
}

int32_t enqueue(int32_t batchSize, void const* const* inputs, void* const* outputs, void* workspace,
         cudaStream_t stream) noexcept override
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

size_t getSerializationSize() const noexcept override
{
    return (message.SerializeAsString().size());
}

void serialize(void* buffer) const noexcept override
{
    message.SerializeToArray(buffer, getSerializationSize());
}

void destroy() noexcept override
{
    std::cout << "destroy called" << std::endl;
}

#if 0
IPluginV2* clone() const override
{
    return (new CopyPlugin(message));
}
#endif

void setPluginNamespace(AsciiChar const* pluginNamespace) noexcept override
{
}

AsciiChar const* getPluginNamespace() const noexcept override
{
    return ("macnica_trt_plugins");
}

nvinfer1::DataType getOutputDataType(int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    return (DataType::kFLOAT);
}

bool isOutputBroadcastAcrossBatch(int32_t outputIndex, bool const* inputIsBroadcasted, int32_t nbInputs) const noexcept
{
    return (false);
}

bool canBroadcastInputAcrossBatch(int32_t inputIndex) const noexcept
{
    return (false);
}

void configurePlugin(Dims const* inputDims, int32_t nbInputs, Dims const* outputDims, int32_t nbOutputs,
         DataType const* inputTypes, DataType const* outputTypes, bool const* inputIsBroadcast,
         bool const* outputIsBroadcast, PluginFormat floatFormat, int32_t maxBatchSize) noexcept
{
    std::cout << "configureWithFormat called" << std::endl;
}

IPluginV2Ext* clone() const noexcept
{
    return (new CopyPlugin(message));
}

};  // class

class CopyPluginCreator : public IPluginCreator {

public:

CopyPluginCreator()
{
}

AsciiChar const* getPluginName() const noexcept override
{
    return ("copy");
}
    
AsciiChar const* getPluginVersion() const noexcept override
{
    return ("1");
}

PluginFieldCollection const* getFieldNames() noexcept override
{
    return (nullptr);
}
  
IPluginV2* createPlugin(AsciiChar const* name, PluginFieldCollection const* fc) noexcept override
{
    return (nullptr);
}
 
IPluginV2* deserializePlugin(AsciiChar const* name, void const* serialData, size_t serialLength) noexcept override
{
    copy_Message message;
    message.ParseFromArray(serialData, serialLength);
    return (new CopyPlugin(message));
}
 
void setPluginNamespace(AsciiChar const* pluginNamespace) noexcept override
{
}
 
AsciiChar const* getPluginNamespace() const noexcept override
{
    return ("macnica_trt_plugins");
}

};  // class

REGISTER_TENSORRT_PLUGIN(CopyPluginCreator);

}   // namespace
