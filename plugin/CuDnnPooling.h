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

#pragma once

#include <cuda.h>
#include <cudnn.h>
#include "Pooling.h"

namespace macnica
{

class CuDnnPooling : public Pooling
{
private:
    cudnnHandle_t cudnnHandle;
    cudnnTensorDescriptor_t srcTensorDesc;
    cudnnTensorDescriptor_t dstTensorDesc;
    cudnnPoolingDescriptor_t poolDesc;
    
public:
    CuDnnPooling();

    ~CuDnnPooling();

    void configure( \
        PoolingType type, \
        int numBatches, \
        int numChannels, \
        int height, \
        int width, \
        int windowHeight, \
        int windowWidth, \
        int strideHeight, \
        int strideWidth \
    );

    void process(float *src, float *dst, cudaStream_t stream);
};

}
