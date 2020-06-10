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

#include <sstream>
#include <fstream>
#include <stdlib.h>

#include <cuda.h>
#include <cudnn.h>

#include "CuDnnPooling.h"
#include "error_util.h"

using namespace std;

macnica::CuDnnPooling::CuDnnPooling()
{
    checkCUDNN( \
        cudnnCreate(&cudnnHandle) \
    );

    checkCUDNN( \
        cudnnCreateTensorDescriptor(&srcTensorDesc) \
    );

    checkCUDNN( \
        cudnnCreateTensorDescriptor(&dstTensorDesc) \
    );

    checkCUDNN( \
        cudnnCreatePoolingDescriptor(&poolDesc) \
    );
}

macnica::CuDnnPooling::~CuDnnPooling()
{
    checkCUDNN( \
        cudnnDestroyPoolingDescriptor(poolDesc) \
    );

    checkCUDNN( \
        cudnnDestroyTensorDescriptor(dstTensorDesc) \
    );
    checkCUDNN( \
        cudnnDestroyTensorDescriptor(srcTensorDesc) \
    );
    checkCUDNN( \
        cudnnDestroy(cudnnHandle) \
    );
}

void macnica::CuDnnPooling::configure( \
    PoolingType type, \
    int numBatches, \
    int numChannels, \
    int height, \
    int width, \
    int windowHeight, \
    int windowWidth, \
    int strideHeight, \
    int strideWidth \
)
{
    {
        const int nDims = 2;
        int winDimA[nDims] = {windowHeight, windowWidth};
        int padA[nDims] = {0, 0};
        int strideA[nDims] = {strideHeight, strideWidth};
        checkCUDNN(
            cudnnSetPoolingNdDescriptor( \
                poolDesc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, \
                nDims, winDimA, padA, strideA \
            ) \
        );
    }

    int numBatchesOut = numBatches;
    int numChannelsOut = numChannels;
    heightOut = height;
    widthOut = width;

    {
        const int nDims = 4;
        int dimA[nDims] = {numBatches, numChannels, height, width};
        int strideA[nDims] = { \
            numChannels * height * width, height * width, width, 1};
        checkCUDNN( \
            cudnnSetTensorNdDescriptor( \
                srcTensorDesc, CUDNN_DATA_FLOAT, nDims, dimA, strideA \
            ) \
        );

        checkCUDNN( \
            cudnnGetPoolingNdForwardOutputDim( \
                poolDesc, srcTensorDesc, nDims, dimA \
            ) \
        );
        numBatchesOut = dimA[0];
        numChannelsOut = dimA[1];
        heightOut = dimA[2];
        widthOut = dimA[3];
    }

    {
        const int nDims = 4;
        int dimA[nDims] = {numBatchesOut, numChannelsOut, heightOut, widthOut};
        int strideA[nDims] = { \
            numChannelsOut * heightOut * widthOut, \
            heightOut * widthOut, widthOut, 1};
        checkCUDNN( \
            cudnnSetTensorNdDescriptor( \
                dstTensorDesc, CUDNN_DATA_FLOAT, nDims, dimA, strideA \
            ) \
        );
    }
}

void macnica::CuDnnPooling::process(float *src, float *dst, cudaStream_t stream)
{
    if (stream) {
        cudaStreamSynchronize(stream);
    }

    const float alpha = 1.0;
    const float beta = 0.0;
    checkCUDNN( \
        cudnnPoolingForward( \
            cudnnHandle, poolDesc, (void *)&alpha, \
            srcTensorDesc, src, (void *)&beta, dstTensorDesc, dst \
        ) \
    );
}
