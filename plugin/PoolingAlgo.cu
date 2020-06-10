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

// System includes
#include <stdio.h>
#include <assert.h>
#include <float.h>
#include <stdlib.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

__global__ void maxPooling( \
    float *src, unsigned int srcHeight, unsigned int srcWidth, \
    float *dst,  unsigned int dstHeight, unsigned int dstWidth, \
    uint2 window, uint2 stride)
{
    unsigned int dstX = threadIdx.x + blockIdx.x * blockDim.x;
    if (dstX >= dstWidth) {
        return;
    }
    unsigned int dstY = threadIdx.y + blockIdx.y * blockDim.y;
    if (dstY >= dstHeight) {
      return;
    }
    unsigned int dstIdx = dstY * dstWidth + dstX;

    unsigned int srcX = dstX * stride.x;
    if (srcX >= srcWidth) {
        return;
    }
    unsigned int srcY = dstY * stride.y;
    if (srcY >= srcHeight) {
        return;
    }
    unsigned int srcIdx = srcY * srcWidth + srcX;

    float maxVal = -FLT_MAX;
    unsigned int idx;

    for (int h = 0;h < window.y;h++) {
        idx = srcIdx;
        for (int w = 0;w < window.x;w++) {
            maxVal = MAX(src[idx], maxVal);
            idx += 1;
        }
        srcIdx += srcWidth;
    }

    dst[dstIdx] = maxVal;
}

__global__ void avgPooling( \
    float *src, unsigned int srcHeight, unsigned int srcWidth, \
    float *dst,  unsigned int dstHeight, unsigned int dstWidth, \
    uint2 window, uint2 stride)
{
    unsigned int dstX = threadIdx.x + blockIdx.x * blockDim.x;
    if (dstX >= dstWidth) {
        return;
    }
    unsigned int dstY = threadIdx.y + blockIdx.y * blockDim.y;
    if (dstY >= dstHeight) {
      return;
    }
    unsigned int dstIdx = dstY * dstWidth + dstX;

    unsigned int srcX = dstX * stride.x;
    if (srcX >= srcWidth) {
        return;
    }
    unsigned int srcY = dstY * stride.y;
    if (srcY >= srcHeight) {
        return;
    }
    unsigned int srcIdx = srcY * srcWidth + srcX;

    float sum = 0.0;
    unsigned int idx;

    for (int h = 0;h < window.y;h++) {
        idx = srcIdx;
        for (int w = 0;w < window.x;w++) {
            sum += src[idx];
            idx += 1;
        }
        srcIdx += srcWidth;
    }

    dst[dstIdx] = sum / (window.y * window.x);
}

void cudaPooling(int type, \
    float *src, unsigned int srcHeight, int srcWidth, \
    float *dst, unsigned int dstHeight, int dstWidth, \
    int windowHeight, int windowWidth, \
    int strideHeight, int strideWidth, cudaStream_t stream)
{
    dim3 block(32, 32);
    int grid_x = (dstWidth + block.x - 1) / block.x;
    int grid_y = (dstHeight + block.y - 1) / block.y;
    dim3 grid(grid_x, grid_y);
    uint2 window = make_uint2(windowWidth, windowHeight);
    uint2 stride = make_uint2(strideWidth, strideHeight);

    if (type) {
        maxPooling<<<grid, block, 0, stream>>>( \
            src, srcHeight, srcWidth, dst, dstHeight, dstWidth, window, stride);
    }
    else {
        avgPooling<<<grid, block, 0, stream>>>( \
            src, srcHeight, srcWidth, dst, dstHeight, dstWidth, window, stride);
    }
}
