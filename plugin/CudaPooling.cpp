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

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

#include "CudaPooling.h"

using namespace std;

void cudaPooling(int type, \
	float *src, unsigned int srcHeight, int srcWidth, \
	float *dst, unsigned int dstHeight, int dstWidth, \
	int windowHeight, int windowWidth, \
	int strideHeight, int strideWidth, cudaStream_t stream);

macnica::CudaPooling::CudaPooling()
{
}

macnica::CudaPooling::~CudaPooling()
{
}

void macnica::CudaPooling::configure( \
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
	this->type = type;
	this->numBatches = numBatches;
	this->numChannels = numChannels;
	this->height = height;
	this->width = width;
	this->windowWidth = windowWidth;
	this->windowHeight = windowHeight;
	this->strideWidth = strideWidth;
	this->strideHeight = strideHeight;

	this->heightOut = this->height / this->strideHeight;
	this->widthOut = this->width / this->strideWidth;
}

void macnica::CudaPooling::process(float *src, float *dst, cudaStream_t stream)
{
	int poolType = 1;
	if (this->type == PoolingType::AVERAGE) {
		poolType = 0;
	}

	for (int b = 0;b < this->numBatches;b++) {
		for (int c = 0;c < this->numChannels;c++) {
			cudaPooling(poolType, \
				src, this->height, this->width, \
				dst, this->heightOut, this->widthOut, \
				this->windowHeight, this->windowWidth, \
				this->strideHeight, this->strideWidth, stream);
			src += (this->height * this->width);
			dst += (this->heightOut * this->widthOut);
		}
	}
}
