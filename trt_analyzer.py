#
# MIT License
#
# Copyright (c) 2020 MACNICA Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE. 
#

import tensorrt as trt

def network_dict(network):
    net_dict = {'Name':[], 'Type':[], 'Inputs':[], 'Outputs':[], 'Type Specific Params':[]}
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        net_dict['Name'].append(str(layer.name))
        net_dict['Type'].append(str(layer.type))
        s = ''
        for j in range(layer.num_inputs):
            s += (' ' + str(layer.get_input(j).shape))
        net_dict['Inputs'].append(s)
        s = ''
        for j in range(layer.num_outputs):
            s += (' ' + str(layer.get_output(j).shape))
        net_dict['Outputs'].append(s)
        s = ''
        if layer.type == trt.LayerType.CONVOLUTION:
            layer.__class__ = trt.IConvolutionLayer
            s += ('ksz=%s maps=%s stride=%s pad=%s' \
                % (str(layer.kernel_size), str(layer.num_output_maps), str(layer.stride), str(layer.padding)))
        elif layer.type == trt.LayerType.POOLING:
            layer.__class__ = trt.IPoolingLayer
            s += ('type=%s wsize=%s stride=%s pad=%s' \
                % (str(layer.type), str(layer.window_size), str(layer.stride), str(layer.padding)))
        elif layer.type == trt.LayerType.ACTIVATION:
            layer.__class__ = trt.IActivationLayer
            s += ('type=%s' % (str(layer.type)))
        elif layer.type == trt.LayerType.MATRIX_MULTIPLY:
            layer.__class__ = trt.IMatrixMultiplyLayer
            s += ('op0=%s op1=%s' \
                % (str(layer.op0), str(layer.op1)))
        elif layer.type == trt.LayerType.ELEMENTWISE:
            layer.__class__ = trt.IElementWiseLayer
            s += ('op=%s' % (str(layer.op)))
        elif layer.type == trt.LayerType.FULLY_CONNECTED:
            layer.__class__ = trt.IFullyConnectedLayer
            s += ('channels=%s' % (str(layer.num_output_channels)))
        net_dict['Type Specific Params'].append(s)
    return net_dict

