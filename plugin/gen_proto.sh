#!/bin/sh
protoc --cpp_out=./ --python_out=./ trt_plugin.proto
mv trt_plugin.pb.cc trt_plugin.pb.cpp
