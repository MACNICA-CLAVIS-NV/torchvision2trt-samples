# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: trt_plugin.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='trt_plugin.proto',
  package='macnica_trt_plugins',
  syntax='proto3',
  serialized_pb=_b('\n\x10trt_plugin.proto\x12\x13macnica_trt_plugins\"\x9c\x01\n\x0fpooling_Message\x12\x0c\n\x04\x64ims\x18\x01 \x03(\x03\x12.\n\x04mode\x18\x02 \x01(\x0e\x32 .macnica_trt_plugins.PoolingMode\x12\x0e\n\x06window\x18\x03 \x03(\x03\x12\x0e\n\x06stride\x18\x04 \x03(\x03\x12+\n\x04impl\x18\x05 \x01(\x0e\x32\x1d.macnica_trt_plugins.AlgoImpl\"\x1c\n\x0c\x63opy_Message\x12\x0c\n\x04\x64ims\x18\x01 \x03(\x03*?\n\x0f\x44\x61taTypeMessage\x12\n\n\x06kFloat\x10\x00\x12\t\n\x05kHalf\x10\x01\x12\t\n\x05kInt8\x10\x02\x12\n\n\x06kInt32\x10\x03*\'\n\x0bPoolingMode\x12\x0b\n\x07Maximum\x10\x00\x12\x0b\n\x07\x41verage\x10\x01*\x1f\n\x08\x41lgoImpl\x12\x08\n\x04\x43UDA\x10\x00\x12\t\n\x05\x43uDNN\x10\x01\x62\x06proto3')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

_DATATYPEMESSAGE = _descriptor.EnumDescriptor(
  name='DataTypeMessage',
  full_name='macnica_trt_plugins.DataTypeMessage',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='kFloat', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='kHalf', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='kInt8', index=2, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='kInt32', index=3, number=3,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=230,
  serialized_end=293,
)
_sym_db.RegisterEnumDescriptor(_DATATYPEMESSAGE)

DataTypeMessage = enum_type_wrapper.EnumTypeWrapper(_DATATYPEMESSAGE)
_POOLINGMODE = _descriptor.EnumDescriptor(
  name='PoolingMode',
  full_name='macnica_trt_plugins.PoolingMode',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='Maximum', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='Average', index=1, number=1,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=295,
  serialized_end=334,
)
_sym_db.RegisterEnumDescriptor(_POOLINGMODE)

PoolingMode = enum_type_wrapper.EnumTypeWrapper(_POOLINGMODE)
_ALGOIMPL = _descriptor.EnumDescriptor(
  name='AlgoImpl',
  full_name='macnica_trt_plugins.AlgoImpl',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='CUDA', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CuDNN', index=1, number=1,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=336,
  serialized_end=367,
)
_sym_db.RegisterEnumDescriptor(_ALGOIMPL)

AlgoImpl = enum_type_wrapper.EnumTypeWrapper(_ALGOIMPL)
kFloat = 0
kHalf = 1
kInt8 = 2
kInt32 = 3
Maximum = 0
Average = 1
CUDA = 0
CuDNN = 1



_POOLING_MESSAGE = _descriptor.Descriptor(
  name='pooling_Message',
  full_name='macnica_trt_plugins.pooling_Message',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='dims', full_name='macnica_trt_plugins.pooling_Message.dims', index=0,
      number=1, type=3, cpp_type=2, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='mode', full_name='macnica_trt_plugins.pooling_Message.mode', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='window', full_name='macnica_trt_plugins.pooling_Message.window', index=2,
      number=3, type=3, cpp_type=2, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='stride', full_name='macnica_trt_plugins.pooling_Message.stride', index=3,
      number=4, type=3, cpp_type=2, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='impl', full_name='macnica_trt_plugins.pooling_Message.impl', index=4,
      number=5, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=42,
  serialized_end=198,
)


_COPY_MESSAGE = _descriptor.Descriptor(
  name='copy_Message',
  full_name='macnica_trt_plugins.copy_Message',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='dims', full_name='macnica_trt_plugins.copy_Message.dims', index=0,
      number=1, type=3, cpp_type=2, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=200,
  serialized_end=228,
)

_POOLING_MESSAGE.fields_by_name['mode'].enum_type = _POOLINGMODE
_POOLING_MESSAGE.fields_by_name['impl'].enum_type = _ALGOIMPL
DESCRIPTOR.message_types_by_name['pooling_Message'] = _POOLING_MESSAGE
DESCRIPTOR.message_types_by_name['copy_Message'] = _COPY_MESSAGE
DESCRIPTOR.enum_types_by_name['DataTypeMessage'] = _DATATYPEMESSAGE
DESCRIPTOR.enum_types_by_name['PoolingMode'] = _POOLINGMODE
DESCRIPTOR.enum_types_by_name['AlgoImpl'] = _ALGOIMPL

pooling_Message = _reflection.GeneratedProtocolMessageType('pooling_Message', (_message.Message,), dict(
  DESCRIPTOR = _POOLING_MESSAGE,
  __module__ = 'trt_plugin_pb2'
  # @@protoc_insertion_point(class_scope:macnica_trt_plugins.pooling_Message)
  ))
_sym_db.RegisterMessage(pooling_Message)

copy_Message = _reflection.GeneratedProtocolMessageType('copy_Message', (_message.Message,), dict(
  DESCRIPTOR = _COPY_MESSAGE,
  __module__ = 'trt_plugin_pb2'
  # @@protoc_insertion_point(class_scope:macnica_trt_plugins.copy_Message)
  ))
_sym_db.RegisterMessage(copy_Message)


# @@protoc_insertion_point(module_scope)
