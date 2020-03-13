import ctypes

import numpy as np
import tensorrt as trt

logger = trt.Logger(trt.Logger.WARNING)
ctypes.cdll.LoadLibrary('./CustomPlugin.so')

def get_plugin_creator(plugin_name):
    trt.init_libnvinfer_plugins(logger, '')
    plugin_creator_list = trt.get_plugin_registry().plugin_creator_list
    creator = None
    for c in plugin_creator_list:
        if c.name == plugin_name:
            creator = c
    return creator

if __name__ == '__main__':
    plugin_creator = get_plugin_creator('CustomPlugin')
    if plugin_creator is None:
        print("Plugin CustomPlugin not found.")
        exit()

    builder = trt.Builder(logger)
    network = builder.create_network()
    builder.max_batch_size = 2
    builder.max_workspace_size = 1 << 28

    data_shape = (3, 64, 64)
    data = network.add_input("data", trt.DataType.FLOAT, data_shape)

    plugin_fields = [trt.PluginField("var{}".format(i), np.array([i], dtype=np.int32), trt.PluginFieldType.INT32) for i in range(5)]
    plugin_field_collection = trt.PluginFieldCollection(plugin_fields)

    customPlugin = plugin_creator.create_plugin("CustomPlugin", plugin_field_collection)

    out = network.add_plugin_v2([data], customPlugin)
    network.mark_output(out.get_output(0))
    engine = builder.build_cuda_engine(network)
    
    # TODO: Add inference example
    # x = np.ones(data_shape, dtype=np.float32)

