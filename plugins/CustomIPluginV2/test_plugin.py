#!/usr/bin/env python3

# Copyright 2020 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import ctypes
import logging

import numpy as np
import tensorrt as trt

logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)
TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def get_plugin_creator_by_name(plugin_registry, plugin_name):
    plugin_creator_list = plugin_registry.plugin_creator_list
    for c in plugin_creator_list:
        if c.name == plugin_name:
            return c

if __name__ == '__main__':
    # Load our CustomPlugin library
    plugin_library = os.path.join(".", "CustomPlugin.so")
    logger.info("Loading plugin library: {}".format(plugin_library))
    ctypes.cdll.LoadLibrary(plugin_library)

    # Initialize/Register plugins
    logger.info("Initializing plugin registry")
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")
    plugin_registry = trt.get_plugin_registry()

    # List all registered plugins. Should see our CustomPlugin in this list.
    logger.info("Registered Plugins:")
    print("\n".join([c.name for c in plugin_registry.plugin_creator_list]))

    # Get plugin creator for our custom plugin.
    plugin_name = "CustomPlugin"
    logger.info("Looking up IPluginCreator for {}".format(plugin_name))
    plugin_creator = get_plugin_creator_by_name(plugin_registry, plugin_name)
    if not plugin_creator:
        raise Exception("[{}] IPluginCreator not found.".format(plugin_name))

    # Add our custom plugin to a network, and build a TensorRT engine from it.
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network:
        data_shape = (3, 64, 64)
        data = network.add_input("data", trt.DataType.FLOAT, data_shape)

        logger.info("Creating PluginFields for {} plugin".format(plugin_name))
        plugin_fields = [trt.PluginField("var{}".format(i), np.array([i], dtype=np.int32), trt.PluginFieldType.INT32) for i in range(5)]
        logger.info("Creating PluginFieldCollection for {} plugin".format(plugin_name))
        plugin_field_collection = trt.PluginFieldCollection(plugin_fields)
        logger.info("Creating {} plugin from PluginFieldCollection".format(plugin_name))
        customPlugin = plugin_creator.create_plugin(plugin_name, plugin_field_collection)

        logger.info("Adding {} plugin to network.".format(plugin_name))
        out = network.add_plugin_v2([data], customPlugin)
        network.mark_output(out.get_output(0))

        # Serialize our engine for future use
        logger.info("Building engine...")
        with builder.build_cuda_engine(network) as engine:
            filename = "custom_plugin.engine"
            with open(filename, "wb") as f:
                f.write(engine.serialize())
                logger.info("Serialized engine file written to {}".format(filename))
        
            # TODO: Add inference example
            # x = np.ones(data_shape, dtype=np.float32)

