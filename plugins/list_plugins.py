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

import ctypes
import tensorrt as trt

# https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/python_api/infer/Plugin/IPluginRegistry.html
def setup_plugin_registry(logger):
    # Example of loading plugins shipped with TRT_RELEASE
    ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so", mode=ctypes.RTLD_GLOBAL)

    # Example of loading OSS plugins built from source
    #ctypes.CDLL("/mnt/TensorRT/build/out/libnvinfer_plugin.so", mode=ctypes.RTLD_GLOBAL)

    # Register the plugins loaded from libraries
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")

    # Get plugin registry to view the registered plugins
    plugin_registry = trt.get_plugin_registry()
    
    return plugin_registry


# https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/python_api/infer/Plugin/IPluginCreator.html
def get_all_plugin_details(plugin_registry):
    details = {}

    for c in plugin_registry.plugin_creator_list:
        details[c.name] = {}
        details[c.name]["tensorrt_version"] = c.tensorrt_version
        details[c.name]["plugin_version"] = c.plugin_version
        details[c.name]["plugin_namespace"] = c.plugin_namespace
        details[c.name]["PluginFields"] = []

        plugin_field_collection = c.field_names
        if plugin_field_collection:
            for i, x in enumerate(list(plugin_field_collection)):
                pfd = {
                    "name": x.name,
                    "data": x.data,
                    "type": str(x.type),
                    "size": x.size
                }
                details[c.name]["PluginFields"].append(pfd)

    return details


def get_all_plugin_names(plugin_registry):
    return [c.name for c in plugin_registry.plugin_creator_list]


if __name__ == "__main__":
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)

    plugin_registry = setup_plugin_registry(TRT_LOGGER)

    from pprint import pprint
    print("\n======== Registered Plugin Details ========\n")
    pprint(get_all_plugin_details(plugin_registry))

    print("\n======== Registered Plugin Names ========\n")
    pprint(get_all_plugin_names(plugin_registry))
