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
import logging
import tensorrt as trt

logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)
TRT_LOGGER = trt.Logger(trt.Logger.INFO)

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


# Example usage: 
#   (1) python list_plugins.py 
#   (2) python list_plugins.py --plugins CustomIPluginV2/CustomPlugin.so
#   (3) python list_plugins.py --plugins CustomIPluginV2/CustomPlugin.so /mnt/TensorRT/build/out/libnvinfer_plugin.so
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Script to list registered TensorRT plugins. Can optionally load custom plugin libraries.")
    parser.add_argument("-p", "--plugins", nargs="*", default=[], help="Path to a plugin (.so) library file. Accepts multiple arguments.")
    args = parser.parse_args()

    for plugin_library in args.plugins:
        # Example default plugin library: "/usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so"
        logger.info("Loading plugin library: {}".format(plugin_library))
        ctypes.CDLL(plugin_library, mode=ctypes.RTLD_GLOBAL)

    logger.info("Registering plugins...")
    # Register the plugins loaded from libraries
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")

    # Get plugin registry to view the registered plugins
    plugin_registry = trt.get_plugin_registry()

    from pprint import pprint
    #logger.info("Registered Plugin Details:")
    #pprint(get_all_plugin_details(plugin_registry))

    logger.info("Registered Plugin Names:") 
    pprint(get_all_plugin_names(plugin_registry))
