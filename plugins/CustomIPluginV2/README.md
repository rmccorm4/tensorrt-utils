# IPluginV2 Dummy Example

This directory contains a minimal example of creating, building,
serializing, and de-serializing a TensorRT engine using a custom
TensorRT Plugin using the IPluginV2 interface.

The goal is to fill any potential gaps in documentation and make
it more clear how to use plugins in TensorRT.


## Disclaimer

The Plugin interface evolves and gets more features with each release. 
`IPluginV2` is a rather old interface from TensorRT 5, but still works in TensorRT 6 and 7.

Make sure to check out the 
[documentation](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#plugin-api-desc)
for more details on each. 

I generally follow these follow these rough guidelines:
* **TensorRT 5**:  `IPluginV2`
* **TensorRT 6**:  Prefer `IPluginV2Ext`, but `IPluginV2` is still supported.
* **TensorRT 7**: Prefer `IPluginV2DynamicExt` or `IPluginV2IOExt`, but 
`IPluginV2` and `IPluginV2Ext` are still supported.

## Usage

```
# (Optional) Start a TensorRT 7 Container


# Build the plugin library .so
make

# Create a plugin object, build an engine, and serialize it
python test_plugin.py

# Deserialize the engine and run inference on it with random data
trtexec --loadEngine=custom_plugin.engine --plugins=./CustomPlugin.so
```

> NOTE: You must load a plugin's library before attempting to deserialize it.
In other words, the above `trtexec` command will fail without
`--plugins=./CustomPlugin.so`

## Example Outputs

Building the plugin library:
```
$ make
g++ -g -std=c++11 -DNDEBUG -fPIC -MD -MP -I. -I/usr/local/cuda/include -I/usr/src/tensorrt/include -o CustomPlugin.o -c CustomPlugin.cpp
g++ -g -std=c++11 -DNDEBUG -shared -o CustomPlugin.so CustomPlugin.o -L/usr/local/cuda/lib64 -L/usr/src/tensorrt/lib  -lnvinfer -lcudart
```

Creating, building, and serializing an engine with the plugin:
```
root@028772c8fd4f:/mnt/CustomIPluginV2# python test_plugin.py  
2020-03-13 21:53:39 - __main__ - INFO - Loading plugin library: ./CustomPlugin.so
Plugin attribute number: 5
2020-03-13 21:53:39 - __main__ - INFO - Initializing plugin registry
2020-03-13 21:53:39 - __main__ - INFO - Registered Plugins:
...
CustomPlugin
...
2020-03-13 21:53:39 - __main__ - INFO - Looking up IPluginCreator for CustomPlugin
2020-03-13 21:53:40 - __main__ - INFO - Creating PluginFields for CustomPlugin plugin
2020-03-13 21:53:40 - __main__ - INFO - Creating PluginFieldCollection for CustomPlugin plugin
2020-03-13 21:53:40 - __main__ - INFO - Creating CustomPlugin plugin from PluginFieldCollection
var0: 0
var1: 1
var2: 2
var3: 3
var4: 4
2020-03-13 21:53:40 - __main__ - INFO - Adding CustomPlugin plugin to network.
2020-03-13 21:53:40 - __main__ - INFO - Building engine...
[TensorRT] INFO: Detected 1 inputs and 1 output network tensors.
2020-03-13 21:53:41 - __main__ - INFO - Serialized engine file written to custom_plugin.engine
```

Loading the engine and doing inference:
```
$ trtexec --loadEngine=custom_plugin.engine --plugins=./CustomPlugin.so
...
[03/13/2020-21:36:01] [I] Plugins: ./CustomPlugin.so
...
[03/13/2020-21:36:01] [I] Loading supplied plugin library: ./CustomPlugin.so
...
[03/13/2020-21:36:06] [I] GPU Compute
[03/13/2020-21:36:06] [I] min: 0.00292969 ms
[03/13/2020-21:36:06] [I] max: 0.022583 ms
[03/13/2020-21:36:06] [I] mean: 0.00384302 ms
[03/13/2020-21:36:06] [I] median: 0.00408936 ms
[03/13/2020-21:36:06] [I] percentile: 0.00415039 ms at 99%
[03/13/2020-21:36:06] [I] total compute time: 0.29489 s
&&&& PASSED TensorRT.trtexec # trtexec --loadEngine=custom_plugin.engine --plugins=./CustomPlugin.so
```
