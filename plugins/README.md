# Plugins

This directory contains some tools and samples related to
TensorRT Plugins.

## Listing Registered Plugins

```
$ python list_plugins.py -h
usage: list_plugins.py [-h] [-p [PLUGINS [PLUGINS ...]]]

Script to list registered TensorRT plugins. Can optionally load custom plugin
libraries.

optional arguments:
  -h, --help            show this help message and exit
  -p [PLUGINS [PLUGINS ...]], --plugins [PLUGINS [PLUGINS ...]]
                        Path to a plugin (.so) library file. Accepts multiple
                        arguments.
```

Default `TRT_RELEASE` plugins:
```
$ python list_plugins.py   
2020-03-13 22:12:35 - __main__ - INFO - Registering plugins...
2020-03-13 22:12:35 - __main__ - INFO - Registered Plugin Names:
['RnRes2Br2bBr2c_TRT',
 'RnRes2Br1Br2c_TRT',
 'CgPersistentLSTMPlugin_TRT',
 'SingleStepLSTMPlugin',
 ...
 'SpecialSlice_TRT',
 'InstanceNormalization_TRT']
```

## Registering OSS Plugins 

When building the OSS components, the default plugin library may be overwritten
or given precendence to the OSS plugin library. As long as you have both .so
library files, you can load both:

Build OSS Components:
```
$ wget https://raw.githubusercontent.com/rmccorm4/tensorrt-utils/master/OSS/build_OSS.sh
$ source build_OSS.sh
```

Load and list multiple plugin libraries:
```
$ python list_plugins.py --plugins TensorRT/build/out/libnvinfer_plugin.so
2020-03-13 22:18:51 - __main__ - INFO - Loading plugin library: /mnt/TensorRT/build/out/libnvinfer_plugin.so
2020-03-13 22:18:51 - __main__ - INFO - Registering plugins...
2020-03-13 22:18:51 - __main__ - INFO - Registered Plugin Names:
['RnRes2Br2bBr2c_TRT',
 'RnRes2Br1Br2c_TRT',
 'CgPersistentLSTMPlugin_TRT',
 'SingleStepLSTMPlugin',
 'CustomEmbLayerNormPluginDynamic',    <------ OSS Plugins added
 'CustomFCPluginDynamic',              <------ OSS Plugins added
 'CustomGeluPluginDynamic',            <------ OSS Plugins added
 'CustomQKVToContextPluginDynamic',    <------ OSS Plugins added
 'CustomSkipLayerNormPluginDynamic',   <------ OSS Plugins added
 ...
 'SpecialSlice_TRT',
 'InstanceNormalization_TRT']
```

## Registering Custom Plugins

Same concept as above for OSS plugins, just giving an example to be extra clear:
```
$ pushd CustomIPluginV2/
$ make
g++ -g -std=c++11 -DNDEBUG -fPIC -MD -MP -I. -I/usr/local/cuda/include -I/usr/src/tensorrt/include -o CustomPlugin.o -c CustomPlugin.cpp
g++ -g -std=c++11 -DNDEBUG -shared -o CustomPlugin.so CustomPlugin.o -L/usr/local/cuda/lib64 -L/usr/src/tensorrt/lib  -lnvinfer -lcudart

$ popd
$ python list_plugins.py --plugins CustomIPluginV2/CustomPlugin.so
2020-03-13 22:16:06 - __main__ - INFO - Loading plugin library: CustomIPluginV2/CustomPlugin.so
Plugin attribute number: 5
2020-03-13 22:16:06 - __main__ - INFO - Registering plugins...
2020-03-13 22:16:06 - __main__ - INFO - Registered Plugin Names:
['RnRes2Br2bBr2c_TRT',
 'RnRes2Br1Br2c_TRT',
 'CgPersistentLSTMPlugin_TRT',
 'SingleStepLSTMPlugin',
 'CustomPlugin',           <------------- Custom plugin registered
 ...
 'SpecialSlice_TRT',
 'InstanceNormalization_TRT']
```
