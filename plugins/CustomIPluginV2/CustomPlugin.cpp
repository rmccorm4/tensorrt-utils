/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "CustomPlugin.h"
#include "NvInfer.h"

#include <vector>
#include <string.h>
#include <iostream>

using namespace nvinfer1;
using namespace std;

namespace
{
    static const char *CUSTOM_PLUGIN_NAME{"CustomPlugin"};
    static const char *CUSTOM_PLUGIN_VERSION{"1"};
}

PluginFieldCollection CustomPluginCreator::mFC{};
vector<PluginField> CustomPluginCreator::mPluginAttributes;

CustomPlugin::CustomPlugin(const string name) :mLayerName(name) {}


/* ----- Custom Plugin ----- */


/*
 * Description
 *     Initialize the layer for execution. 
 *
 *     This is called when the engine is created.
 *
 * Returns
 *     0 for success, else non-zero (which will cause engine termination).
 */
int CustomPlugin::initialize()
{
    return 0; 
}


/* 
 * Description
 *     Release resources acquired during plugin layer initialization.
 *
 *     This is called when the engine is destroyed. 
 */
void CustomPlugin::terminate() {}


/*
 * Description
 *     Get the number of outputs from the layer.
 *
 *     This function is called by the implementations of INetworkDefinition and
 *     IBuilder. In particular, it is called prior to any call to initialize().
 *
 * Returns
 *     The number of outputs.
 */
int CustomPlugin::getNbOutputs() const
{
    return 1;
}


/*
 * Description
 *     Get the dimensions of an output tensor.
 *
 *     This function is called by the implementations of INetworkDefinition and
 *     IBuilder. In particular, it is called prior to any call to initialize().
 *
 * Parameters
 *     index        The index of the output tensor.
 *     inputs       The input tensors.
 *     nbInputDims  The number of input tensors.
 *
 * Returns
 *     The output dimensions of a layer.
 */
Dims CustomPlugin::getOutputDimensions(int index, const Dims *inputs, int nbInputDims) 
{
    // TODO: Add a meaningful output
    Dims output;
    return output;
}


/*
 * Description
 *     Execute the layer asynchronously.
 *
 *     This is called manually by the user for async inference.
 *
 * Parameters
 *     batchSize    The number of inputs in the batch.
 *     inputs       The memory for the input tensors.
 *     outputs      The memory for the output tensors.
 *     workspace    Workspace for execution.
 *     stream       The stream in which to execute the kernels.
 *
 * Returns
 *     0 for success, else non-zero (which will cause engine termination). 
 */
int CustomPlugin::enqueue(int batchSize, const void *const *inputs, void **outputs, void *workspace, cudaStream_t stream)
{
    // 0 for success, else non-zero (which will cause engine termination).
    return -1; // TODO: Inference not implemented for this example yet.
}


/*
 * Description
 *     Find the workspace size required by the layer.
 *
 *     This function is called during engine startup, after initialize(). 
 *     
 *     The workspace size returned should be sufficient for any batch size up
 *     to the maximum.
 *
 * Returns
 *     The workspace size.
 */
size_t CustomPlugin::getWorkspaceSize(int batchSize) const
{
    return 0;
}


/*
 * Description
 *     Configure the layer.
 *
 *     The dimensions passed here do not include the outermost batch size
 *     (i.e. for 2-D image networks, they will be 3-D CHW dimensions).
 *
 *     This function is called by the builder prior to initialize(). 
 *
 *     It provides an opportunity for the layer to make algorithm choices on
 *     the basis of its weights, dimensions, and maximum batch size.
 *
 * Parameters
 *     inputDims    The input tensor dimensions.
 *     nbInputs     The number of inputs.
 *     outputDims   The output tensor dimensions.
 *     nbOutputs    The number of outputs.
 *     type         The data type selected for the engine.
 *     format       The format selected for the engine.
 *     maxBatchSize The maximum batch size.
 *
 * Warning
 *     For the format field, the values PluginFormat::kCHW4, 
 *     PluginFormat::kCHW16, and PluginFormat::kCHW32 will not be passed in,
 *     this is to keep backward compatibility with TensorRT 5.x series. Use 
 *     PluginV2IOExt or PluginV2DynamicExt for other PluginFormats.
 */
void CustomPlugin::configureWithFormat(const Dims *inputs, int nbInputs, const Dims *outputs, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize)
{
    // TODO: Add meaningful example or link to one.
}


/*
 * Description
 *     Check format support.
 *
 *     This function is called by the implementations of INetworkDefinition,
 *     IBuilder, and safe::ICudaEngine/ICudaEngine. In particular, it is called
 *     when creating an engine and when deserializing an engine.
 *
 * Parameters
 *     type     DataType requested.
 *     format   PluginFormat requested.
 * 
 * Returns
 *     true if the plugin supports the type-format combination, false otherwise.
 *
 * Warning
 *     For the format field, the values PluginFormat::kCHW4, 
 *     PluginFormat::kCHW16, and PluginFormat::kCHW32 will not be passed in,
 *     this is to keep backward compatibility with TensorRT 5.x series. Use 
 *     PluginV2IOExt or PluginV2DynamicExt for other PluginFormats.
 */
bool CustomPlugin::supportsFormat(DataType type, PluginFormat format) const
{
    return true;
}


/*
 * Description
 *     Find the size of the serialization buffer required.
 *
 * Returns
 *     The size of the serialization buffer.
 */
size_t CustomPlugin::getSerializationSize() const
{
    return sizeof(mLayerName);
}


/*
 * Description
 *     Serialize the layer. Useful for saving engines for future use.
 *
 * Parameters
 *     buffer   A pointer to a buffer to serialize data. Size of buffer must
 *              be equal to value returned by getSerializationSize.
 */
void CustomPlugin::serialize(void *buffer) const {}


/*
 * Description
 *     Destroy the plugin object. This will be called when the network, builder or engine is destroyed.
 */
void CustomPlugin::destroy() { delete this; }


/*
 * Description
 *     Clone the plugin object. This copies over internal plugin parameters and
 *     returns a new plugin object with these parameters.
 *
 * Returns
 *     A Plugin object.
 */
IPluginV2 *CustomPlugin::clone() const
{
    return new CustomPlugin(mLayerName);
}


/*
 * Description
 *     Set the namespace that this plugin object belongs to. Ideally, all plugin
 *     objects from the same plugin library should have the same namespace.
 */
void CustomPlugin::setPluginNamespace(const char *libNamespace)
{
    mNamespace = libNamespace;
}


/* 
 * Description
 *     Should match the plugin namespace returned by the corresponding
 *     plugin creator.
 *
 * Returns
 *     The namespace of the plugin object.
 */
const char *CustomPlugin::getPluginNamespace() const
{
    return mNamespace.c_str();
}


/*
 * Description
 *     Should match the plugin name returned by the corresponding 
 *     plugin creator.
 *
 * Returns
 *     The Plugin type.
 */
const char *CustomPlugin::getPluginType() const
{
    return CUSTOM_PLUGIN_NAME;
}


/* 
 * Description
 *     Should match the plugin version returned by the corresponding 
 *     plugin creator.
 *
 * Returns
 *     The Plugin version.
 */
const char *CustomPlugin::getPluginVersion() const
{
    return CUSTOM_PLUGIN_VERSION;
}


/* ----- Custom IPluginCreator ----- */


/*
 * Description
 *     Custom IPluginCreator Constructor
 */
CustomPluginCreator::CustomPluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("var1", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("var2", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("var3", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("var4", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("var5", nullptr, PluginFieldType::kINT32, 1));

    // Fill PluginFieldCollection with PluginField arguments metadata
    mFC.nbFields = mPluginAttributes.size();
    cout << "Plugin attribute number: " << mPluginAttributes.size() << endl;
    mFC.fields = mPluginAttributes.data();
}


/*
 * Returns
 *     A Plugin object, or nullptr in case of error.
 */
IPluginV2 *CustomPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc)
{
    const PluginField *fields = fc->fields;

    // TODO: Do something more meaningful here
    for (int i = 0; i < fc->nbFields; i++)
    {
        const char *name = fields[i].name;
        auto data = fields[i].data;
        cout << name << ": " << static_cast<int>(*(static_cast<const int *>(data))) << endl;
    }

    return new CustomPlugin(name);
}


/* 
 * Description
 *     Called during deserialization of plugin layer.
 *
 * Returns
 *     A Plugin object.
 */
IPluginV2 *CustomPluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength)
{
    return new CustomPlugin(name);
}


/*
 * Description
 *     Set the namespace of the plugin creator based on the plugin library it
 *     belongs to. This can be set while registering the plugin creator.
 */
void CustomPluginCreator::setPluginNamespace(const char *pluginNamespace)
{
    mNamespace = pluginNamespace;
}


/*
 * Description
 *     Should match the plugin namespace returned by the corresponding plugin.
 *
 * Returns
 *     The namespace of the plugin object.
 */
const char *CustomPluginCreator::getPluginNamespace() const
{
    return mNamespace.c_str();
}


/* 
 * Description
 *     Should match the plugin name returned by the corresponding plugin.
 *
 * Returns
 *     The Plugin name
 */
const char *CustomPluginCreator::getPluginName() const
{
    return CUSTOM_PLUGIN_NAME;
}


/*
 * Description
 *     Should match the plugin version returned by the corresponding plugin.
 *
 * Returns
 *     The Plugin version.
 */
const char *CustomPluginCreator::getPluginVersion() const
{
    return CUSTOM_PLUGIN_VERSION;
}


/*
 * Description
 *     Return a PluginFieldCollection (list) of PluginField objects that needs
 *     to be passed to createPlugin.
 *
 * Returns
 *     A PluginFieldCollection object.
 */
const PluginFieldCollection *CustomPluginCreator::getFieldNames()
{
    return &mFC;
}


/*
 * TensorRT also provides the ability to register a plugin by calling
 * REGISTER_TENSORRT_PLUGIN(pluginCreator) which statically registers the
 * Plugin Creator to the Plugin Registry. During runtime, the Plugin Registry
 * can be queried using the extern function getPluginRegistry(). The Plugin 
 * Registry stores a pointer to all the registered Plugin Creators and can be
 * used to look up a specific Plugin Creator based on the plugin name and version.
 */
REGISTER_TENSORRT_PLUGIN(CustomPluginCreator);
