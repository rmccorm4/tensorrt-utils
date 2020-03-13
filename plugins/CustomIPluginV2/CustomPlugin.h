#ifndef CUSTOM_PLUGIN_H
#define CUSTOM_PLUGIN_H

#include "NvInferPlugin.h"

#include <vector>
#include <string.h>
#include <iostream>

using namespace nvinfer1;
using namespace std;

class CustomPlugin : public IPluginV2
{
public:
    CustomPlugin(const string name);

    CustomPlugin() = delete;

    int getNbOutputs() const override;

    Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) override;

    int initialize() override;

    void terminate() override;

    size_t getWorkspaceSize(int batchSize) const override;

    int enqueue(int batchSize, const void *const *inputs, void **outputs, void *workspace, cudaStream_t stream) override;

    size_t getSerializationSize() const override;

    void serialize(void *buffer) const override;

    void configureWithFormat(const Dims *inputDims, int nbInputs, const Dims *outputDims, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) override;

    bool supportsFormat(DataType type, PluginFormat format) const override;

    const char *getPluginType() const override;

    const char *getPluginVersion() const override;

    void destroy() override;

    IPluginV2 *clone() const override;

    void setPluginNamespace(const char *pluginNamespace) override;

    const char *getPluginNamespace() const override;

private:
    const string mLayerName;
    string mNamespace;
};

class CustomPluginCreator : public IPluginCreator
{
public:
    CustomPluginCreator();

    const char *getPluginName() const override;

    const char *getPluginVersion() const override;

    const PluginFieldCollection *getFieldNames() override;

    IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) override;

    IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) override;

    void setPluginNamespace(const char *pluginNamespace) override;

    const char *getPluginNamespace() const override;
private:
    static PluginFieldCollection mFC;
    static vector<PluginField> mPluginAttributes;
    string mNamespace;
};

#endif /* CustomPlugin.h */
