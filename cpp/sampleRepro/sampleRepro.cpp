#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace nvinfer1;

bool file_exists(const char* filename) {
  std::ifstream infile(filename);
  return infile.good();
}

int main(int argc, char** argv) {
  auto onnxPath = "model.onnx";
  auto enginePath = "model.engine";
  if (!file_exists(onnxPath)) {
    gLogError << "File doesn't exist: " << onnxPath << " - exiting..." << std::endl;
    exit(1);
  }

  // Create IBuilder and and IBuilderConfig for creating ICudaEngine
  auto builder = createInferBuilder(gLogger);
  auto config = builder->createBuilderConfig();
  config->setMaxWorkspaceSize(1 << 24);
  config->setFlag(BuilderFlag::kFP16);

  // Create and parse INetworkDefinition from ONNX model
  auto network = builder->createNetworkV2(1U << static_cast<int>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
  auto parser = nvonnxparser::createParser(*network, gLogger);
  auto verbosity = 3;
  auto parsed = parser->parseFromFile(onnxPath, verbosity);
  std::cout << "Parsing success: " << parsed << std::endl;
  if (!parsed) {
    gLogError << "Failed to parse: " << onnxPath << " - exiting..." << std::endl;
    exit(1);
  }

  // Build ICudaEngine with IBuilderConfig
  auto engine = builder->buildEngineWithConfig(*network, *config);
    
  // Serialize ICudaEngine to file for future use
  std::ofstream p(enginePath, std::ios::binary);
  if (!p) {
    return false;
  }
  nvinfer1::IHostMemory* ptr = engine->serialize();
  assert(ptr);
  p.write(reinterpret_cast<const char*>(ptr->data()), ptr->size());
  ptr->destroy();
  p.close();
  gLogInfo << "TensorRT engine file saved to: " << enginePath << std::endl;
}
