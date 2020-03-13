# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

CUDA_PATH = /usr/local/cuda
TRT_RELEASE = /usr/src/tensorrt
GCC = g++
NVCC = $(CUDA_PATH)/bin/nvcc
CCFLAGS = -g -std=c++11 -DNDEBUG
INCLUDES += -I. -I$(CUDA_PATH)/include -I${TRT_RELEASE}/include
LDFLAGS := -L${CUDA_PATH}/lib64 -L${TRT_RELEASE}/lib $(LDFLAGS)
LDFLAGS += -lnvinfer -lcudart

SO = CustomPlugin.so
OBJ = $(shell find . -name '*.o')
D = $(shell find . -name '*.d')
DEP = $(OBJ:.o=.d)

all: $(SO) $(CUDA_BIN)

CustomPlugin.so: CustomPlugin.o

-include $(DEP)

clean:
	rm -rf $(SO) $(CUDA_BIN) $(OBJ) $(D)

%.o: %.cpp
	$(GCC) $(CCFLAGS) -fPIC -MD -MP $(INCLUDES) -o $@ -c $<

%.o: %.cu
	$(NVCC) $(CCFLAGS) -M -MT $@ $(INCLUDES) -o $(@:.o=.d) $<
	$(NVCC) $(CCFLAGS) $(INCLUDES) -Xcompiler -fPIC -o $@ -c $<

$(SO):
	$(GCC) $(CCFLAGS) -shared -o $@ $+ $(LDFLAGS)
