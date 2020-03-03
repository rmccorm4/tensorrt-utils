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

import warnings
# Filter numpy warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
# Filter tensorflow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from uff.model import uff_pb2


def parse_uff_metadata(filename):
    """
    Parses a UFF file and returns its input/output metadata as an dictionary.

    Args:
        filename (str): Path to the UFF file to parse.

    Returns:
        Dict: A dictionary of the UFF model's inputs and outputs.

    >>> parse_uff_metadata('model.uff')
    {'Inputs': [{'name': 'input_image', 'shape': [1, 224, 224, 3]}],
     'Outputs': [{'MarkOutput_0', 'name:'}]}
    """
    with open(filename, "rb") as f:
        model = uff_pb2.MetaGraph()
        # Protobuf API
        model.ParseFromString(f.read())

    inputs, outputs = [], []
    for graph in model.graphs:
        for node in graph.nodes:
            if node.operation == "Input":
                shape = list(node.fields["shape"].i_list.val)
                inputs.append({"name": node.id, "shape":shape})

            elif node.operation == "MarkOutput":
                outputs.append({"name:", node.id})

    metadata = {"Inputs": inputs, "Outputs": outputs}
    return metadata


if __name__ == '__main__':
    import argparse
    from pprint import pprint
    parser = argparse.ArgumentParser("Parse a UFF file and return it's metadata.")
    parser.add_argument("--uff", required=True, type=str, help="UFF file to parse.")
    args = parser.parse_args()

    metadata = parse_uff_metadata(args.uff)
    pprint(metadata)
