# Copyright 2019 NVIDIA Corporation
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

import logging

import numpy as np
from PIL import Image


logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def preprocess_imagenet(image, channels=3, height=224, width=224):
    """Pre-processing for Imagenet-based Image Classification Models:
        resnet50, vgg16, mobilenet, etc. (Doesn't seem to work for Inception)

    Parameters
    ----------
    image: PIL.Image
        The image resulting from PIL.Image.open(filename) to preprocess
    channels: int
        The number of channels the image has (Usually 1 or 3)
    height: int
        The desired height of the image (usually 224 for Imagenet data)
    width: int
        The desired width of the image  (usually 224 for Imagenet data)

    Returns
    -------
    img_data: numpy array
        The preprocessed image data in the form of a numpy array

    """
    # Get the image in CHW format
    resized_image = image.resize((width, height), Image.ANTIALIAS)
    img_data = np.asarray(resized_image).astype(np.float32)

    if len(img_data.shape) == 2:
        # For images without a channel dimension, we stack
        img_data = np.stack([img_data] * 3)
        logger.debug("Received grayscale image. Reshaped to {:}".format(img_data.shape))
    else:
        img_data = img_data.transpose([2, 0, 1])

    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    assert img_data.shape[0] == channels

    for i in range(img_data.shape[0]):
        # Scale each pixel to [0, 1] and normalize per channel.
        img_data[i, :, :] = (img_data[i, :, :] / 255 - mean_vec[i]) / stddev_vec[i]

    return img_data


def preprocess_inception(image, channels=3, height=224, width=224):
    """Pre-processing for InceptionV1. Inception expects different pre-processing
    than {resnet50, vgg16, mobilenet}. This may not be totally correct,
    but it worked for some simple test images.

    Parameters
    ----------
    image: PIL.Image
        The image resulting from PIL.Image.open(filename) to preprocess
    channels: int
        The number of channels the image has (Usually 1 or 3)
    height: int
        The desired height of the image (usually 224 for Imagenet data)
    width: int
        The desired width of the image  (usually 224 for Imagenet data)

    Returns
    -------
    img_data: numpy array
        The preprocessed image data in the form of a numpy array

    """
    # Get the image in CHW format
    resized_image = image.resize((width, height), Image.BILINEAR)
    img_data = np.asarray(resized_image).astype(np.float32)

    if len(img_data.shape) == 2:
        # For images without a channel dimension, we stack
        img_data = np.stack([img_data] * 3)
        logger.debug("Received grayscale image. Reshaped to {:}".format(img_data.shape))
    else:
        img_data = img_data.transpose([2, 0, 1])

    return img_data
