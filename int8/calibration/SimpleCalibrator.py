import os
import logging
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

class SimpleCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, network, config):
        super().__init__()

        # TODO: Not sure of difference between get_batch_size and what's returned in get_batch ?
        # Notes:
        #     get_batch_size() is required to return non-null value
        #     get_batch_size() can return  0 with seemingly no consequence with/without calibration cache
        #     get_batch_size() can return -1 with seemingly no consequence with/without calibration cache
        #     get_batch() seems to do the work, as long as get_batch_size doesn't throw an error
        self.batch_size = -1
        self.shapes = []
        self.device_inputs = None
        num_calibration_samples = 1000
        self.iterator = (i for i in range(num_calibration_samples))
        self.cache_file = "simple_calibration.cache"
        self.network = network
        self.calib_profile = config.get_calibration_profile()

    def get_batch(self, input_names, p_str=None):
        try:
            # Use iterator here to avoid having to pass input names to constructor
            next(self.iterator) 
            if not self.shapes:
                self.set_shapes(input_names)

            if not self.device_inputs:
                self.device_inputs = [cuda.mem_alloc(np.zeros(s, dtype=np.float32).nbytes) for s in self.shapes]

            if not self.batch_size:
                # Get batch size from first input in calibration shapes. Assumes batch sizes
                # are the same for every input
                self.batch_size = self.shapes[0][0]

            batches = [np.random.random(s).astype(np.float32) for s in self.shapes]
            for i in range(len(batches)):
                cuda.memcpy_htod(self.device_inputs[i], batches[i])

            return [int(d) for d in self.device_inputs]
        except StopIteration:
            return None

    def get_batch_size(self):
        return self.batch_size

    def set_shapes(self, input_names):
        if self.calib_profile:
            self.shapes = [self.calib_profile.get_shape(name) for name in input_names]
        else:
            self.shapes = []
            # This assumes order of input_names matches the network input indices
            for i, name in enumerate(input_names):
                shape = self.network.get_input(i).shape
                _shape = []
                found_dynamic = False
                # Replace any dynamic dimensions with ones if any
                for dim in shape:
                    if dim < 0:
                        dim = 1
                        found_dynamic = True
                    
                    _shape.append(dim)

                _shape = tuple(_shape)
                if found_dynamic:        
                    logger.warning("[{}] has dynamic shape: {}. Set to {} instead.".format(name, shape, _shape))

                self.shapes.append(_shape)

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                logger.info("Using calibration cache to save time: {:}".format(self.cache_file))
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            logger.info("Caching calibration data for future use: {:}".format(self.cache_file))
            f.write(cache)
