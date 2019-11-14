# OSS

To build the [TensorRT OSS Components](https://github.com/NVIDIA/TensorRT) **interactively** inside of a TensorRT NGC Container,
you can run the following [build_OSS.sh](build_OSS.sh) script inside of the container:

```bash
# Pull and Run TensorRT Container from NGC
docker run --runtime=nvidia -it nvcr.io/nvidia/tensorrt:19.10-py3

# Build OSS Components from https://github.com/NVIDIA/TensorRT
source build_OSS.sh
```

Alternatively, you can take the commands from [build_OSS.sh](build_OSS.sh) and
create a custom `Dockerfile` that pulls `FROM: nvcr.io/nvidia/tensorrt:19.10-py3`
and then proceeds to run the build commands, similar to: https://github.com/NVIDIA/TensorRT/blob/master/docker/ubuntu-18.04.Dockerfile
