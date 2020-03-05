import os
import docker

docker_client = docker.from_env(version="auto")

# TODO: argparse
model = "model.onnx"
#OSS_SCRIPT = "https://raw.githubusercontent.com/rmccorm4/tensorrt-utils/master/OSS/build_OSS.sh"
OSS_SCRIPT = "/opt/tensorrt/install_opensource.sh"

# Requires Docker >= 19.03
gpus = "all"

configs = {}
config0 = {}
# TODO: Include nightly build
config0["image"] = "nvcr.io/nvidia/tensorrt:19.12-py3"
config0["pre_command"] = ["/bin/bash", "-c", "chmod +x {}; {}".format(OSS_SCRIPT, OSS_SCRIPT)]
config0["command"] = ["trtexec --onnx={}".format(model)]
config0["post_command"] = []
config0["gpus"] = gpus

command = config0["pre_command"] + config0["command"]
print("Command:", command)
output = docker_client.containers.run(config0["image"], command=command, runtime="nvidia", stream=True)#, gpus=config0["gpus"])
for line in output:
    print(line)

"""
result = container.wait()
if result['StatusCode'] != 0:
    print(result)
    print(container.logs())
    print("Error running container!")
"""
