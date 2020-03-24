#!/usr/bin/env python3
import os
import re
import sys
import time
import docker
from pprint import pprint
from collections import defaultdict

# ONNX file in current directory
model = sys.argv[1]

# Docker API
client = docker.from_env()
# MUST PULL THESE IMAGES LOCALLY BEFORE RUNNING SCRIPT
#images = ["nvcr.io/nvidia/tensorrt:20.02-py3"]
images = ["nvcr.io/nvidia/tensorrt:19.12-py3", "nvcr.io/nvidia/tensorrt:20.02-py3"]
cmd_trtexec = ["cd /mnt", "&&", "trtexec", "--explicitBatch", "--onnx={}".format(model)]
# TODO: This won't work as is for TRT 6, need tp specify release/6.0 branch for TRT 6
#cmd_build_OSS = ["wget", "https://raw.githubusercontent.com/rmccorm4/tensorrt-utils/master/OSS/build_OSS.sh", "&&", "source", "build_OSS.sh"]
# TODO: This works for both TRT 6 + 7, but doesn't capture latest changes in master for TRT 7
cmd_build_OSS = ["/bin/bash", "/opt/tensorrt/install_opensource.sh"]
cmd_trtexec_OSS = cmd_build_OSS + ["&&"] + cmd_trtexec
commands = [cmd_trtexec, cmd_trtexec_OSS]

# Summary info
passes = []
fails = []
warnings = defaultdict(lambda:defaultdict(list))
errors = defaultdict(lambda:defaultdict(list))
start = time.time()

# Try each combination
for image in images:
    for command in commands:
        command = "/bin/bash -c '{}'".format(" ".join(command))
        container = client.containers.run(
                image,
                command=command,
                volumes={os.getcwd(): {"bind":"/mnt/", "mode":"rw"}},
                detach=True
        )
        # Stream output
        for line in container.logs(stream=True):
            line = line.decode().strip()
            print(line)
            err_regex = re.compile(r"\[[0-9]+\]", re.MULTILINE)
            if "[E]" in line or err_regex.search(line): 
                errors[image][command].append(line)
            elif "[W]" in line:
                warnings[image][command].append(line)

        result = container.wait()
        if result["StatusCode"] == 0:
            passes.append("✔️ PASS: {} - {}".format(image, command))
        else:
            fails.append("❌  FAIL: {} - {}: \n\t{}".format(image, command, "\n\t".join(errors[image][command])))

end = time.time()

print("==== SUMMARY ====")
print("Time taken: {:.2f} seconds".format(end-start))
[print(p) for p in passes]
[print(f) for f in fails]
