import tvm.relay as relay
import tvm
from tvm.contrib import graph_executor
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_no', type=int, default=0)
args = parser.parse_args()

gpu_no = args.gpu_no

import sys

sys.path.append("..")

import os

os.environ['CUDA_VISIBLE_DEVICES'] = ("%d" % args.gpu_no)

import torch

if __name__ == '__main__':
    print("torch.cuda.device_count():", torch.cuda.device_count())
    print("torch.cuda.current_device():", torch.cuda.current_device())
    device = torch.device("cuda:%d" % gpu_no if torch.cuda.is_available() else "cpu")
    print("device =", device)
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
    model.eval()
    target = "cuda"
    input_name = "data"
    shape_dict = {input_name: (1, 3, 224, 224)}
    mod, params = tvm.relay.frontend.from_pytorch(model, shape_dict)
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)

    dev = tvm.device(str(target), 0)
    module = graph_executor.GraphModule(lib["default"](dev))
    # package = tvmc.compile(model, target="cuda", package_path="./")