import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_no', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--mps', type=int, default=100)
args = parser.parse_args()

batch_size = args.batch_size
mps_percentage = args.mps
gpu_no = args.gpu_no

import sys

sys.path.append("..")

import os
import numpy as np
import torch
import time


def benchmark(model, input_shape=(8, 3, 224, 224), dtype='fp32', nwarmup=50, nruns=100):
    input_data = torch.randn(input_shape)
    device = torch.device("cuda:%d" % gpu_no if torch.cuda.is_available() else "cpu")
    input_data = input_data.to(device)
    if dtype == 'fp16':
        input_data = input_data.half()

    print("Warm up ...")
    with torch.no_grad():
        #for _ in range(nwarmup):

        features = model(input_data)


if __name__ == '__main__':
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
    model.eval()
    model.cuda()
    benchmark(model=model, input_shape=(batch_size, 3, 224, 224))

    # mps_controller.closeMPS(gpu_no)
