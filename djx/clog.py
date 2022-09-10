import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_no', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=1)
args = parser.parse_args()

batch_size = args.batch_size
gpu_no = args.gpu_no

import sys

sys.path.append("..")

import os

os.environ['CUDA_VISIBLE_DEVICES'] = ("%d" % 1)

import numpy as np
import torch
import time


def benchmark(model, input_shape=(8, 3, 224, 224), dtype='fp32', nwarmup=50, nruns=1000):
    input_data = torch.randn(input_shape)
    device = torch.device("cuda:%d" % gpu_no if torch.cuda.is_available() else "cpu")
    input_data = input_data.to(device)
    if dtype == 'fp16':
        input_data = input_data.half()

    print("Warm up ...")
    with torch.no_grad():
        for _ in range(nwarmup):
            features = model(input_data)
    torch.cuda.synchronize()
    print("Start timing ...")
    timings = []
    with torch.no_grad():
        for i in range(1, nruns + 1):
            start_time = time.time()
            features = model(input_data)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
            if i % 10 == 0:
                print('Iteration %d/%d, ave batch time %.2f ms' % (i, nruns, np.mean(timings) * 1000))

    print("Input shape:", input_data.size())
    print("Output features size:", features.size())
    print('Average batch time: %.2f ms' % (np.mean(timings) * 1000))


if __name__ == '__main__':
    print("torch.cuda.device_count():", torch.cuda.device_count())
    print("torch.cuda.current_device():", torch.cuda.current_device())
    device = torch.device("cuda:%d" % gpu_no if torch.cuda.is_available() else "cpu")
    print("device =", device)
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
    model.to(device)
    model.eval()
    benchmark(model=model, input_shape=(batch_size, 3, 224, 224))
