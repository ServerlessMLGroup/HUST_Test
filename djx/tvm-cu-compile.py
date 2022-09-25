import onnx
from tvm.contrib.download import download_testdata
from PIL import Image
import numpy as np
import tvm.relay as relay
from tvm.driver import tvmc
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

    model = tvmc.load("/model/resnet50-v2-7.onnx") #Step 1: Load

    package = tvmc.compile(model, target="cuda", package_path="/cu")