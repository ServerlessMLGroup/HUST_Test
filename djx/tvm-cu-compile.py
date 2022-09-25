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

if __name__ == '__main__':
    model = tvmc.load("/model/resnet50-v2-7.onnx") #Step 1: Load

    package = tvmc.compile(model, target="cuda", package_path="/cu")