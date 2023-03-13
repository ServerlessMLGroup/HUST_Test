import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_no', type=int, default=1)
args = parser.parse_args()

gpu_no = args.gpu_no

import sys

sys.path.append("..")

import os

#os.environ['CUDA_VISIBLE_DEVICES'] = "%d" % gpu_no

#os.system("nvcc getSize.cu -o getSize -lcuda")
print("before import torch:")
os.system("./get1Size")

import torch
#torch.backends.cudnn.deterministic = False
torch.backends.cudnn.enabled = False
#torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    device = torch.device("cuda:%d" % gpu_no if torch.cuda.is_available() else "cpu")
    input_data = torch.randn(2)
    input_data = input_data.to(device)
    os.system("./get1Size")

    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
    model.to(device)

    #torch.randn(1, device='cuda')
    #torch.cuda._lazy_init()1
    #print("device = ", device)
    print("after import torch:")

    os.system("./get1Size")
