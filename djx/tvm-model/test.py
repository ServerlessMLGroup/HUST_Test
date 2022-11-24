import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_no', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=1)
args = parser.parse_args()

batch_size = args.batch_size
gpu_no = args.gpu_no

import sys

sys.path.append("..")

import numpy as np
import torch


if __name__ == '__main__':
    device = torch.device("cuda")
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    model.to(device)
    model.eval()
    batch_size = 1
    image_shape = (3, 224, 224)
    data_shape = (batch_size,) + image_shape
    input_data = np.ones(data_shape).astype("float32")
    input_data = input_data * 10
    print(input_data)
    input_data = torch.from_numpy(input_data)
    
    device = torch.device("cuda")
    input_data = input_data.to(device)

    with torch.no_grad():
        output = model(input_data)
    torch.cuda.synchronize()
    # print(output[0])
    print(output[0][0:10])
