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


def benchmark(model, input_shape=(1, 3, 224, 224), dtype='fp32', nwarmup=50, nruns=100):
    input_data = torch.randn(input_shape)
    device = torch.device("cuda")
    input_data = input_data.to(device)
    if dtype == 'fp16':
        input_data = input_data.half()

    print("Warm up ...")
    with torch.no_grad():
        for _ in range(nwarmup):
            features = model(input_data)
    torch.cuda.synchronize()
    print("Start timing ...")
    #timings = []
    #with torch.no_grad():
     #   for i in range(1, nruns + 1):
            #start_time = time.time()
            #features = model(input_data)
            #torch.cuda.synchronize()
          #  end_time = time.time()
           # timings.append(end_time - start_time)
            #if i % 10 == 0:
             #   print('Iteration %d/%d, ave batch time %.2f ms' % (i, nruns, np.mean(timings) * 1000))

   # print("Input shape:", input_data.size())
   # print("Output features size:", features.size())
   # print('Average batch time: %.2f ms' % (np.mean(timings) * 1000))


if __name__ == '__main__':
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # mps_controller.openMPS(gpu_no, mps_percentage)
    # torch.cuda.set_device(args.cuda_device)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.cuda_device
    #device = torch.device("cuda:%d" % gpu_no if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")
    print("device = ", device)
    # resnet stage
    # model = torch.hub.load('/root/.cache/torch/hub/pytorch_vision_v0.10.0', 'resnet152', source="local",
    #                       pretrained=True)
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    # print("resnet model load cost:%s" % ((resnet_load)))
    # model = model.eval().to(device)
    model.to(device)
    model.eval()
    batch_size = 1
    num_class = 1000
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
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    # print(output[0])
    print(output.flatten()[0:10])

    # mps_controller.closeMPS(gpu_no)
