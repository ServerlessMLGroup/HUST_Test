import argparse
import numpy as np
import torch
import time


def benchmark(model, input_shape=(8, 3, 224, 224), dtype='fp32', nwarmup=50, nruns=100):
    input_data = torch.randn(input_shape)
    device = torch.device("cuda:%d" % args.cuda_device if torch.cuda.is_available() else "cpu")
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
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # args parse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resnet_times', type=int, default=1)
    parser.add_argument('--cuda_device', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()

    # torch.cuda.set_device(args.cuda_device)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.cuda_device
    device = torch.device("cuda:%d" % args.cuda_device if torch.cuda.is_available() else "cpu")
    # resnet stage
    model = torch.hub.load('/root/.cache/torch/hub/pytorch_vision_v0.10.0', 'resnet152', source="local",
                           pretrained=True)
    # print("resnet model load cost:%s" % ((resnet_load)))
    # model = model.eval().to(device)
    model.to(device)
    model.eval()
    benchmark(model=model, input_shape=(args.resnet_batch_size, 3, 224, 224))