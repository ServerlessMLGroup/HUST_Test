import argparse
import datetime
import time

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_no', type=int, default=0)
args = parser.parse_args()

gpu_no = args.gpu_no

import sys

sys.path.append("..")

import os

os.environ['CUDA_VISIBLE_DEVICES'] = "%d" % gpu_no

from multiprocessing import Process
import torch
import numpy as np
import torch.multiprocessing as mp


def init_file(func_file_name):
    current_time = datetime.datetime.now()
    dir_name, _ = os.path.split(os.path.abspath(__file__))
    log_file_path = os.path.dirname(dir_name) + '/logs'
    log_file_name = ('%s_%s_%s_%s.txt' %
                     (func_file_name, current_time.month, current_time.day, (current_time.hour + 8) % 24))
    if not os.path.exists(log_file_path):
        os.mkdir(log_file_path)

    file_full_name = log_file_path + "/" + log_file_name
    return file_full_name


def benchmark(log_file_name, worker_name, device, model, input_shape=(8, 3, 224, 224), dtype='fp32', nwarmup=50,
              nruns=500):
    log_file_handler = open(log_file_name, 'a+', encoding='utf-8')
    input_data = torch.randn(input_shape)
    input_data = input_data.to(device)
    if dtype == 'fp16':
        input_data = input_data.half()

    print("%s:Warm up ..." % worker_name)
    with torch.no_grad():
        for _ in range(nwarmup):
            features = model(input_data)
    torch.cuda.synchronize()
    print("[%s]%s:Start timing ..." % (time.time(), worker_name))
    timings = []
    with torch.no_grad():
        for i in range(1, nruns + 1):
            start_time = time.time()
            features = model(input_data)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
            if i % 10 == 0:
                # logger.info('Iteration %d/%d, %d-%d ave batch time %.2f ms' % (i, nruns, i, i - 10,
                # np.mean(timings) * 1000))
                print('%s:Iteration %d/%d, %d-%d ave batch time %.2f ms' % (
                    worker_name, i, nruns, i, i - 10, np.mean(timings) * 1000))
                log_file_handler.write('[%s] %s:Iteration %d/%d, %d-%d ave batch time %.2f ms\n' % (
                    time.time(), worker_name, i, nruns, i, i - 10, np.mean(timings) * 1000))
                timings.clear()
    print("[%s]%s:End!--------" % (time.time(), worker_name))
    log_file_handler.close()
    # logger.info("Input shape:", input_data.size())
    # print("Input shape:", input_data.size())
    # logger.info("Output features size:", features.size())
    # print("Output features size:", features.size())


class WorkerProc(Process):
    def __init__(self, name, start_pipe, mps_percentage, batch_size=32, nruns=500):
        super(WorkerProc, self).__init__()
        self.name = name
        # self.logger = log.get_logger(name, "torch-diff-mps")
        self.start_pipe = start_pipe
        self.mps_percentage = mps_percentage
        self.batch_size = batch_size
        self.log_file = init_file("torch-diff-mps-%s" % name)
        self.nruns = nruns

    def run(self):
        begin_meg = self.start_pipe.recv()
        if begin_meg != 'BEGIN':
            # self.logger.error('%s do not receive BEGIN!' % self.name)
            print('%s do not receive BEGIN!' % self.name)
        # cmd = 'echo set_active_thread_percentage 213 %d | nvidia-cuda-mps-control' % self.mps_percentage
        # os.system(cmd)
        # self.logger.info(cmd)
        # print(cmd)
        device = torch.device("cuda:%d" % gpu_no if torch.cuda.is_available() else "cpu")
        # self.logger.info("torch.cuda.current_device():", torch.cuda.current_device())
        print("torch.cuda.current_device():", torch.cuda.current_device())
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
        model.to(device)
        model.eval()
        benchmark(log_file_name=self.log_file, worker_name=self.name, device=device, model=model,
                  input_shape=(self.batch_size, 3, 224, 224), nruns=self.nruns)


if __name__ == '__main__':
    device = torch.device("cuda:%d" % gpu_no if torch.cuda.is_available() else "cpu")
    print("device = ", device)

    batch_size=32

    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)

    model.to(device)
    model.eval()

    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()
    s3 = torch.cuda.Stream()
    s4 = torch.cuda.Stream()
    default_stream = torch.cuda.current_stream()
    print("Default Stream: {}".format(default_stream))
    # 等待创建A的stream执行完毕
    torch.cuda.Stream.synchronize(default_stream)

    with torch.cuda.stream(s1):
        print("current stream: {}".format(torch.cuda.current_stream()))
        benchmark(log_file_name=“Stream-1”, worker_name=“Stream-1”, device=device, model=model,
              input_shape=(batch_size, 3, 224, 224), nruns=self.nruns)

    with torch.cuda.stream(s2):
        print("current stream: {}".format(torch.cuda.current_stream()))
        benchmark(log_file_name=“Stream - 2”, worker_name =“Stream-2”, device = device, model = model,
              input_shape = (batch_size,3, 224,224), nruns = self.nruns)

    with torch.cuda.stream(s3):
        print("current stream: {}".format(torch.cuda.current_stream()))
        benchmark(log_file_name=“Stream-3”, worker_name=“Stream-3”, device=device, model=model,
              input_shape=(batch_size, 3, 224, 224), nruns=self.nruns)

    with torch.cuda.stream(s4):
        print("current stream: {}".format(torch.cuda.current_stream()))
        benchmark(log_file_name=“Stream-4”, worker_name=“Stream-4”, device=device, model=model,
              input_shape=(batch_size, 3, 224, 224), nruns=self.nruns)



    # mps_controller.closeMPS(gpu_no)