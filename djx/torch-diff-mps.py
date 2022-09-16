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
                     (func_file_name, current_time.month, current_time.day, current_time.hour))
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


def main():
    worker_list = []
    worker_meg_list = []
    for i in range(1, 2):
        p_parent_worker, p_child_worker = mp.Pipe()
        os.environ['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'] = "10"
        worker = WorkerProc(("worker-10-%d" % i), p_child_worker, mps_percentage=10, batch_size=8, nruns=300)
        worker.start()
        worker_meg_list.append(p_parent_worker)
        worker_list.append(worker)

    p_parent_worker, p_child_worker = mp.Pipe()
    os.environ['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'] = "90"
    worker = WorkerProc(("worker-90-%d" % 1), p_child_worker, mps_percentage=90, batch_size=128, nruns=300)
    worker.start()
    worker_meg_list.append(p_parent_worker)
    worker_list.append(worker)

    for worker_channel in worker_meg_list:
        worker_channel.send('BEGIN')

    for worker in worker_list:
        worker.join()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
