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


def benchmark(log_file_name, worker_name, device, model, Stream, input_shape=(8, 3, 224, 224), dtype='fp32', nwarmup=50,
              nruns=500):
    log_file_handler = open(log_file_name, 'a+', encoding='utf-8')
    input_data = torch.randn(input_shape)
    input_data = input_data.to(device)
    if dtype == 'fp16':
        input_data = input_data.half()

    print("%s:Warm up ..." % worker_name)
<<<<<<< HEAD
    with torch.no_grad():
        for _ in range(nwarmup):
            features = model(input_data)
    #torch.cuda.synchronize()
    print("[%s]%s:Start timing ..." % (time.time(), worker_name))
    timings = []
    with torch.no_grad():
        for i in range(1, nruns + 1):
            start_time = time.time()
            features = model(input_data)
            #torch.cuda.synchronize()
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
=======

    with torch.cuda.stream(Stream):
        print("current stream: {}".format(torch.cuda.current_stream()))
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

class WorkerThre(threading.Thread):
    def __init__(self, name, Stream,batch_size=32, nruns=500):
        super(WorkerThre, self).__init__()
>>>>>>> 0515e2d88d72f6a7624cb57a39b449295d4517d0
        self.name = name
        self.Stream = Stream
        # self.logger = log.get_logger(name, "torch-diff-mps")
        self.batch_size = batch_size
        self.log_file = init_file("torch-diff-stream-%s" % name)
        self.nruns = nruns

    def run(self):
        device = torch.device("cuda:%d" % gpu_no if torch.cuda.is_available() else "cpu")
        # self.logger.info("torch.cuda.current_device():", torch.cuda.current_device())
        print("torch.cuda.current_device():", torch.cuda.current_device())
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
        with torch.cuda.stream(self.Stream):
            model.to(device)
            model.eval()
        benchmark(log_file_name=("Stream-%d" % self.name), worker_name=("Worker-Stream-%d" % self.name), device=device, model=model,
                  Stream=self.Stream, input_shape=(self.batch_size, 3, 224, 224), nruns=self.nruns)


    def main():
        torch.randn(1024, device='cuda')
        s1 = torch.cuda.Stream()
        s2 = torch.cuda.Stream()
        s3 = torch.cuda.Stream()
        s4 = torch.cuda.Stream()

        worker1 = WorkerThre(("worker-Stream-%d" % 1), s1, batch_size=32, nruns=300)
        worker2 = WorkerThre(("worker-Stream-%d" % 2), s2, batch_size=32, nruns=300)
        worker3 = WorkerThre(("worker-Stream-%d" % 3), s3, batch_size=32, nruns=300)
        worker4 = WorkerThre(("worker-Stream-%d" % 4), s4, batch_size=32, nruns=300)



<<<<<<< HEAD
    model4 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
    model4.to(device)
    model4.eval()

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
        time.sleep(10)
        benchmark(log_file_name="Stream-1", worker_name="Stream-1", device=device, model=model1,
              input_shape=(batch_size, 3, 224, 224), nruns= 300)
=======
        worker1.start()
        timestamp('Worker-1', 'start_compution')
        worker2.start()
        timestamp('Worker-2', 'start_compution')
        worker3.start()
        timestamp('Worker-3', 'start_compution')
        worker4.start()
        timestamp('Worker-4', 'start_compution')
>>>>>>> 0515e2d88d72f6a7624cb57a39b449295d4517d0



    if __name__ == '__main__':
        main()

    # mps_controller.closeMPS(gpu_no)
