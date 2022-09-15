import argparse
from queue import Queue
from multiprocessing import Process
import torch
import time
from pipeswitch.worker_common import ModelSummary
from pipeswitch.worker_terminate import WorkerTermThd
from util.util import timestamp
import sys
import os
import numpy as np
import torch
import time


class InferenceWorker(Process):
    def __init__(self, sm_limit, server_pid, pipe, number):
        super(WorkerProc, self).__init__()
        self.sm_limit = sm_limit
        self.server_pid = server_pid
        self.pipe = pipe
        self.number = number

    def benchmark(model, input_shape=(8, 3, 224, 224), dtype='fp32', nwarmup=50, nruns=100):
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

    def run(self):
        timestamp('worker', 'start')
        str = "worker " + str(number) + " start:"
        print(str)
        agent, model_name = self.pipe.recv()
            model_summary = model_map[hash(model_name)]
            TERMINATE_SIGNAL[0] = 1
            timestamp('worker_proc', 'get_model')

            data_b = self.pipe.recv()
            timestamp('worker_proc', 'get_data')

            # start doing inference
            # frontend_scheduler will directly put
            # mod_list[0] in to self.complete_queue_trans
            try:
                if 'training' in model_name:
                    self.pipe.send('FNSH')
                    agent.send(b'FNSH')

                with torch.cuda.stream(model_summary.cuda_stream_for_computation):
                    output = model_summary.execute(data_b)
                    print('Get output', output)
                    del output

                if 'inference' in model_name:
                    self.pipe.send('FNSH')
                    agent.send(b'FNSH')
            except Exception as e:
                complete_queue.put('FNSH')

            # start do cleaning
            TERMINATE_SIGNAL[0] = 0
            timestamp('worker_comp_thd', 'complete')

            model_summary.reset_initialized(model_summary.model)

            # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

            # mps_controller.openMPS(gpu_no, mps_percentage)
            # torch.cuda.set_device(args.cuda_device)
            # os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.cuda_device

            # get pid
            # pid = os.getpid()
            pid = 359
            print(pid)
            # set constarinment
            cmd = "echo set_active_thread_percentage " + str(pid) + " 20 | nvidia-cuda-mps-control"
            print(cmd)
            # os.system("get_device_client_list")
            # os.system(cmd)

            device = torch.device("cuda:%d" % gpu_no if torch.cuda.is_available() else "cpu")
            print("device = ", device)

            # resnet stage
            # model = torch.hub.load('/root/.cache/torch/hub/pytorch_vision_v0.10.0', 'resnet152', source="local",
            #                       pretrained=True)
            model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
            # print("resnet model load cost:%s" % ((resnet_load)))
            # model = model.eval().to(device)
            model.to(device)
            model.eval()
            benchmark(model=model, input_shape=(batch_size, 3, 224, 224))

            # mps_controller.closeMPS(gpu_no)

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_no', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--mps', type=int, default=100)
args = parser.parse_args()

batch_size = args.batch_size
mps_percentage = args.mps
gpu_no = args.gpu_no



sys.path.append("..")



os.environ['CUDA_VISIBLE_DEVICES'] = "%d" % gpu_no




