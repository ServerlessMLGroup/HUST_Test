import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_no', type=int, default=0)
args = parser.parse_args()

gpu_no = args.gpu_no

import sys

sys.path.append("..")

import os

os.environ['CUDA_VISIBLE_DEVICES'] = "%d" % gpu_no

from multiprocessing import Process
import torch.multiprocessing as mp

from util import log


class WorkerProc(Process):
    def __init__(self, name, start_pipe, mps_percentage, batch_size=32):
        super(WorkerProc, self).__init__()
        self.name = name
        # self.logger = log.get_logger(name, "test-multi-log", main_stdout)
        self.start_pipe = start_pipe
        self.mps_percentage = mps_percentage
        self.batch_size = batch_size

    def run(self):
        begin_meg = self.start_pipe.recv()
        if begin_meg != 'BEGIN':
            print('%s do not receive BEGIN!' % self.name)
        cmd = 'echo set_active_thread_percentage 213 %d | nvidia-cuda-mps-control' % self.mps_percentage
        print(cmd)


def main():
    worker_list = []
    p_parent_worker1, p_child_worker1 = mp.Pipe()
    worker1 = WorkerProc(("worker%d" % 1), p_child_worker1, 30, sys.stdout, 32)
    worker1.start()
    worker_list.append((("worker%d" % 1), p_parent_worker1, 30, 32))

    p_parent_worker2, p_child_worker2 = mp.Pipe()
    worker2 = WorkerProc(("worker%d" % 2), p_child_worker2, 30, sys.stdout, 32)
    worker2.start()
    worker_list.append((("worker%d" % 2), p_parent_worker2, 30, 32))

    p_parent_worker3, p_child_worker3 = mp.Pipe()
    worker3 = WorkerProc(("worker%d" % 3), p_child_worker3, 30, sys.stdout, 32)
    worker3.start()
    worker_list.append((("worker%d" % 3), p_parent_worker3, 30, 32))

    p_parent_worker4, p_child_worker4 = mp.Pipe()
    worker4 = WorkerProc(("worker%d" % 4), p_child_worker4, 10, sys.stdout, 32)
    worker4.start()
    worker_list.append((("worker%d" % 4), p_parent_worker4, 10, 32))

    for worker in worker_list:
        worker[1].send('BEGIN')

    worker1.join()
    worker2.join()
    worker3.join()
    worker4.join()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
