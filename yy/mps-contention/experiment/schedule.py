import sys
from queue import Queue
import torch
import torch.multiprocessing as mp

from pipeswitch.frontend_tcp import FrontendTcpThd
from pipeswitch.frontend_schedule import FrontendScheduleThd
from pipeswitch.worker import WorkerProc
from util.util import timestamp, TcpAgent, TcpServer

def main():
    print('main process begin')
    parent1, child1 = mp.Pipe()
    parent2, child2 = mp.Pipe()
    parent3, child3 = mp.Pipe()


    # Create workers
    num_workers = 2
    worker_list = []
    for _ in range(num_workers):
        p_parent, p_child = mp.Pipe()
        param_trans_parent, param_trans_child = mp.Pipe()
        term_parent, term_child = mp.Pipe()
        worker = WorkerProc(model_list, p_child, param_trans_child, term_child)
        worker.start()
        torch.cuda.send_shared_cache()
        worker_list.append((p_parent, worker, param_trans_parent, term_parent))
        timestamp('frontend', 'create_worker')

    # Create request queue and scheduler thread
    requests_queue = Queue()
    t_sch = FrontendScheduleThd(model_list, requests_queue, worker_list)
    t_sch.start()
    timestamp('frontend', 'start_schedule')

    # Accept connections
    server = TcpServer('localhost', 12345)
    timestamp('tcp', 'listen')
    while True:
        conn, _ = server.accept()
        agent = TcpAgent(conn)
        timestamp('tcp', 'connected')
        t_tcp = FrontendTcpThd(requests_queue, agent)
        t_tcp.start()

    # Wait for end
    t_sch.join()

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
