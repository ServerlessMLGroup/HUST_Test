__global__ int pull_taskID(task_queue){
    int top_block_id = task_queue.front(); // 原子操作
    task_queue.pop(); // 原子操作
    return top_block_id;
}


__global__ void Elastic_Kernel((origin_block_num, origin_thread_num, mem, streamid), &kernel_task_queue) {
    smID = get_current_smID();
    kill_redundant_worker();
    while(!empty_task()){
        if(!is_current_sm_idle(smID)) {
            sleep(10us);
            continue;
        }
        block_id = pull_taskID(kernel_task_queue);
        ... run as block_id:
    }
}

__global__ void Launch_kernel()