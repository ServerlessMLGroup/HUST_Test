typedef struct TASK {
    block_id,
    ...
}TASK;

__device__ int pull_taskID(&task_pool, smID){
    TASK task_info = task_pool.GetTASK(); // concurrent
    return task_info;
}

__device__ void Elastic_Kernel((streamid, sm_num, original_block_num), &kernel_task_pool) {
    smID = get_current_smID();
    kill_redundant_worker(); // 适应sm
    while(!empty_task(kernel_task_pool)){
        if(!is_current_sm_idle(smID)) {
            nanosleep();
            continue;
        }
        task_info = pull_taskID(kernel_task_pool, smID);
        ... run as task_info.block_id:
    }
}

__global__ void Launch_Kernel(args..., flags...) {
    init_flags(flags...);
    kernel_task_pool = init_task_pool();
    Elastic_Kernel((...), &kernel_task_pool);
}
