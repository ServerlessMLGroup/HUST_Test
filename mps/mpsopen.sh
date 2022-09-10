## variable 1 is the gpu we choose
## variable 2 is the percentage of mps we choose, 1 to 100
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log

export CUDA_VISIBLE_DEVICES=$1

sudo nvidia-smi -i $1 -c EXCLUSIVE_PROCESS
nvidia-cuda-mps-control -d
echo set_default_active_thread_percentage $2 | nvidia-cuda-mps-control

#show whther mps is opened
ps -ef | grep mps
