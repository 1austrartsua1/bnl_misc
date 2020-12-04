

export WORLD_SIZE=1
export CUDA_VISIBLE_DEVICES=0
export MASTER_PORT=7440
export MASTER_ADDR=127.0.0.1
python cifar10_ddp.py
