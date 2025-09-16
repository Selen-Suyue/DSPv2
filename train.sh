PYTHONDONTWRITEBYTECODE=1 \
CUDA_VISIBLE_DEVICES=0 \
torchrun --master_addr 127.0.0.1 --master_port 14527 \
    --nproc_per_node 1 --nnodes 1 --node_rank 0 \
    train.py \
    --data_path data/processed_bowling \
    --num_action 16  \
    --ckpt_dir logs/dspv2/bowling \
    --batch_size 16 --num_epochs 300 --save_epochs 50 --num_workers 48 \
    --seed 233 --task bowling 