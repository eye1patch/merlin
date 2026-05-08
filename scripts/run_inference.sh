

accelerate launch --multi_gpu --mixed_precision=bf16 inference.py --config configs/inference.yaml
# torchrun --nproc_per_node=1 inference.py --config configs/inference.yaml