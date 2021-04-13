#!/usr/bin/env bash

SFX='1'
FILE='run_nerf'
CUDA_VISIBLE_DEVICES=0 python $FILE.py --config configs/finetune_llff.txt --train_scenes orchids --eval_scenes orchids --expname finetune_orchids_$SFX &
CUDA_VISIBLE_DEVICES=1 python $FILE.py --config configs/finetune_llff.txt --train_scenes horns --eval_scenes horns --expname finetune_horns_$SFX &
CUDA_VISIBLE_DEVICES=2 python $FILE.py --config configs/finetune_llff.txt --train_scenes trex --eval_scenes trex --expname finetune_trex_$SFX &
CUDA_VISIBLE_DEVICES=3 python $FILE.py --config configs/finetune_llff.txt --train_scenes room --eval_scenes room --expname finetune_room_$SFX &
CUDA_VISIBLE_DEVICES=4 python $FILE.py --config configs/finetune_llff.txt --train_scenes flower --eval_scenes flower --expname finetune_flower_$SFX &
CUDA_VISIBLE_DEVICES=5 python $FILE.py --config configs/finetune_llff.txt --train_scenes leaves --eval_scenes leaves --expname finetune_leaves_$SFX &
CUDA_VISIBLE_DEVICES=6 python $FILE.py --config configs/finetune_llff.txt --train_scenes fern --eval_scenes fern --expname finetune_fern_$SFX &
CUDA_VISIBLE_DEVICES=7 python $FILE.py --config configs/finetune_llff.txt --train_scenes fortress --eval_scenes fortress --expname finetune_fortress_$SFX

