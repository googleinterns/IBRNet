#!/usr/bin/env bash
cd eval/

CUDA_VISIBLE_DEVICES=0 python eval.py --config ../configs/eval_nerf_synthetic.txt --eval_scenes mic &
CUDA_VISIBLE_DEVICES=1 python eval.py --config ../configs/eval_nerf_synthetic.txt --eval_scenes chair &
CUDA_VISIBLE_DEVICES=2 python eval.py --config ../configs/eval_nerf_synthetic.txt --eval_scenes lego &
CUDA_VISIBLE_DEVICES=3 python eval.py --config ../configs/eval_nerf_synthetic.txt --eval_scenes ficus &
CUDA_VISIBLE_DEVICES=4 python eval.py --config ../configs/eval_nerf_synthetic.txt --eval_scenes materials &
CUDA_VISIBLE_DEVICES=5 python eval.py --config ../configs/eval_nerf_synthetic.txt --eval_scenes hotdog &
CUDA_VISIBLE_DEVICES=6 python eval.py --config ../configs/eval_nerf_synthetic.txt --eval_scenes ship &
CUDA_VISIBLE_DEVICES=7 python eval.py --config ../configs/eval_nerf_synthetic.txt --eval_scenes drums &

