#!/usr/bin/env bash
cd eval/
CUDA_VISIBLE_DEVICES=0 python eval.py --config ../configs/eval_deepvoxels.txt --eval_scenes cube &
CUDA_VISIBLE_DEVICES=1 python eval.py --config ../configs/eval_deepvoxels.txt --eval_scenes vase &
CUDA_VISIBLE_DEVICES=2 python eval.py --config ../configs/eval_deepvoxels.txt --eval_scenes greek &
CUDA_VISIBLE_DEVICES=3 python eval.py --config ../configs/eval_deepvoxels.txt --eval_scenes armchair &