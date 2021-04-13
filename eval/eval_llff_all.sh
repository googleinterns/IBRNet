#!/usr/bin/env bash
cd eval/
CUDA_VISIBLE_DEVICES=0 python eval.py --config ../configs/eval_llff.txt --eval_scenes horns &
CUDA_VISIBLE_DEVICES=1 python eval.py --config ../configs/eval_llff.txt --eval_scenes trex &
CUDA_VISIBLE_DEVICES=2 python eval.py --config ../configs/eval_llff.txt --eval_scenes room &
CUDA_VISIBLE_DEVICES=3 python eval.py --config ../configs/eval_llff.txt --eval_scenes flower &
CUDA_VISIBLE_DEVICES=4 python eval.py --config ../configs/eval_llff.txt --eval_scenes orchids &
CUDA_VISIBLE_DEVICES=5 python eval.py --config ../configs/eval_llff.txt --eval_scenes leaves &
CUDA_VISIBLE_DEVICES=6 python eval.py --config ../configs/eval_llff.txt --eval_scenes fern &
CUDA_VISIBLE_DEVICES=7 python eval.py --config ../configs/eval_llff.txt --eval_scenes fortress &

