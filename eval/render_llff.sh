#!/usr/bin/env bash

cd eval/
FILE='render_llff_video'
CUDA_VISIBLE_DEVICES=0 python $FILE.py --config ../configs/render_llff_video.txt --train_scenes orchids --eval_scenes orchids &
CUDA_VISIBLE_DEVICES=1 python $FILE.py --config ../configs/render_llff_video.txt --train_scenes horns --eval_scenes horns &
CUDA_VISIBLE_DEVICES=2 python $FILE.py --config ../configs/render_llff_video.txt --train_scenes trex --eval_scenes trex &
CUDA_VISIBLE_DEVICES=3 python $FILE.py --config ../configs/render_llff_video.txt --train_scenes room --eval_scenes room &
CUDA_VISIBLE_DEVICES=4 python $FILE.py --config ../configs/render_llff_video.txt --train_scenes flower --eval_scenes flower &
CUDA_VISIBLE_DEVICES=5 python $FILE.py --config ../configs/render_llff_video.txt --train_scenes leaves --eval_scenes leaves &
CUDA_VISIBLE_DEVICES=6 python $FILE.py --config ../configs/render_llff_video.txt --train_scenes fern --eval_scenes fern &
CUDA_VISIBLE_DEVICES=7 python $FILE.py --config ../configs/render_llff_video.txt --train_scenes fortress --eval_scenes fortress

