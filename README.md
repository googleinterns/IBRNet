# IBRNet: Learning Multi-View Image-Based Rendering
PyTorch implementation of paper "IBRNet: Learning Multi-View Image-Based Rendering", CVPR 2021.

> IBRNet: Learning Multi-View Image-Based Rendering  
> [Qianqian Wang](https://www.cs.cornell.edu/~qqw/), [Zhicheng Wang](https://www.linkedin.com/in/zhicheng-wang-96116897/), [Kyle Genova](https://www.kylegenova.com/), [Pratul Srinivasan](https://pratulsrinivasan.github.io/), [Howard Zhou](https://www.linkedin.com/in/howard-zhou-0a34b84/), [Jonathan T. Barron](https://jonbarron.info), [Ricardo Martin-Brualla](http://www.ricardomartinbrualla.com/), [Noah Snavely](https://www.cs.cornell.edu/~snavely/), [Thomas Funkhouser](https://www.cs.princeton.edu/~funk/)    
> CVPR 2021
> 

#### [project page](https://ibrnet.github.io/) | [paper](http://arxiv.org/abs/2102.13090) | [data & model](https://drive.google.com/drive/folders/1qfcPffMy8-rmZjbapLAtdrKwg3AV-NJe?usp=sharing)

![Demo](assets/ancient.gif)

## Installation
Clone this repo with submodules:
```
git clone --recurse-submodules https://github.com/googleinterns/IBRNet
cd IBRNet/
```

The code is tested with Python3.7, PyTorch == 1.5 and CUDA == 10.2. We recommend you to use [anaconda](https://www.anaconda.com/) to make sure that all dependencies are in place. To create an anaconda environment:
```
conda env create -f environment.yml
conda activate ibrnet
```

## Datasets

### 1. Training datasets
```
├──data/
    ├──ibrnet_collected_1/
    ├──ibrnet_collected_2/
    ├──real_iconic_noface/
    ├──spaces_dataset/
    ├──RealEstate10K-subset/
    ├──google_scanned_objects/

```
Please first `cd data/`, and then download datasets into `data/` following the instructions below. The organization of the datasets should be the same as above.

#### (a) **Our captures**
We captured 67 forward-facing scenes (each scene contains 20-60 images). To download our data [ibrnet_collected.zip](https://drive.google.com/file/d/1rkzl3ecL3H0Xxf5WTyc2Swv30RIyr1R_/view?usp=sharing) (4.1G) for training, run:
```
gdown https://drive.google.com/uc?id=1rkzl3ecL3H0Xxf5WTyc2Swv30RIyr1R_
unzip ibrnet_collected.zip
```

P.S. We've captured some more scenes in [ibrnet_collected_more.zip](https://drive.google.com/file/d/1Uxw0neyiIn3Ve8mpRsO6A06KfbqNrWuq/view?usp=sharing), but we didn't include them for training. Feel free to download them if you would like more scenes for your task, but you wouldn't need them to reproduce our results.
#### (b) [**LLFF**](https://bmild.github.io/llff/) released scenes
Download and process [real_iconic_noface.zip](https://drive.google.com/drive/folders/1M-_Fdn4ajDa0CS8-iqejv0fQQeuonpKF) (6.6G) using the following commands:
```angular2
# download 
gdown https://drive.google.com/uc?id=1ThgjloNt58ZdnEuiCeRf9tATJ-HI0b01
unzip real_iconic_noface.zip

# [IMPORTANT] remove scenes that appear in the test set
cd real_iconic_noface/
rm -rf data2_fernvlsb data2_hugetrike data2_trexsanta data3_orchid data5_leafscene data5_lotr data5_redflower
cd ../
``` 
#### (c) [**Spaces Dataset**](https://github.com/augmentedperception/spaces_dataset)
Download spaces dataset by:
```
git clone https://github.com/augmentedperception/spaces_dataset
```


#### (d) [**RealEstate10K**](https://google.github.io/realestate10k/)
The full RealEstate10K dataset is very large and can be difficult to download.
Hence, we provide a subset of RealEstate10K training scenes containing only 200 scenes. In our experiment, we found using more scenes from RealEstate10K only provides marginal improvement. To download our [camera files](https://drive.google.com/file/d/1IgJIeCPPZ8UZ529rN8dw9ihNi1E9K0hL/view?usp=sharing) (2MB):

```
gdown https://drive.google.com/uc?id=1IgJIeCPPZ8UZ529rN8dw9ihNi1E9K0hL
unzip RealEstate10K_train_cameras_200.zip -d RealEstate10K-subset
```
Besides the camera files, you also need to download the corresponding video frames from YouTube. You can download the frames (29G) by running the following commands. The script uses `ffmpeg` to extract frames, so please make sure you have [ffmpeg](https://ffmpeg.org/) installed.

```
git clone https://github.com/qianqianwang68/RealEstate10K_Downloader
cd RealEstate10K_Downloader
python generate_dataset.py train
cd ../
```

#### (e) [**Google Scanned Objects**](https://app.ignitionrobotics.org/GoogleResearch/fuel/collections/Google%20Scanned%20Objects)
Google Scanned Objects contain 1032 diffuse objects with various shapes and appearances.
We use [gaps](https://github.com/tomfunkhouser/gaps) to render these objects for training. Each object is rendered at 512 × 512 pixels
from viewpoints on a quarter of the sphere. We render 250
views for each object. To download [our renderings](https://drive.google.com/file/d/1w1Cs0yztH6kE3JIz7mdggvPGCwIKkVi2/view?usp=sharing) (7.5GB), run:
```
gdown https://drive.google.com/uc?id=1w1Cs0yztH6kE3JIz7mdggvPGCwIKkVi2
unzip google_scanned_objects_renderings.zip
```
The mapping between our renderings and the public Google Scanned Objects can be found in [this spreadsheet](https://docs.google.com/spreadsheets/d/14FivSzpjtqraR8IFmKOWWFXRUh4JsmTJqF2hr_ZY2R4/edit?usp=sharing&resourcekey=0-vVIKfNOVddY20NhBWr2ipQ).

### 2. Evaluation datasets
```
├──data/
    ├──deepvoxels/
    ├──nerf_synthetic/
    ├──nerf_llff_data/
```
The evaluation datasets include DeepVoxel synthetic dataset, NeRF realistic 360 dataset and the real forward-facing dataset. To download all three datasets (6.7G), run the following command under `data/` directory:
```
bash download_eval_data.sh
```

## Evaluation
First download our pretrained model under the project root directory:
```
gdown https://drive.google.com/uc?id=165Et85R8YnL-5NcehG0fzqsnAUN8uxUJ
unzip pretrained_model.zip
```

You can use `eval/eval.py` to evaluate the pretrained model. For example, to obtain the PSNR, SSIM and LPIPS on the *fern* scene in the real forward-facing dataset, you can first specify your paths in `configs/eval_llff.txt` and then run:
```
cd eval/
python eval.py --config ../configs/eval_llff.txt
``` 
## Rendering videos of smooth camera paths
You can use `render_llff_video.py` to render videos of smooth camera paths for the real forward-facing scenes. For example, you can first specify your paths in `configs/eval_llff.txt` and then run:
```
cd eval/
python render_llff_video.py --config ../configs/eval_llff.txt
```
You can also capture your own data of forward-facing scenes and synthesize novel views using our method. Please follow the instructions from [LLFF](https://github.com/Fyusion/LLFF) on how to capture and process the images. 


## Training
We strongly recommend you to train the model with multiple GPUs:
```
# this example uses 8 GPUs (nproc_per_node=8) 
python -m torch.distributed.launch --nproc_per_node=8 train.py --config configs/pretrain.txt
```
Alternatively, you can train with a single GPU by setting `distributed=False` in `configs/pretrain.txt` and running:
```
python train.py --config configs/pretrain.txt
```

## Finetuning
To finetune on a specific scene, for example, *fern*, using the pretrained model, run:
```
# this example uses 2 GPUs (nproc_per_node=2) 
python -m torch.distributed.launch --nproc_per_node=2 train.py --config configs/finetune_llff.txt
```

## Additional information
- Our current implementation is not well-optimized in terms of the time efficiency at inference. Rendering a 1000x800 image can take from 30s to over a minute depending on specific GPU models. Please make sure to maximize the GPU memory utilization by increasing the size of the chunk to reduce inference time. You can also try to decrease the number of input source views (but subject to performance loss).  
- If you want to create and train on your own datasets, you can implement your own Dataset class following our examples in `ibrnet/data_loaders/`. You can verify the camera poses using `data_verifier.py` in `ibrnet/data_loaders/`.
- Since the evaluation datasets are either object-centric or forward-facing scenes, our provided view selection methods are very simple (based on either viewpoints or camera locations). If you want to evaluate our method on new scenes with other kinds of camera distributions, you might need to implement your own view selection methods to identify the most effective source views.
- If you have any questions, you can contact qw246@cornell.edu.
## Citation
```
@inproceedings{wang2021ibrnet,
  author    = {Wang, Qianqian and Wang, Zhicheng and Genova, Kyle and Srinivasan, Pratul and Zhou, Howard  and Barron, Jonathan T. and Martin-Brualla, Ricardo and Snavely, Noah and Funkhouser, Thomas},
  title     = {IBRNet: Learning Multi-View Image-Based Rendering},
  booktitle = {CVPR},
  year      = {2021}
}

```
