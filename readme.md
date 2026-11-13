
# MonoHair:High-Fidelity Hair Modeling from a Monocular Video  <font color='red'> ( CVPR 2024 Oral ) </font> [[Projectpage](https://keyuwu-cs.github.io/MonoHair/ "Projectpage")] #
This repository is the official code for MonoHair. Give a monocular video, MonoHair reconstruct a high-fidelity 3D strand model. 

In this repository we also give some examples for reconstructing 3D hair from a monocular video or multi-view images.

- We generate 3D avatar using flame template and fit the flame coarse geometry using multiview images (only for real human capture), more details please check [DELTA](https://github.com/yfeng95/DELTA "Delta") 
- For coarse goemtry initialization, please check [Instant-NGP](https://github.com/NVlabs/instant-ngp "Instant-NGP").
- For hair exterior geometry synthesis, we propose patch-based multi-view optimization (PMVO) method, please check our [Paper](https://arxiv.org/abs/2403.18356 "MonoHair").
- For hair interior inference, please check [DeepMVSHair](https://dl.acm.org/doi/abs/10.1145/3550469.3555385 "DeepMVSHair").
- For strands generate, please also check our paper.

![](fig/results.png)



## Getting Started ##
Clone the repository and install requirements:

    git clone https://github.com/KeyuWu-CS/MonoHair.git --recursive
	cd MonoHair
	conda create -n MonoHair python==3.10.12
	conda activate MonoHair
	pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
	pip install -r requirements.txt
	

## Dependencies and submodules ##

Install [Pytorch](https://pytorch.org/ "torch"), [Pytorch3d](https://github.com/facebookresearch/pytorch3d) and [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn). We have tested on Ubuntu 22.04.4, gcc==9.5, python==3.10.12 pytorch==1.11.0, pytorch3d==0.7.2 with CUDA 11.3 on RTX 3090Ti. You can install any version that is compatible with these dependencies. We know torch==1.3.0 have some bug when employing [MODNet](https://github.com/ZHKKKe/MODNet "MODNet").

	# if have problem when install pytorch 3d, you can try to install fvcore: pip install  fvcore==0.1.5.post20220512
	pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu113_pyt1110/download.html
	

Initialize submodules of [Instant-NGP](https://github.com/NVlabs/instant-ngp "Instant-NGP"), [MODNet](https://github.com/ZHKKKe/MODNet "MODNet"), [CDGNet](https://github.com/tjpulkl/CDGNet "CDGNet"), [DELTA](https://github.com/yfeng95/DELTA "DELTA") and [face-parsing](https://github.com/zllrunning/face-parsing.PyTorch "face-parsing"). 

    git submodule update --init --recursive
	pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

If you have problem when install tiny-cuda-nn or pytorch3d, please refer to their repositories.

Compile Instant-NGP and move our modified [run.py](https://github.com/KeyuWu-CS/MonoHair/blob/master/run.py) to instant-ngp/scripts.

	cp run.py submodules/instant-ngp/scripts/
	cd submodules/instant-ngp
	cmake . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
	cmake --build build --config RelWithDebInfo -j
	cd ../..

	

if you have problem with Instant-NGP compile. Please refer their [instruction](https://github.com/NVlabs/instant-ngp)

## Download assets ##
Download pretrained model for [MODNet](https://github.com/ZHKKKe/MODNet "MODNet"), [CDGNet](https://github.com/tjpulkl/CDGNet "CDGNet") and [face-parsing](https://github.com/zllrunning/face-parsing.PyTorch "face-parsing").

	# Download some pretrained model and data for avatar optimization.  
	# You should download pretrained model of CDGNet in thier repository, their are 
	# two kind of "LIP_epoch_149.pth", please download the one with a size of 300MB.
	bash fetch_pretrained_model.sh
    bash fetch_data.sh    #this will cost long time.

## Download examples ##
Download our example datas [One Driven](https://1drv.ms/f/s!AhfQmEHzY54Ya2gGaslXnM2IPCk?e=phk5me "One Driven"). For obtaining a certain results, we have run colmap and save the pretrained instant-ngp weight. Then you need to run the follow four steps to get the results. We also provide full results (include intermediate results) in full folder. You can use it to check the results of each step.  
**Tips:** Since the wig use the same unreal human head, we don't use flame (smplx) model as template and don't run multi-view bust fitting. 

## 3D Hair Reconstruction ##

	# Prepare data: instant-ngp intialization, segmentation, gabor filter etc. You skip this step if use our provided data.
    python prepare_data.py --yaml=configs/reconstruct/big_wavy1 

	# Hair exterior optimization
	python PMVO.py --yaml=configs/reconstruct/big_wavy1

	# Hair interior inference
	python infer_inner.py --yaml=configs/reconstruct/big_wavy1
	
	# Strands generation
	python HairGrow.py --yaml=configs/reconstruct/big_wavy1

## Calibrated synthetic cases ##

For a case containing `cropped/images/*.jpg`, `sparse/{cameras,images,points3D}.bin`,
and `head_flame.obj` in FLAME topology, run:

```bash
python run_synthetic.py input/synth/ct2hair/bangs
```

Inputs must use millimeters, +Y up, +Z front, and +X left. The adapter reads
PINHOLE/SIMPLE_PINHOLE COLMAP calibration without estimating or optimizing cameras,
and uses the supplied head without fitting or deforming it. Images are processed
at half size by default (1200 pixels high × 800 wide for these cases), with each camera's focal lengths
and principal point scaled to match. Use `--image-scale 1` for native resolution.
All registered images are used; additional GT masks and reconstructions are ignored.

The output for this example is `output/synth/ct2hair/bangs/`:

- `head_flame.obj` and `sparse/*.bin`: byte-identical copies of the inputs.
- `connected_strands.hair`: the existing MonoHair binary strand format, in the
  **original millimeter coordinates**. Read it with `Utils.Utils.load_strand`.
- `coordinate_transform.json`: input checksums, image names, and the forward/inverse
  transforms. Internal coordinates are meters, translated so corresponding FLAME
  scalp centroids match the existing canonical frame; world axes are not rotated.
- `work/`: working images, camera conversions, generated `case.yaml`, NGP snapshot,
  meshes, masks, orientation/depth maps, stage logs, and reconstruction results.
  Geometry in this directory uses the internal frame, including
  `work/results/synthetic/full/connected_strands.hair`. Use the case-root export for GT evaluation.
  The raw `work/ngp/base.obj` uses NGP's `(y,z,x)` axis order; its converted copy
  `work/ours/colmap_points.obj` uses the internal `(x,y,z)` frame.

The original capture configurations and their defaults remain supported. Synthetic
interior inference still uses the existing 16-view virtual rig and pretrained models.
No manual key frame, COLMAP rerun, SMPL-X fitting, or pre-trained per-case NGP snapshot
is needed. NGP uses the inferred MODNet silhouette as image alpha to exclude the
background. Its defaults are 35,000 training steps and a 512³ extraction grid,
restricted to MonoHair's existing 0.64 × 0.64 × 0.48 meter reconstruction volume.

The standard MODNet, CDGNet, and DeepMVSHair checkpoints, CUDA, compiled `pyngp`, and
headless EGL rendering are required. Missing dependencies are reported before a full
run starts. Existing checkpoint locations can be supplied without copying them:

```bash
python run_synthetic.py input/synth/ct2hair/bangs \
  --modnet-checkpoint /path/to/modnet_photographic_portrait_matting.ckpt \
  --cdgnet-checkpoint /path/to/LIP_epoch_149.pth
```

Use the roughly 300 MB **CDGNet** checkpoint linked in
[its repository](https://github.com/tjpulkl/CDGNet); the similarly named CE2P model is incompatible.

Input checks and preparation do not need model weights or CUDA:

```bash
for case_dir in input/synth/ct2hair/*; do
  python run_synthetic.py "$case_dir" --check-only || break
done
python run_synthetic.py input/synth/ct2hair/bangs --stage prepare
```

For an interrupted run, individual stages are available in order:
`prepare`, `images`, `ngp`, `depth`, `pmvo`, `inner`, `grow`, `export`.
For example, `--stage inner` uses the generated case configuration and also runs
the subsequent PMVO interior fusion. Stages use the prepared image scale; rerun
preparation and downstream stages when changing it. A normal run executes every
stage afresh. `--input-root` and `--output-root` change the two roots while retaining
the relative case path.

Geometry and preservation regression tests (including the ten cases when present):

```bash
mkdir -p output/tests
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/test_synthetic.py --basetemp=output/tests/synthetic
```

## Visualization ##
Download our released program at  [One Driven](https://1drv.ms/f/s!AhfQmEHzY54Ya2gGaslXnM2IPCk?e=phk5me "One Driven") to visualize the results. 

    # First copy the output/10-16/full/connected_strands.hair to ../Ours/Voxel_hair
	cp data/case_name/output/10-16/full/connected_strands.hair data/case_name/ours/Voxel_hair

	# unzip VoxelHair_demo_v3.zip and click VoxelHair_v1.exe. 
	# Then click "Load Strands" to visualize the results. You also can use Blender to achieve realistic rendering.


## Test your own data ##
In our given examples, we ignored the steps of running colmap and training instant ngp. So if you want to test your own captured videos. Please refer to the [following steps](https://github.com/KeyuWu-CS/MonoHair/tree/master/video_preprocess).



## Citation ##

    @inproceedings{wu2024monohair,
	  title={MonoHair: High-Fidelity Hair Modeling from a Monocular Video},
	  author={Wu, Keyu and Yang, Lingchen and Kuang, Zhiyi and Feng, Yao and Han, Xutao and Shen, Yuefan and Fu, Hongbo and Zhou, Kun and Zheng, Youyi},
	  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
	  pages={24164--24173},
	  year={2024}
	}

## Acknowledgments ##
Here are some great resources we benefit from:

- [MODNet](https://github.com/ZHKKKe/MODNet "MODNet") and [CDGNet](https://github.com/tjpulkl/CDGNet "CDGNet") to segment hair.
- [Instant-NGP](https://github.com/NVlabs/instant-ngp "Instant-NGP") for coarse initizalization.
- [DELTA](https://github.com/yfeng95/DELTA "DELTA") for bust fitting.
  

## TO DO List ##
- [x] Upload full example data (before June.24)
- [x] Check version problem (before June.24)
- [x] Release visualization program (before June.30)
- [ ] Automatic method to add key_frame.json

