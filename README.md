<p align="center">
  <img src="figures/teaser.png" alt="OVT-B Teaser" width="100%">
</p>

<h1 align="center">OVT-B: A New Large-Scale Benchmark for Open-Vocabulary Multi-Object Tracking</h1>

<p align="center">
  <b>NeurIPS 2024 — Datasets and Benchmarks Track</b>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2410.17534">Haiji Liang</a> &nbsp;&middot;&nbsp;
  <a href="https://arxiv.org/abs/2410.17534">Ruize Han</a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2410.17534"><img src="https://img.shields.io/badge/arXiv-2410.17534-b31b1b.svg" alt="arXiv"></a>
  <a href="https://openreview.net/forum?id=5S0y3OhfRs"><img src="https://img.shields.io/badge/OpenReview-NeurIPS%202024-blue.svg" alt="OpenReview"></a>
  <a href="https://paperswithcode.com/dataset/ovt-b"><img src="https://img.shields.io/badge/Papers%20With%20Code-OVT--B-21cbce.svg" alt="PwC"></a>
  <a href="#license"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License"></a>
</p>

---

## What is OVT-B?

**Open-Vocabulary Multi-Object Tracking (OVMOT)** aims to track objects of arbitrary categories — including novel classes never seen during training. While open-vocabulary detection in single images has been widely studied, open-vocabulary *tracking* in video remains underexplored, largely due to the lack of suitable benchmarks.

**OVT-B** fills this gap. It is the largest open-vocabulary multi-object tracking benchmark to date, designed to push the boundaries of OVMOT research.

### OVT-B vs. OV-TAO-val

| | **OVT-B (Ours)** | OV-TAO-val |
|---|---|---|
| Categories | **1,048** | ~200 |
| Videos | **1,973** | ~900 |
| Bounding Box Annotations | **637,608** | — |
| Video Sources | 7 datasets | 1 dataset |

OVT-B is **5x larger** in category coverage and **2x larger** in video count compared to the only prior open-vocabulary tracking dataset.

## Dataset Statistics

| Statistic | Value |
|---|---|
| Total categories | 1,048 |
| Total videos | 1,973 |
| Total bounding box annotations | 637,608 |
| Source datasets | AnimalTrack, GMOT-40, ImageNet-VID, LVVIS, OVIS, UVO, YouTube-VIS-2021 |
| Evaluation metric | TETA (Tracking Every Thing Accuracy) |

## Data Samples

<p align="center">
  <img src="assets/ovtb_frame.png" alt="Sample 1" width="49%">
  <img src="assets/ovtb_frame2.png" alt="Sample 2" width="49%">
</p>
<p align="center">
  <img src="assets/ovtb_frame3.png" alt="Sample 3" width="49%">
  <img src="assets/ovtb_frame4.png" alt="Sample 4" width="49%">
</p>

## Quick Start

### 1. Download the Dataset

Download OVT-B from one of the following links:

- [Google Drive](https://drive.google.com/drive/folders/1Qfmb6tEF92I2k84NgrkjEbOKnFlsrTVZ?usp=drive_link)
- [BaiduYun](https://pan.baidu.com/s/1hy44z_om609jIhXjRxXCug?pwd=8yy3) (code: `8yy3`)

Extract the files and organize them under `data/ovtb/` (see [Directory Structure](#directory-structure) below).

### 2. Install

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv venv --python 3.7 && source .venv/bin/activate

# PyTorch (adjust cu113 to match your CUDA version)
uv pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 \
    --index-url https://download.pytorch.org/whl/cu113

# OpenMMLab
uv pip install mmcv-full==1.4.4 \
    --find-links https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
uv pip install mmdet==2.23.0

# CLIP + project + TETA evaluation metric
uv pip install git+https://github.com/openai/CLIP.git
uv pip install -e ".[eval]"
```

### 3. Prepare the Data

```
data/ovtb/
├── OVT-B/
│   ├── AnimalTrack/
│   ├── GMOT-40/
│   ├── ImageNet-VID/
│   ├── LVVIS/
│   ├── OVIS/
│   ├── UVO/
│   └── YouTube-VIS-2021/
├── ovtb_ann.json
├── ovtb_class.pth
├── ovtb_classname.py
├── ovtb_prompt.pth
└── ovtb_classes.txt
```

Copy the `CLASS`, `base_id`, and `novel_id` from `ovtb_classname.py` and add them to the `class_name.py` file under the `roi_head` folder of your open-vocabulary detector.

### 4. Download Pretrained Models

- [OVTrack model](https://drive.google.com/file/d/1vDAFRmuNMCwhKtW7KHONpzkooLysU8nX/view?usp=sharing) -> `saved_models/ovtrack_detpro_prompt.pth`
- [DetPro prompt](https://drive.google.com/file/d/1N7ht5b44R2LgExhk0smWLydpTO-RuwMe/view?usp=sharing) -> `saved_models/pretrained_models/detpro_prompt.pt`

### 5. Run Evaluation on OVT-B

```bash
# Multi-GPU evaluation (8 GPUs)
tools/dist_test.sh configs/ovtrack-teta/ovtb/ovtrack_r50.py \
    saved_models/ovtrack_detpro_prompt.pth 8 25000 \
    --eval track \
    --eval-options resfile_path=results/ovtb_results/

# Single-GPU evaluation
python tools/test.py configs/ovtrack-teta/ovtb/ovtrack_r50.py \
    saved_models/ovtrack_detpro_prompt.pth \
    --eval track \
    --eval-options resfile_path=results/ovtb_results/
```

The evaluation uses the TETA metric and reports **LocS** (Localization Score), **AssocS** (Association Score), **ClsS** (Classification Score), and overall **TETA** for both base and novel categories.

### Other Tracker Configs on OVT-B

We also provide configs for several tracker variants under `configs/ovtrack-teta/ovtb/`:

| Tracker | Config |
|---|---|
| OVTrack | `ovtrack_r50.py` |
| OVTrack+ | `ovtrack_plus.py` |
| ByteTrack | `ovtrack_bytetrack.py` |
| OC-SORT | `ovtrack_ocsort.py` |
| StrongSORT | `ovtrack_strongsort.py` |

## Directory Structure

```
OVT-B-Dataset/
├── assets/                     # Sample visualization images
├── configs/
│   ├── _base_/                 # Base model configs
│   ├── ovtrack-custom/         # Custom video inference config
│   ├── ovtrack-tao/            # OV-TAO-val evaluation configs
│   └── ovtrack-teta/
│       ├── ov_tao_val/         # OV-TAO-val TETA configs
│       └── ovtb/               # OVT-B evaluation configs
├── diffusers_clean/            # Diffusion-based data generation module
├── docs/                       # Additional documentation
│   ├── GET_STARTED.md          # Full setup guide (TAO + OVTrack training)
│   ├── INSTALL.md              # Installation details
│   └── OVTRACK_GEN.md          # Diffusion image generation guide
├── figures/                    # Paper figures
├── ovtrack/                    # Core library
│   ├── apis/                   # Train / test / inference entry points
│   ├── core/                   # Evaluation (TETA, MOT metrics), tracking utils
│   ├── datasets/               # OVTBDataset, TAO, COCO-Video, BDD loaders
│   └── models/                 # OVTrack model, trackers, ROI heads, losses
├── tools/                      # Scripts: train, test, inference, data conversion
├── requirements.txt
├── setup.py
└── LICENSE
```

## News

- **[2024-09-26]** Paper accepted to **NeurIPS 2024** Datasets and Benchmarks Track.
- **[2024-09-02]** Source code released for multiple tracker baselines on OVT-B.

## Citation

If you use OVT-B in your research, please cite:

```bibtex
@inproceedings{liang2024ovtb,
  title     = {{OVT-B}: A New Large-Scale Benchmark for Open-Vocabulary Multi-Object Tracking},
  author    = {Liang, Haiji and Han, Ruize},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track},
  year      = {2024}
}
```

## License

This project is released under the [Apache 2.0 License](LICENSE).

## Acknowledgments

- [TETA](https://github.com/SysCV/tet) — Evaluation metric implementation
- [OVTrack](https://github.com/SysCV/OVTrack) — Baseline method for open-vocabulary MOT
- [DetPro](https://github.com/dyabel/detpro) — PyTorch re-implementation of ViLD
- [MMTracking](https://github.com/open-mmlab/mmtracking) — OC-SORT, ByteTrack, and StrongSORT implementations
- [MMDetection](https://github.com/open-mmlab/mmdetection) — Detection framework backbone
