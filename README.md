# RPIFormer

RPIFormer is a streamlined research codebase for low-light image enhancement. This repository corresponds to our manuscript **Progressive Illumination-aware Transformer for Retinex-based Low-Light Image Enhancement**, which is being prepared for resubmission to **The Visual Computer**.

The package keeps only the components required to reproduce the final model, the four dataset settings used in the manuscript, and the evaluation pipeline.

## Statement

- This code is directly related to the manuscript currently submitted to **The Visual Computer**.
- If you use this repository in academic work, please cite the corresponding RPIFormer manuscript when the bibliographic information becomes available.
- This release is a cleaned research package built on a BasicSR-style framework and derived from our earlier baseline-based experiments. The upstream relation is kept explicit for reproducibility and licensing clarity.

## Features

- Final **RPIFormer** architecture with illumination-guided self-attention, parallel feature-illumination cross-attention, and Euler-style residual gating.
- Lightweight code package for training, paired-dataset testing, and PSNR / SSIM / LPIPS evaluation.
- Clean training and testing configs for `LOLv1`, `LOLv2-real`, `LOLv2-synthetic`, and `SID`.

## Repository Layout

```text
RPIFormer/
|-- basicsr/
|-- options/
|   |-- train/
|   `-- test/
|-- tools/
|   |-- train.py
|   |-- test_dataset.py
|   `-- image_utils.py
|-- scripts/
|   `-- eval_three_metrics.py
|-- requirements.txt
`-- README.md
```

## Environment Setup

The codebase is recommended to run with:

- Python `3.10`
- PyTorch `2.5.1+cu121`
- TorchVision `0.20.1+cu121`
- CUDA runtime `12.1`

Create a new conda environment and install the dependencies:

```bash
# create a new conda environment
conda create -n rpiformer python=3.10 -y
conda activate rpiformer

# install python dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

Pinned package versions are listed in [requirements.txt](requirements.txt).

No extra compilation or package build step is required for this repository.

## Dataset Preparation

Create a `data/` directory in the repository root and place datasets with the following structure:

```text
data/
|-- LOLv1/
|   |-- Train/
|   |   |-- input/
|   |   `-- target/
|   `-- Test/
|       |-- input/
|       `-- target/
|-- LOLv2/
|   |-- Real_captured/
|   |   |-- Train/
|   |   |   |-- Low/
|   |   |   `-- Normal/
|   |   `-- Test/
|   |       |-- Low/
|   |       `-- Normal/
|   `-- Synthetic/
|       |-- Train/
|       |   |-- Low/
|       |   `-- Normal/
|       `-- Test/
|           |-- Low/
|           `-- Normal/
`-- SID/
    |-- short_sid2/
    `-- long_sid2/
```

Expected datasets:

- `LOLv1`
- `LOLv2/Real_captured`
- `LOLv2/Synthetic`
- `SID` RGB `.npy` subset with `short_sid2` and `long_sid2`

Dataset download:

- Baidu Netdisk: <https://pan.baidu.com/s/1Lmv9VQ-7Wu3LwXrqzL6v4Q?pwd=gi1s>
- Extraction code: `gi1s`

Notes:

- `SID` in this repository follows the RGB `.npy` subset used in the experiments.
- Do not commit dataset files to GitHub.

## Checkpoint Preparation

Create a `pretrained_weights/` directory in the repository root and place checkpoints with the following layout:

```text
pretrained_weights/
|-- euler/
|   |-- best_model_LOL_v2_real.pth
|   |-- best_model_LOL_v2_synthetic.pth
|   `-- best_model_SID.pth
`-- rpiformer/
    |-- best_LOL_v1_model.pth
    |-- best_LOL_v2_real_model.pth
    |-- best_LOL_v2_synthetic_model.pth
    `-- best_SID_model.pth
```

Recommended filenames for the released final model:

- `pretrained_weights/rpiformer/best_LOL_v1_model.pth`
- `pretrained_weights/rpiformer/best_LOL_v2_real_model.pth`
- `pretrained_weights/rpiformer/best_LOL_v2_synthetic_model.pth`
- `pretrained_weights/rpiformer/best_SID_model.pth`

For reproducing training with the provided configs, you also need:

- `pretrained_weights/euler/best_model_LOL_v2_real.pth`
- `pretrained_weights/euler/best_model_LOL_v2_synthetic.pth`
- `pretrained_weights/euler/best_model_SID.pth`

`LOLv1` training starts from scratch and does not require a warm-start checkpoint.

Checkpoint download:

- Baidu Netdisk: <https://pan.baidu.com/s/1phFUj8jvFC_MoEgamBayGA?pwd=pwd1>
- Extraction code: `pwd1`
- Archive name: `pretrained_weights.zip`

Notes:

- The `euler/` checkpoints are used only as initialization weights for part of the training configs.
- The `rpiformer/` checkpoints are the released final weights used for evaluation.
- Adjust paths in the YAML files if you keep a different folder structure.

## Recommended Local Verification Order

1. Copy the datasets into `data/`.
2. Copy the required checkpoints into `pretrained_weights/`.
3. Create the Python environment and run `pip install -r requirements.txt`.
4. Verify that testing works first by running the four release-evaluation commands below.
5. After testing is stable, launch training with one config to confirm the training pipeline starts normally.
6. If everything is correct, remove local datasets and checkpoints again and host them externally for release.

## Testing

Run evaluation from the repository root:

```bash
python tools/test_dataset.py --opt options/test/rpiformer_lolv1.yml --weights pretrained_weights/rpiformer/best_LOL_v1_model.pth --dataset LOL_v1 --gpus 0
python tools/test_dataset.py --opt options/test/rpiformer_lolv2_real.yml --weights pretrained_weights/rpiformer/best_LOL_v2_real_model.pth --dataset LOL_v2_real --gpus 0
python tools/test_dataset.py --opt options/test/rpiformer_lolv2_synthetic.yml --weights pretrained_weights/rpiformer/best_LOL_v2_synthetic_model.pth --dataset LOL_v2_synthetic --gpus 0
python tools/test_dataset.py --opt options/test/rpiformer_sid.yml --weights pretrained_weights/rpiformer/best_SID_model.pth --dataset SID --gpus 0
```

The test script reports:

- `PSNR`
- `SSIM`
- `LPIPS`

Restored images are written to:

```text
results/<dataset>/<config_name>/<checkpoint_name>/
```

For `SID`, the script also saves cached `input` and `gt` folders under `results/SID/`.

## Training

Run training from the repository root:

```bash
python tools/train.py --opt options/train/rpiformer_lolv1.yml
python tools/train.py --opt options/train/rpiformer_lolv2_real.yml
python tools/train.py --opt options/train/rpiformer_lolv2_synthetic.yml
python tools/train.py --opt options/train/rpiformer_sid.yml
```

Notes:

- `LOLv2-real`, `LOLv2-synthetic`, and `SID` use Euler-stage checkpoints as warm-start initialization.
- Training logs and checkpoints are written to `experiments/<config_name>/`.
- To test whether training can start normally on a local machine, it is enough to launch one config and confirm that dataloader creation, model construction, optimizer setup, and the first validation cycle all run without error.

## Metric Re-evaluation

You can recompute metrics on exported images with:

```bash
python scripts/eval_three_metrics.py --pred <result_dir> --gt <gt_dir>
```

## Reproducibility Notes

- The repo intentionally excludes datasets, logs, and checkpoints from Git tracking.
- Paths are relative to the repository root so that the package can be moved to GitHub directly.
- The testing configs are aligned with the released final model, including the valid `SID` setting based on `best_SID_model.pth`.
- Training and testing configs are split for clarity, but the `network_g` definitions are matched pairwise across all four datasets.

## Acknowledgements

This repository is built on the BasicSR ecosystem and our earlier experimental code. We keep those dependencies explicit for traceability and license compliance.
