# DiffSynth-Train

Training pipelines for handwriting synthesis and writer identification/retrieval, built around **DiffusionPen** (diffusion-based handwritten text generation) and a separate **writer identification/retrieval** model.

---

## Table of Contents

- [1. DiffusionPen](#1-diffusionpen)
  - [1.1 Style Encoder Training](#11-style-encoder-training)
  - [1.2 DiffusionPen Training / Finetuning](#12-diffusionpen-training--finetuning)
  - [1.3 Sampling](#13-sampling)
  - [1.4 Pretrained Checkpoints](#14-pretrained-checkpoints)
  - [1.5 Dataset Format](#15-dataset-format)
- [2. Writer Identification / Retrieval](#2-writer-identification--retrieval)
  - [2.1 Training](#21-training)
  - [2.2 Dataset Format](#22-dataset-format)
- [Notes](#notes)

---

## 1. DiffusionPen

### 1.1 Style Encoder Training

The style encoder must be trained **independently**, before training the diffusion model itself:

```bash
python style_encoder_train.py \
    --pretrained False \
    --save_path './style_models/' \
    --epochs 100 \
    --dataset_fold 'dataset path' \
    --batch_size 320
```

### 1.2 DiffusionPen Training / Finetuning

Once the style encoder is trained, use it to train (or finetune) the diffusion model:

```bash
python train.py \
    --train_mode train \
    --dataset_fold '/path/to/dataset' \
    --save_path '/path/to/save/diffusionpen_model' \
    --style_path '/path/to/style_models/mixed_iam_mobilenetv2_100.pth' \
    --finetuning 1 \
    --finetune_path '/path/to/diffusionpen_iam_model_path/models/ckpt.pt' \
    --epochs 500
```

**Arguments:**

| Argument | Description |
|---|---|
| `--train_mode` | Set to `train` for training/finetuning |
| `--dataset_fold` | Path to the training dataset (see [format](#15-dataset-format)) |
| `--save_path` | Directory where checkpoints will be saved |
| `--style_path` | Path to the pretrained style encoder checkpoint (from step 1.1) |
| `--finetuning` | Set to `1` to finetune from an existing checkpoint, `0` to train from scratch |
| `--finetune_path` | Path to the base checkpoint to finetune from (see [pretrained checkpoints](#14-pretrained-checkpoints)) |
| `--epochs` | Number of training epochs |

### 1.3 Sampling

To generate handwriting samples from a trained model:

```bash
python train.py \
    --train_mode sampling \
    --sampling_mode single_sampling \
    --style_path '/path/to/style_models/mixed_iam_mobilenetv2_100.pth' \
    --save_path '/path/to/save/diffusionpen_model' \
    --dataset_fold '/path/to/dataset' \
    --output_dir '/path/to/output/samples' \
    --n_images 870 \
    --load_check 1
```

> Add `--n_images N` to generate a fixed number of images per writer instead of the default.

### 1.4 Pretrained Checkpoints

This project builds on the following released checkpoints:

- **DiffusionPen base checkpoint:** [konnik/DiffusionPen](https://huggingface.co/konnik/DiffusionPen) on Hugging Face — used as the starting point for finetuning (`--finetune_path`).
- **VAE encoder-decoder & DDIM:** [stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5).

### 1.5 Dataset Format

DiffusionPen expects the dataset to be organized by writer ID:

```
dataset_fold/
├── train/
│   ├── writer_id_1/
│   │   ├── img_001.png
│   │   ├── img_002.png
│   │   └── ...
│   ├── writer_id_2/
│   │   └── ...
│   └── ...
└── test/
    ├── writer_id_1/
    │   └── ...
    └── ...
```

If your data isn't already in this structure, feel free to write your own dataloader — the training script only expects `train/` and `test/` splits organized by writer ID.

---

## 2. Writer Identification / Retrieval

### 2.1 Training

```bash
python train.py \
    --dataset_path '/path/to/dataset' \
    --dataset 'CVL' \
    --image-type 'binarised' \
    --batchsize 16
```

**Arguments:**

| Argument | Description |
|---|---|
| `--dataset_path` | Path to the dataset root |
| `--dataset` | Dataset name, e.g. `CVL` or `bullinger` |
| `--image-type` | Image preprocessing type: `binarised` or `grayscale` |
| `--batchsize` | Training batch size |

### 2.2 Dataset Format

For writer identification/retrieval, the dataset should be flat (no writer subfolders):

```
dataset_path/
├── train/
│   ├── img_001.png
│   ├── img_002.png
│   └── ...
└── test/
    ├── img_001.png
    └── ...
```

---

## Notes

- The style encoder (1.1) and the identification/retrieval model (2.1) are separate models with different dataset layouts — double-check the expected format before training.
- Custom dataloaders can be substituted for either pipeline as long as the `train/`/`test/` split contract is respected.
