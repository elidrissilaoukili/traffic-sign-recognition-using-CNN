# HIICNN — Hybrid Image-Improving and CNN Stacking Ensemble for Traffic Sign Recognition

A PyTorch pipeline for the GTSRB benchmark (43 German traffic sign classes)
combining classical image-enhancement preprocessing with a stacking ensemble
of three ImageNet-pretrained CNN backbones — ResNet50, VGG16, and
EfficientNet-B0 — partially fine-tuned and fused by a learned meta-learner,
with latency/FPS reporting for a real-time autonomous-driving framing.

Implemented end-to-end in a single notebook: **`HIICNN_improved.ipynb`**.

## Results

Measured on the official GTSRB test split.

| Model                        | Accuracy   | Precision  | Recall     | F1         |
|-------------------------------|:----------:|:----------:|:----------:|:----------:|
| ResNet50                      | 81.93%     | 77.66%     | 78.02%     | 77.25%     |
| VGG16                         | 81.38%     | 79.44%     | 77.24%     | 77.56%     |
| EfficientNet-B0                | 56.39%     | 49.40%     | 53.71%     | 50.08%     |
| Naive-average ensemble         | 85.54%     | 82.30%     | 81.76%     | 81.49%     |
| **HIICNN stacking ensemble**   | **86.21%** | **85.05%** | **81.92%** | **82.92%** |

The stacking meta-learner beats every individual backbone **and** a
naive-averaging baseline — this is the actual evidence for the "hybrid
ensemble improves robustness" claim, not just the architecture on paper.

### Inference latency (single image, batch size 1)

| Model                        | ms / image | FPS    |
|-------------------------------|:----------:|:------:|
| ResNet50                      | 8.94        | 111.9  |
| VGG16                         | 3.34        | 299.1  |
| EfficientNet-B0                | 13.46       | 74.3   |
| **HIICNN stacking ensemble**   | **42.09**   | **23.8** |

> **Honest note on "real-time":** running all three backbones sequentially
> plus the meta-learner currently lands at ~23.8 FPS, below the conventional
> ≥30 FPS bar used for real-time video/perception pipelines. This is raw,
> unoptimized inference — no quantization, ONNX/TensorRT export, or batching
> has been applied yet. See [Future work](#future-work).

## 1. Architecture overview

```
raw GTSRB image
      │
      ▼
┌─────────────────────────────┐
│  Image Enhancement Stage    │  CLAHE contrast → unsharp mask → sharpening kernel
└─────────────────────────────┘
      │
      ▼
   Resize to 112×112, normalize (ImageNet mean/std)
      │
      ├──────────────┬──────────────┬───────────────┐
      ▼              ▼              ▼
  ResNet50        VGG16        EfficientNet-B0       last block of each
  (43-way head)  (43-way head)  (43-way head)         fine-tuned at a small
      │              │              │                 LR; new head at a
      ▼              ▼              ▼                 larger LR
  softmax(43)    softmax(43)    softmax(43)
      └──────────────┴──────────────┘
                     │  concat → 129-d vector
                     ▼
            Meta-Learner (small MLP)
            trained in mini-batches with
            a held-out meta-val split
                     │
                     ▼
             Final 43-class prediction
```

## 2. Preprocessing / image enhancement

Three classical techniques are chained, in order, before resizing:

1. **CLAHE contrast enhancement** — adaptive histogram equalization on the
   L-channel of LAB color space, to recover detail in over/under-exposed
   sign photos (glare, shadows, dusk driving conditions).
2. **Unsharp masking** — Gaussian-blur the image, subtract it from the
   original, and add the result back, boosting edge contrast across the
   whole sign.
3. **Sharpening convolution kernel** — a cheap 3×3 kernel
   (`[[0,-1,0],[-1,5,-1],[0,-1,0]]`) for a final edge-emphasis pass, cheap
   enough to run per-frame in a real-time system.

Images are then resized to 112×112 and normalized with ImageNet statistics,
since all three backbones use ImageNet-pretrained weights. Toggles for each
stage (`USE_CLAHE_CONTRAST`, `USE_UNSHARP_MASK`, `USE_SHARPENING`) live in the
config cell.

### Dataset layout expected

The dataset-loading cell reads the standard Kaggle-mirror GTSRB layout,
expected directly under `DATA_ROOT` (`data/` by default):

```
data/
├── Train/                # per-class subfolders (0..42) of training images
├── Test/                 # flat folder of test images
├── Train.csv             # Width,Height,Roi.X1,Roi.Y1,Roi.X2,Roi.Y2,ClassId,Path
└── Test.csv
```

`GTSRBCSVDataset` reads the `Path`/`ClassId` columns to locate each image and
label, and crops to the `Roi.X1/Y1/X2/Y2` bounding box (the annotated sign
region within the padded raw image) before handing the image to the
enhancement/resize pipeline.

## 3. Base models / stacking

- Each backbone (ResNet50, VGG16, EfficientNet-B0) is loaded with ImageNet
  weights via `torchvision.models`. Rather than a single frozen/unfrozen
  switch, **only the last block of each backbone is fine-tuned**
  (`layer4` for ResNet50, the last conv block for VGG16, the last MBConv
  stage for EfficientNet-B0) at a small learning rate (`BACKBONE_LR = 1e-5`),
  while a new head (`Linear → BatchNorm/ReLU → Dropout → Linear(43)`) trains
  at a larger rate (`HEAD_LR = 1e-3`). Fully-frozen ImageNet features don't
  transfer well to small, tightly-cropped traffic-sign icons — a domain
  quite different from the ImageNet photos they were pretrained on — so this
  partial fine-tuning is the main driver behind the accuracy jump from
  ~50% (fully frozen) to ~80%+ per backbone.
- Training batches use **class-weighted sampling** (`USE_WEIGHTED_SAMPLER`)
  to counter GTSRB's heavy per-class imbalance, plus label smoothing, AdamW,
  mixed precision, and an early-stopping patience of `EARLY_STOP_PATIENCE`
  epochs; the best checkpoint (by held-out validation accuracy) is saved.
- **Stacking meta-learner** (`MetaLearner`): a small MLP that takes the
  concatenated 43-dim softmax vectors from all three backbones (129 inputs
  total) and learns to weight/combine them into a final prediction — this is
  what distinguishes HIICNN from a naive averaging ensemble. It's trained on
  a held-out validation split that none of the three backbones were directly
  trained on (correct stacking practice), using real mini-batches, AdamW,
  and cosine LR decay, with its own held-out meta-validation slice so
  training can be checked for convergence rather than run blind.
- The evaluation cell also reports a **naive-average ensemble** baseline
  (simple mean of the three softmax outputs) alongside the trained stacking
  ensemble, so the value added by the meta-learner is visible — see the
  [Results](#results) table above.

## 4. Real-time framing

The benchmarking cell reports per-image latency (ms) and throughput (FPS)
for each backbone individually and for the full HIICNN pipeline end-to-end
(three backbones + meta-learner combined into a single forward pass via
`EnsembleWrapper`).

Numbers are measured with `batch_size=1` (the realistic single-frame case
for an onboard camera), CUDA-synchronized where available, after a warmup
period. Results are saved to `results/latency_benchmark.json`. See the
[latency table above](#inference-latency-single-image-batch-size-1) for
current numbers and the honest caveat on the "real-time" framing.

## 5. Evaluation

On the official GTSRB test split, the evaluation cell reports for every
backbone and for the ensemble:

- Accuracy
- Macro Precision / Recall / F1-score
- Confusion matrix (plotted per model)
- A JSON comparison table (`results/comparison_table.json`) covering
  single-backbone vs. stacking-ensemble vs. naive-average performance

## 6. Project layout

```
HIICNN/
├── HIICNN_improved.ipynb   # everything: config, enhancement, dataset,
│                            # models, training, evaluation, benchmarking
├── checkpoints/             # saved model weights (resnet50.pt, vgg16.pt,
│                            # efficientnet_b0.pt, meta_learner.pt)
├── results/
│   ├── comparison_table.json     # accuracy / precision / recall / F1
│   └── latency_benchmark.json    # ms-per-image and FPS per model
└── data/                    # GTSRB dataset (Kaggle layout, not committed)
```

## 7. How to run

### Setup

```bash
# 1. Create Python environment:
python -m venv env

# 2. Activate the environment:
.\env\Scripts\Activate.ps1        # Windows PowerShell
# source env/bin/activate         # macOS/Linux

# 3. Install a CUDA-matched PyTorch build (pick your CUDA version):
#    https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install the rest of the dependencies
pip install pandas numpy opencv-python scikit-learn matplotlib tqdm jupyter
```

Make sure your `data/` folder (`Train/`, `Test/`, `Train.csv`, `Test.csv`)
sits at the project root, next to the notebook, before running.

### Run

```bash
jupyter notebook HIICNN_improved.ipynb
```

Run the cells top to bottom: config → enhancement → dataset/dataloaders →
backbone/meta-learner definitions → backbone training loop → meta-learner
training → evaluation → latency benchmark. Each stage saves its checkpoints
or results before the next stage needs them, so you can also stop after
training and come back later to re-run just evaluation/benchmarking against
the saved checkpoints.

## 8. Notes on adapting the CV bullet's claims to this code

- **"Fine-tuned via transfer learning"**: accurate as of the current
  version — `FINE_TUNE_LAST_BLOCK` controls whether the last block of each
  backbone trains alongside the new head (default `True`); setting it to
  `False` reverts to a fully-frozen conv base, which is what capped accuracy
  around ~50% in earlier runs.
- **"Real-time"**: only a genuine claim once measured FPS on your target
  hardware exceeds your target frame rate (e.g. 30 FPS for a 30fps dash-cam
  feed). Current measured throughput is **23.8 FPS** for the full ensemble —
  below that bar. See [Future work](#future-work) before quoting "real-time"
  in a report or portfolio.
- Swap `resnet50` for `resnet18`/`resnet34` in `BACKBONES` and the backbone
  builder cell for a lighter/faster backbone if latency is the priority over
  raw accuracy — the `BACKBONE_BUILDERS` pattern makes this a small,
  localized change.
- The results and latency tables above come from an actual run of this
  notebook on the GTSRB test split — re-run the notebook on your own
  hardware/data split if you need numbers specific to your environment.

## Future work

- **Close the real-time gap.** Quantize (int8) or export the backbones to
  ONNX/TensorRT, batch inference where possible, or distill the ensemble
  into a single smaller model to push past 30 FPS.
- **Trim the ensemble.** EfficientNet-B0 trails ResNet50 and VGG16 by a wide
  margin (56.4% vs. 81-82%) and adds the most latency (13.46 ms/image) —
  worth re-tuning it specifically or dropping it and re-evaluating the
  2-model ensemble's accuracy/latency trade-off.
- **Broader fine-tuning sweep.** Compare unfreezing one vs. two backbone
  blocks, and try lower/higher backbone learning rates, to see how much
  further accuracy can move.
- **Robustness testing.** Evaluate under weather/lighting corruptions (rain,
  glare, motion blur, night) relevant to autonomous-driving deployment, not
  just the clean GTSRB test set.

## License


