# HEC-CD

Training-free building change detection with frozen SatlasPretrain, SAM 3, and DINOv3.
The method combines **spatial building evidence**, **shared-mask temporal verification**,
**harmonic consensus**, and **GHT calibration** with an Otsu-derived variance prior.
No target-domain network training is required.

## Setup

Use Python 3.10 and a CUDA-enabled PyTorch installation. The reference
environment uses PyTorch 2.5.1, torchvision 0.20.1, timm 1.0.27, NumPy 2.2.6,
SciPy 1.15.3, and scikit-image 0.25.2. GPU inference was checked on an RTX 3090.

Clone the repository and run the remaining commands from its root directory:

```bash
git clone https://github.com/WangLukang/HEC-CD.git
cd HEC-CD
```

In a Python 3.10 environment, install a CUDA-enabled build of PyTorch 2.5.1
and torchvision 0.20.1 using the
[official installation instructions](https://pytorch.org/get-started/previous-versions/#v251)
for your system, then install the remaining dependencies:

```bash
pip install -r requirements.txt
pip install --no-deps --target external/satlas satlaspretrain-models==0.3.1
git clone https://github.com/earth-insights/SegEarth-OV-3.git external/SegEarth-OV-3
git -C external/SegEarth-OV-3 checkout 87e5cc18c8bc9c0b4713b975587281ac1c97c79a
```

The SAM 3 adapter uses the semantic-response and presence outputs from this
SegEarth-OV-3 revision.

Download the following weights separately, accepting each provider's access terms:

| Model | Source | Local filename |
|---|---|---|
| SatlasPretrain Aerial Swin-B SI | [AllenAI](https://huggingface.co/allenai/satlas-pretrain) | `checkpoints/aerial_swinb_si.pth` |
| SAM 3 | [Meta](https://huggingface.co/facebook/sam3) | `checkpoints/sam3.pt` |
| DINOv3 ViT-L SAT-493M | [timm](https://huggingface.co/timm/vit_large_patch16_dinov3.sat493m) | `checkpoints/dinov3_vitl.safetensors` |

The DINOv3 file is the provider's `model.safetensors`, renamed locally.

## Data

LEVIR-CD and S2Looking follow their original splits. WHU-CD uses the 744-pair
test setting consistent with [DynamicEarth](https://arxiv.org/abs/2501.12931).
HEC-CD processes non-overlapping 256 x 256 RGB tiles: 2,048 / 744 / 16,000
test pairs for LEVIR-CD / WHU-CD / S2Looking.

See [Data preparation](docs/data.md) for downloads, preprocessing scripts,
labels, directories and locally generated manifests.

## Run

After data preparation, run each dataset separately. CSV paths are relative to
the CSV file. For example, on LEVIR-CD:

```bash
python -m hec_cd predict --manifest data/LEVIR-CD256/manifests/test_pairs.csv --output outputs/levir \
  --satlas-root external/satlas \
  --satlas-checkpoint checkpoints/aerial_swinb_si.pth \
  --sam3-root external/SegEarth-OV-3 --sam3-checkpoint checkpoints/sam3.pt \
  --dinov3-checkpoint checkpoints/dinov3_vitl.safetensors
```

Outputs are 8-bit consensus maps in `scores/`, binary maps in `predictions/`,
and the threshold/histogram in `calibration.json`.
GHT uses the collection's nonzero scores with `rho=10`.
Inference and calibration use images only; the output folder must be new or empty.
A worked [output example](examples/README.md) shows both evidence pathways,
fusion and the final prediction.

To decode an existing collection of 8-bit grayscale consensus maps:

```bash
python -m hec_cd calibrate --scores outputs/levir/scores --output outputs/decoded
```

## Evaluate and test

For evaluation only, provide a separate CSV with columns `id,mask`. Masks must be
binary grayscale images (0/1 or 0/255), with unchanged pixels coded as zero.

```bash
python -m hec_cd evaluate --predictions outputs/levir/predictions --manifest data/LEVIR-CD256/manifests/test_labels.csv
python -m unittest discover -s tests -v
```

Evaluation reports global pixel counts and P/R/F1/IoU/OA as fractions in `[0,1]`.
See [Baseline sources and settings](docs/baselines.md) for the comparison methods.

## Implementation

`backbones.py` extracts frozen model outputs; `evidence.py` constructs and verifies
shared candidate masks; `core.py` handles fusion and decoding; `ght.py` implements
threshold calibration. The release preserves the original intermediate quantization:
Satlas residual truncation, SAM response rounding and half-precision presence,
then round-to-nearest consensus scores. Spatial decoding applies the inclusive
calibrated cutoff followed by one 3 x 3 cross opening, then retains four-connected
components with at least 192 pixels in each tile.
SAM 3 processes each date separately in BF16; DINOv3 processes both dates in FP16.

## License

Original HEC-CD code and documentation are provided under the [MIT License](LICENSE),
except for [`hec_cd/ght.py`](hec_cd/ght.py), which is adapted from
[Barron's GHT reference implementation](https://github.com/jonbarron/hist_thresh)
and remains under [Apache-2.0](LICENSE-GHT.txt). Its original copyright,
attribution, and modification notice are retained in the file.

External libraries, model code, pretrained weights, and datasets are not covered
by the HEC-CD license; their respective licenses and access terms still apply.

## Contact

For usage questions or bug reports, please open a
[GitHub issue](https://github.com/WangLukang/HEC-CD/issues).
