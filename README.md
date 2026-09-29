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
SegEarth-OV-3 revision. Do not substitute another SAM 3 processor without checking
that interface. The repository does not need the mmsegmentation evaluation pipeline.

Download the following weights separately, accepting each provider's access terms:

| Model | Source | Local filename |
|---|---|---|
| SatlasPretrain Aerial Swin-B SI | [AllenAI](https://huggingface.co/allenai/satlas-pretrain) | `checkpoints/aerial_swinb_si.pth` |
| SAM 3 | [Meta](https://huggingface.co/facebook/sam3) | `checkpoints/sam3.pt` |
| DINOv3 ViT-L SAT-493M | [timm](https://huggingface.co/timm/vit_large_patch16_dinov3.sat493m) | `checkpoints/dinov3_vitl.safetensors` |

The DINOv3 file is the provider's `model.safetensors`, renamed locally. Data,
checkpoints, and third-party source trees are not included.

## Run

Prepare registered **256 x 256 RGB tiles** and a CSV containing one complete target
collection. Paths are relative to the CSV file (absolute paths also work):

```csv
id,t1,t2
scene_001,A/scene_001.png,B/scene_001.png
scene_002,A/scene_002.png,B/scene_002.png
```

```bash
python -m hec_cd predict --manifest data/pairs.csv --output outputs/run1 \
  --satlas-root external/satlas \
  --satlas-checkpoint checkpoints/aerial_swinb_si.pth \
  --sam3-root external/SegEarth-OV-3 --sam3-checkpoint checkpoints/sam3.pt \
  --dinov3-checkpoint checkpoints/dinov3_vitl.safetensors
```

Outputs are quantized consensus maps in `scores/`, binary maps in `predictions/`,
and the frozen threshold/histogram in `calibration.json`. Run each dataset
separately. GHT uses all nonzero scores in that collection, with fixed
`rho=10`; no ground-truth masks are read during inference or calibration.
The output folder must be new or empty.

To decode an existing collection of 8-bit grayscale consensus maps:

```bash
python -m hec_cd calibrate --scores outputs/run1/scores --output outputs/decoded
```

## Evaluate and test

For evaluation only, provide a separate CSV with columns `id,mask`. Masks must be
binary grayscale images (0/1 or 0/255), with unchanged pixels coded as zero.

```bash
python -m hec_cd evaluate --predictions outputs/run1/predictions --manifest data/labels.csv
python -m unittest discover -s tests -v
```

Evaluation reports global pixel counts and P/R/F1/IoU/OA as fractions in `[0,1]`.

## Implementation

`backbones.py` extracts frozen model outputs; `evidence.py` constructs and verifies
shared candidate masks; `core.py` handles fusion and decoding; `ght.py` implements
threshold calibration. The release preserves the original intermediate quantization:
Satlas residual truncation, SAM response rounding and half-precision presence,
then round-to-nearest consensus scores. Spatial decoding applies the inclusive
calibrated cutoff followed by one 3 x 3 cross opening.
SAM 3 processes each date separately with text encoding inside BF16 autocast;
DINOv3 processes the two dates together in FP16. Mixed-precision results may
vary slightly with hardware, software versions, and batch size.

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
