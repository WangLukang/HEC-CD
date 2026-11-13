# Data preparation

## Sources and splits

| Dataset | Source | Train / validation / test | HEC-CD test inputs |
|---|---|---|---|
| LEVIR-CD | [Official data](https://justchenhao.github.io/LEVIR/) | 445 / 64 / 128 original scenes | 2,048 tiles |
| WHU-CD | [Official data](https://study.rsgis.whu.edu.cn/pages/download/building_dataset.html) | 5,947 / 743 / 744 tiles | 744 tiles |
| S2Looking | [Official data](https://github.com/S2Looking/Dataset) | 3,500 / 500 / 1,000 original scenes | 16,000 tiles |

LEVIR-CD and S2Looking follow their original splits. Each 1024 x 1024 test
scene is divided into 16 non-overlapping 256 x 256 tiles, without resizing.
WHU-CD follows the 744-pair test setting used by
[DynamicEarth](https://arxiv.org/abs/2501.12931). The supplied split lists are
retained without re-sampling.

## LEVIR-CD and WHU-CD

Download and extract the prepared packages:
[LEVIR-CD256](https://www.dropbox.com/scl/fi/r28vh4c6soxk7q9l2hg1a/LEVIR-CD256.zip?rlkey=h85y8cipx8xq1xzf1265uzm4h&dl=1),
[WHU-CD-256](https://www.dropbox.com/scl/fi/ci39uxbzwmxr3asxs5h1s/WHU-CD-256.zip?rlkey=fl82nzgzsw0wc6gi7botjuapu&dl=1).
These packages already contain aligned 256 x 256 image pairs, binary labels,
and `list/train.txt`, `list/val.txt`, and `list/test.txt`.

```bash
python scripts/prepare_ovcd_standard_splits.py --dataset levir --root data/LEVIR-CD256
python scripts/prepare_ovcd_standard_splits.py --dataset whu --root data/WHU-CD-256
```

The script validates split counts, disjoint membership, paired files and image
dimensions, then exports the original lists as CSV files. Images and source
labels are preserved. Evaluation labels are saved in `label_binary/` as 0/255:
8-bit masks use `value >= 128`; masks already encoded as 0/1 use `value > 0`.

## S2Looking

Extract the official dataset to `data/S2Looking`. Its test folder contains
`Image1`, `Image2`, `label1` (new buildings), and `label2` (demolished buildings).

```bash
python scripts/prepare_s2looking_dynamicearth.py --source data/S2Looking --output data/S2Looking256
```

The script crops the two dates and labels at identical coordinates. The change
label is the union of `label1` and `label2`; an existing combined `label` folder
is checked against that union. Output labels are grayscale 0/255. Tile names
use `<scene>_r<row>_c<column>.png`, with zero-based row and column indices.
The output directory must be new or empty.

## Prepared layout and manifests

```text
dataset/
  A/<id>.png                  # earlier date
  B/<id>.png                  # later date
  label/<id>.png              # source label (S2Looking: binary union)
  label_binary/<id>.png       # LEVIR/WHU evaluation labels
  list/test.txt               # existing test membership
  manifests/test_pairs.csv    # id,t1,t2
  manifests/test_labels.csv   # id,mask
```

CSV paths are relative to the CSV directory, for example
`scene,../A/scene.png,../B/scene.png`. The label CSV points to `label_binary/`
for LEVIR/WHU and `label/` for S2Looking. Prepared labels use zero for unchanged
pixels and 255 for changed pixels. Inference uses only `test_pairs.csv`;
`test_labels.csv` is used separately for evaluation. Complete manifests are
generated locally from the supplied splits.

## Evaluation configuration

Use the weights and command in the root README, replacing `--manifest` and
`--output` for each dataset:

| Dataset | Inference manifest | Output directory |
|---|---|---|
| LEVIR-CD | `data/LEVIR-CD256/manifests/test_pairs.csv` | `outputs/levir` |
| WHU-CD | `data/WHU-CD-256/manifests/test_pairs.csv` | `outputs/whu` |
| S2Looking | `data/S2Looking256/manifests/test_pairs.csv` | `outputs/s2looking` |

Each run pools the nonzero consensus scores from that dataset to estimate one
GHT threshold. Fixed prompts are `building`, `rooftop`, `house`, and
`urban structure`. The feature-discrepancy scale is 0.20 and the relative
variance-prior strength is 10; spatial decoding is described in the README.
Evaluation accumulates TP, FP, FN and TN over all test pixels
before computing P, R, F1, IoU and OA.
