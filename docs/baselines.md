# Baseline sources and evaluation settings

Third-party implementations and weights are available from their authors:

| Method | Official repository | Input size: LEVIR / WHU / S2Looking |
|---|---|---|
| SCM | [UCD-SCM](https://github.com/StephenApX/UCD-SCM) | 1024 / 256 / 1024 |
| APE-DINO, APE-DINOv2 | [DynamicEarth](https://github.com/likyoo/DynamicEarth) | 256 / 256 / 256 |
| SAM-DINOv2-SOV | [DynamicEarth](https://github.com/likyoo/DynamicEarth) | 1024 / 256 / 1024 |
| OpenDPR | [OpenDPR](https://github.com/guoqi2002/OpenDPR) | 1024 / 256 / 256 |
| OmniOVCD | [OmniOVCD](https://github.com/Erxucomeon/OmniOVCD) | 256 / 256 / 1024 |
| ReA-OVCD | [ReA-OVCD](https://github.com/Funny0101/ReA-OVCD) | 256 / 256 / 256 |

All sizes are square. Native 1024-pixel predictions are cropped into aligned
256-pixel tiles after inference. All methods are evaluated on the test pixels
described in [Data preparation](data.md), using global confusion counts.

- **SCM:** FastSAM-X, CLIP ViT-B/32, RFF and PSA; confidence 0.4, IoU 0.9,
  model size 1024. WHU tiles are resized to 1024 for inference and binary outputs
  are returned to 256 by nearest-neighbor sampling. A collection-level Otsu
  threshold is estimated from nonzero distances in the 1024 inference space.
- **APE-DINO / APE-DINOv2:** DynamicEarth I-M-C; APE confidence 0.2,
  instance IoU 0.01, change threshold 135; comparator patch sizes 16 / 14.
- **SAM-DINOv2-SOV:** DynamicEarth M-C-I with SAM ViT-H, DINOv2 and SegEarth-OV;
  SAM grid densities 32 / 8 / 32 and change thresholds 145 / 125 / 145.
- **OpenDPR:** SAM ViT-H and DINOv2 prototype retrieval; SAM grid densities
  32 / 8 / 32 and change thresholds 145 / 125 / 145. S2Looking uses the
  LEVIR-CD configuration and support set.
- **OmniOVCD:** building probability thresholds 0.4 / 0.4 / 0.6 and instance
  IoU thresholds 0.25 / 0.3 / 0.1; detector confidence 0.5.
- **ReA-OVCD:** released LEVIR-CD and WHU-CD parameters; S2Looking uses the
  LEVIR-CD parameters. Sliding inference is disabled for pre-cropped tiles.
- **PCA-KM:** an independent implementation based on
  [Celik (2009)](https://doi.org/10.1109/LGRS.2009.2025059), using 5 x 5 difference
  patches, PCA variance retention 0.95, two-cluster K-means (`n_init=10`, seed 0),
  and the minority cluster as change, without morphological postprocessing.

Source code, weights and licenses for third-party methods remain with the
linked repositories.
