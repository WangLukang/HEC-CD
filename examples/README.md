# Example outputs

One cached result from each test collection is included:

| Folder | Sample ID | Full collection size | Inclusive cutoff (0-255) |
|---|---|---:|---:|
| `levir` | `test_32_9` | 2,048 | 84 |
| `whu` | `whucd_01797` | 744 | 74 |
| `s2looking` | `132_r2_c1` | 16,000 | 67 |

Each folder contains a 256 x 256 uint8 consensus map in `scores/`, a 0/255
binary result in `predictions/`, and `calibration.json` from its **full test
collection**. The JSON records the nonzero-score calibration, inclusive cutoff,
sample count and 256-bin histogram. Zero counts are retained in the saved
histogram but excluded from GHT fitting.

To verify an example from the repository root:

```python
import json
from pathlib import Path
import numpy as np
from PIL import Image
from hec_cd.core import postprocess, select_unlabeled_histogram

root = Path('examples/levir')
record = json.loads((root / 'calibration.json').read_text())
cutoff = select_unlabeled_histogram(np.array(record['histogram'])).selected_u8
assert cutoff == record['selected_u8']
score_path = next((root / 'scores').glob('*.png'))
with Image.open(score_path) as image:
    prediction = postprocess(np.asarray(image), cutoff)
with Image.open(root / 'predictions' / score_path.name) as image:
    assert np.array_equal(prediction, np.asarray(image) > 0)
```

The included image is a sample of the full collection; calibrating that image
alone defines a different collection. Retrieve the matching input pair through
[Data preparation](../docs/data.md).
