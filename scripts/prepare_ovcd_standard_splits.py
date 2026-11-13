"""Validate released 256-pixel packages and export their existing split lists."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

COUNTS = {
    'levir': {'train': 7120, 'val': 1024, 'test': 2048},
    'whu': {'train': 5947, 'val': 743, 'test': 744},
}


def write_manifests(root: Path, split: str, names: list[str], label_directory: str = 'label') -> None:
    """Separate image-only inference inputs from evaluation labels."""
    output = root/'manifests'
    output.mkdir(exist_ok=True)
    for suffix, columns in [('pairs', ('id','t1','t2')), ('labels', ('id','mask'))]:
        with (output/f'{split}_{suffix}.csv').open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for name in names:
                row = {'id': Path(name).stem, 't1': f'../A/{name}',
                       't2': f'../B/{name}', 'mask': f'../{label_directory}/{name}'}
                writer.writerow({key: row[key] for key in columns})


def prepare_dataset(root: Path, expected: dict[str, int]) -> dict[str, int]:
    splits = {}
    for split, count in expected.items():
        names = [s.strip() for s in (root/'list'/f'{split}.txt').read_text(encoding='utf-8-sig').splitlines() if s.strip()]
        if len(names) != count or len({n.casefold() for n in names}) != count:
            raise ValueError(f'{split}: expected {count} unique samples, got {len(names)}')
        if any(Path(n).name != n or '/' in n or '\\' in n or Path(n).suffix.lower() != '.png' for n in names):
            raise ValueError(f'{split}: expected PNG filenames without directories')
        splits[split] = names
    seen = set()
    for split, names in splits.items():
        keys = {n.casefold() for n in names}
        if seen & keys:
            raise ValueError(f'{split}: split lists overlap')
        seen |= keys
    union = {n for names in splits.values() for n in names}
    for sub in ('A','B','label'):
        if {p.name for p in (root/sub).glob('*.png')} != union:
            raise ValueError(f'{sub}: files differ from the split-list union')
    for name in sorted(union):
        for sub in ('A','B','label'):
            with Image.open(root/sub/name) as im:
                if im.size != (256,256):
                    raise ValueError(f'{sub}/{name}: expected 256x256 pixels')
    # Match the original evaluation's >=128 rule and retain source labels.
    (root/'label_binary').mkdir(exist_ok=True)
    for name in sorted(union):
        with Image.open(root/'label'/name) as im:
            a = np.asarray(im.convert('L'))
        binary = a > 0 if a.max() <= 1 else a >= 128
        Image.fromarray(binary.astype(np.uint8)*255).save(root/'label_binary'/name)
    for split, names in splits.items():
        write_manifests(root, split, names, label_directory='label_binary')
    return {split: len(names) for split, names in splits.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=COUNTS, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    print(prepare_dataset(args.root, COUNTS[args.dataset]))
