"""Crop the official S2Looking test split into aligned 256-pixel tiles."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

if __package__:
    from .prepare_ovcd_standard_splits import write_manifests
else:
    from prepare_ovcd_standard_splits import write_manifests


def binary_label(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        a = np.asarray(im)
    if a.ndim == 3:
        a = a.max(axis=2)
    if a.shape != (1024,1024):
        raise ValueError(f'{path}: expected a 1024x1024 label')
    return a > 0


def prepare_test(source: Path, output: Path, expected_count: int = 1000) -> int:
    source = source/'test'
    names = sorted(p.name for p in (source/'Image1').glob('*.png'))
    if len(names) != expected_count:
        raise ValueError(f'Expected {expected_count} official test pairs, got {len(names)}')
    for sub in ('Image2','label1','label2'):
        if {p.name for p in (source/sub).glob('*.png')} != set(names):
            raise ValueError(f'{sub}: file set differs from Image1')
    if (source/'label').is_dir() and {p.name for p in (source/'label').glob('*.png')} != set(names):
        raise ValueError('label: file set differs from Image1')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Output directory must be new or empty')
    for sub in ('A','B','label','list'):
        (output/sub).mkdir(parents=True, exist_ok=True)
    tiles = []
    for name in names:
        with Image.open(source/'Image1'/name) as im:
            first = np.asarray(im.convert('RGB'))
        with Image.open(source/'Image2'/name) as im:
            second = np.asarray(im.convert('RGB'))
        if first.shape != (1024,1024,3) or second.shape != first.shape:
            raise ValueError(f'{name}: expected 1024x1024 RGB images')
        label = binary_label(source/'label1'/name) | binary_label(source/'label2'/name)
        if (source/'label').is_dir() and not np.array_equal(label, binary_label(source/'label'/name)):
            raise ValueError(f'{name}: combined label differs from label1 OR label2')
        for row in range(4):
            for col in range(4):
                tile = f'{Path(name).stem}_r{row}_c{col}.png'
                sl = np.s_[row*256:(row+1)*256,col*256:(col+1)*256]
                for sub, a in [('A',first),('B',second),('label',label.astype(np.uint8)*255)]:
                    Image.fromarray(a[sl]).save(output/sub/tile)
                tiles.append(tile)
    (output/'list/test.txt').write_text('\n'.join(tiles)+'\n', encoding='utf-8')
    write_manifests(output, 'test', tiles)
    return len(tiles)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Official root containing test/Image1, Image2, label1 and label2')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(f'Prepared {prepare_test(args.source, args.output)} test tiles')
