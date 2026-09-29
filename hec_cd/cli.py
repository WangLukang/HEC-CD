"""Unlabeled collection inference, calibration, and separate evaluation."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image

from .core import (harmonic_consensus, metrics_from_counts, postprocess,
                   quantize, select_unlabeled_histogram)


def read_manifest(path: Path, columns: tuple[str, ...]) -> list[dict]:
    """Paths are relative to the manifest; inference never requests mask columns."""
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if not {'id', *columns}.issubset(reader.fieldnames or []):
            raise ValueError('Manifest requires columns: ' + ','.join(('id', *columns)))
        rows, seen = [], set()
        for row in reader:
            identifier = row['id'].strip()
            reserved = {'CON','PRN','AUX','NUL',*(f'COM{i}' for i in range(1,10)),*(f'LPT{i}' for i in range(1,10))}
            if (not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', identifier)
                    or identifier.endswith('.') or identifier.split('.')[0].upper() in reserved):
                raise ValueError(f'Invalid sample id: {identifier!r}')
            if identifier.casefold() in seen:
                raise ValueError(f'Duplicate sample id: {identifier}')
            seen.add(identifier.casefold())
            entry = {'id': identifier}
            for key in columns:
                if not row[key] or not row[key].strip():
                    raise ValueError(f'Missing {key} path for {identifier}')
                value = Path(row[key].strip())
                entry[key] = value if value.is_absolute() else path.parent/value
                if not entry[key].is_file():
                    raise FileNotFoundError(entry[key])
            rows.append(entry)
    if not rows:
        raise ValueError('Manifest has no samples')
    return rows


def new_output(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f'Output directory must be new or empty: {path}')
    path.mkdir(parents=True, exist_ok=True)


def read_score(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        score = np.asarray(image)
    if score.ndim != 2 or score.dtype != np.uint8:
        raise ValueError(f'Expected an 8-bit grayscale score PNG: {path}')
    return score


def decode_collection(score_files: list[Path], output: Path) -> dict:
    """Freeze one threshold from all score pixels before writing predictions."""
    if not score_files:
        raise ValueError('No score PNG files found')
    histogram = np.zeros(256, dtype=np.int64)
    for file in score_files:
        histogram += np.bincount(read_score(file).ravel(), minlength=256)
    result = select_unlabeled_histogram(histogram)
    calibration = asdict(result)
    calibration.update(sample_count=len(score_files), histogram=histogram.tolist())
    (output/'calibration.json').write_text(json.dumps(calibration, indent=2)+'\n', encoding='utf-8')
    predictions = output/'predictions'
    predictions.mkdir()
    for file in score_files:
        mask = postprocess(read_score(file), result.selected_u8)
        Image.fromarray(mask.astype(np.uint8)*255).save(predictions/file.name)
    return calibration


def predict(args) -> None:
    rows = read_manifest(args.manifest, ('t1','t2'))
    # Check the complete collection before loading checkpoints or writing results.
    for row in rows:
        for key in ('t1','t2'):
            with Image.open(row[key]) as image:
                if image.size != (256,256):
                    raise ValueError(f'Expected a 256x256 input tile: {row[key]}')
    new_output(args.output)
    from .backbones import FrozenBackbones
    from .evidence import object_evidence
    model = FrozenBackbones(
        satlas_root=args.satlas_root, satlas_checkpoint=args.satlas_checkpoint,
        sam3_root=args.sam3_root, sam3_checkpoint=args.sam3_checkpoint,
        dinov3_checkpoint=args.dinov3_checkpoint, device=args.device)
    scores = args.output/'scores'
    scores.mkdir()
    for index, row in enumerate(rows, 1):
        with Image.open(row['t1']) as a, Image.open(row['t2']) as b:
            evidence = model.infer_pair(a.convert('RGB'), b.convert('RGB'))
        ro = object_evidence(evidence['instances1'], evidence['instances2'],
            evidence['p1'], evidence['p2'], evidence['features1'], evidence['features2'])
        score = harmonic_consensus(evidence['rs'], ro)
        Image.fromarray(quantize(score)).save(scores/(row['id']+'.png'))
        print(f'[{index}/{len(rows)}] {row["id"]}', flush=True)
    calibration = decode_collection(sorted(scores.glob('*.png')), args.output)
    print(f'GHT threshold: {calibration["selected_u8"]}/255')


def evaluate(predictions: Path, manifest: Path) -> dict:
    counts = np.zeros(4, dtype=np.int64)
    rows = read_manifest(manifest, ('mask',))
    for row in rows:
        predicted = read_score(predictions/(row['id']+'.png'))
        if not np.isin(predicted, [0,1,255]).all():
            raise ValueError(f'Expected a binary prediction, not a score map: {row["id"]}')
        predicted = predicted > 0
        target = read_score(row['mask'])
        if not np.isin(target, [0,1,255]).all():
            raise ValueError(f'Expected a binary ground-truth mask: {row["mask"]}')
        target = target > 0
        if predicted.shape != target.shape:
            raise ValueError(f'Prediction/label shape mismatch: {row["id"]}')
        counts += [np.count_nonzero(predicted & target), np.count_nonzero(predicted & ~target),
                   np.count_nonzero(~predicted & target), np.count_nonzero(~predicted & ~target)]
    result = metrics_from_counts(*map(int, counts))
    result.update(counts=dict(zip(('tp','fp','fn','tn'),map(int,counts))), sample_count=len(rows))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description='HEC-CD building change detection')
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('predict', help='Infer a complete unlabeled collection and calibrate GHT')
    p.add_argument('--manifest', type=Path, required=True, help='CSV: id,t1,t2')
    p.add_argument('--output', type=Path, required=True)
    for option in ('satlas-root','satlas-checkpoint','sam3-root','sam3-checkpoint','dinov3-checkpoint'):
        p.add_argument('--'+option, type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p = commands.add_parser('calibrate', help='Apply GHT to a collection of saved 8-bit score PNGs')
    p.add_argument('--scores', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = commands.add_parser('evaluate', help='Evaluate binary predictions; labels are used only here')
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True, help='CSV: id,mask')
    args = parser.parse_args()
    try:
        if args.command == 'predict':
            predict(args)
        elif args.command == 'calibrate':
            files = sorted(args.scores.glob('*.png'))
            if not files:
                raise ValueError('No score PNG files found')
            new_output(args.output)
            result = decode_collection(files, args.output)
            print(f'GHT threshold: {result["selected_u8"]}/255')
        else:
            print(json.dumps(evaluate(args.predictions,args.manifest),indent=2))
    except (ValueError,FileNotFoundError) as error:
        parser.exit(2, f'Error: {error}\n')
