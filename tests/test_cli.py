"""Collection calibration and evaluation must work without loading backbones."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def command(self, *args):
        return subprocess.run([sys.executable, '-m', 'hec_cd', *map(str, args)],
                              cwd=ROOT, capture_output=True, text=True)

    def test_help_does_not_load_models_or_require_checkpoints(self):
        result = self.command('--help')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('predict', result.stdout)

    def test_calibration_freezes_one_threshold_and_evaluates_saved_masks(self):
        # A 20x20 foreground survives the cross opening with four corners removed.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scores = root/'scores'
            scores.mkdir()
            image = np.zeros((64, 64), dtype=np.uint8)
            image[10:30, 10:30] = 200
            Image.fromarray(image).save(scores/'one.png')
            Image.fromarray(np.zeros_like(image)).save(scores/'empty.png')
            output = root/'decoded'
            result = self.command('calibrate', '--scores', scores, '--output', output)
            self.assertEqual(result.returncode, 0, result.stderr)
            calibration = json.loads((output/'calibration.json').read_text())
            self.assertEqual(calibration['selected_u8'], 200)
            self.assertEqual(calibration['nonzero_count'], 400)
            self.assertEqual(calibration['sample_count'], 2)
            self.assertEqual(calibration['parameters']['rho'], 10)
            mask = np.asarray(Image.open(output/'predictions/one.png'))
            self.assertEqual(np.count_nonzero(mask), 396)
            self.assertEqual(np.count_nonzero(np.asarray(Image.open(output/'predictions/empty.png'))), 0)
            Image.fromarray((image > 0).astype('uint8')*255).save(root/'target.png')
            with (root/'labels.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['id', 'mask'])
                writer.writeheader()
                writer.writerow({'id': 'one', 'mask': 'target.png'})
            result = self.command('evaluate', '--predictions', output/'predictions',
                                  '--manifest', root/'labels.csv')
            self.assertEqual(result.returncode, 0, result.stderr)
            metrics = json.loads(result.stdout)
            self.assertEqual(metrics['counts'], {'tp':396, 'fp':0, 'fn':4, 'tn':3696})
            self.assertAlmostEqual(metrics['f1'], 792/796)

    def test_rejects_existing_output_without_overwriting(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'scores').mkdir()
            Image.fromarray(np.zeros((32,32),dtype='uint8')).save(root/'scores/a.png')
            (root/'out').mkdir()
            sentinel = root/'out/important.txt'
            sentinel.write_text('retain')
            result = self.command('calibrate', '--scores', root/'scores', '--output', root/'out')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('empty', result.stderr)
            self.assertEqual(sentinel.read_text(), 'retain')

    def test_calibration_applies_component_filter_to_saved_masks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scores = root/'scores'
            scores.mkdir()
            image = np.zeros((40, 40), dtype=np.uint8)
            image[2:16, 2:16] = 128
            image[20:33, 20:35] = 128
            Image.fromarray(image).save(scores/'one.png')
            output = root/'decoded'
            result = self.command('calibrate', '--scores', scores, '--output', output)
            self.assertEqual(result.returncode, 0, result.stderr)
            calibration = json.loads((output/'calibration.json').read_text())
            self.assertEqual(calibration['selected_u8'], 128)
            self.assertEqual(calibration['nonzero_count'], 391)
            mask = np.asarray(Image.open(output/'predictions/one.png'))
            self.assertEqual(np.count_nonzero(mask), 192)
            self.assertEqual(mask[8, 8], 255)
            self.assertEqual(mask[25, 25], 0)

    def test_empty_score_collection_is_an_error(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'scores').mkdir()
            result = self.command('calibrate', '--scores', root/'scores', '--output', root/'out')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('No score', result.stderr)
            self.assertFalse((root/'out/calibration.json').exists())

    def test_prediction_rejects_unsafe_ids_before_loading_models(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root/'pairs.csv'
            manifest.write_text('id,t1,t2\n../escape,first.png,second.png\n')
            result = self.command('predict', '--manifest', manifest, '--output', root/'out',
                '--satlas-root', root, '--sam3-root', root,
                '--satlas-checkpoint', root/'satlas.pth', '--sam3-checkpoint', root/'sam3.pt',
                '--dinov3-checkpoint', root/'dino.safetensors')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Invalid sample id', result.stderr)

    def test_case_collisions_are_rejected_before_any_prediction_is_written(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            Image.fromarray(np.zeros((256,256,3),dtype='uint8')).save(root/'a.png')
            (root/'pairs.csv').write_text('id,t1,t2\nScene,a.png,a.png\nscene,a.png,a.png\n')
            result = self.command('predict', '--manifest', root/'pairs.csv', '--output', root/'out',
                '--satlas-root', root, '--sam3-root', root,
                '--satlas-checkpoint', root/'s.pth', '--sam3-checkpoint', root/'s.pt',
                '--dinov3-checkpoint', root/'d.safetensors')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Duplicate sample id', result.stderr)
            self.assertFalse((root/'out').exists())

    def test_reserved_file_names_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'pairs.csv').write_text('id,t1,t2\nCON,first.png,second.png\n')
            result = self.command('predict', '--manifest', root/'pairs.csv', '--output', root/'out',
                '--satlas-root', root, '--sam3-root', root,
                '--satlas-checkpoint', root/'s.pth', '--sam3-checkpoint', root/'s.pt',
                '--dinov3-checkpoint', root/'d.safetensors')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Invalid sample id', result.stderr)

    def test_evaluation_rejects_continuous_scores_in_place_of_predictions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'predictions').mkdir()
            Image.fromarray(np.full((32,32),100,dtype='uint8')).save(root/'predictions/a.png')
            Image.fromarray(np.zeros((32,32),dtype='uint8')).save(root/'target.png')
            (root/'labels.csv').write_text('id,mask\na,target.png\n')
            result = self.command('evaluate', '--predictions', root/'predictions', '--manifest',root/'labels.csv')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('binary prediction', result.stderr)


if __name__ == '__main__':
    unittest.main()
