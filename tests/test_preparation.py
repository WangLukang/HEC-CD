"""Dataset preparation preserves split membership, pixels and usable CSV paths."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from hec_cd.cli import read_manifest
from scripts.prepare_ovcd_standard_splits import prepare_dataset
from scripts.prepare_s2looking_dynamicearth import prepare_test


class PreparationTests(unittest.TestCase):
    def test_manifest_paths_resolve_and_keep_original_split(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'list').mkdir()
            for split in ('train', 'val', 'test'):
                (root/'list'/f'{split}.txt').write_text(f'{split}_1.png\n')
                for sub in ('A', 'B', 'label'):
                    (root/sub).mkdir(exist_ok=True)
                    Image.new('L' if sub == 'label' else 'RGB', (256,256)).save(root/sub/f'{split}_1.png')
            # Released labels can contain near-white pixels (e.g. 254).
            raw = np.zeros((256,256), np.uint8)
            raw[0,:4] = [1,127,254,255]
            Image.fromarray(raw).save(root/'label/test_1.png')
            prepare_dataset(root, {'train': 1, 'val': 1, 'test': 1})
            rows = read_manifest(root/'manifests/test_pairs.csv', ('t1','t2'))
            self.assertEqual([r['id'] for r in rows], ['test_1'])
            self.assertEqual(rows[0]['t1'].resolve(), (root/'A/test_1.png').resolve())
            labels = read_manifest(root/'manifests/test_labels.csv', ('mask',))
            self.assertEqual(labels[0]['mask'].resolve(), (root/'label_binary/test_1.png').resolve())
            with Image.open(labels[0]['mask']) as im:
                np.testing.assert_array_equal(np.array(im)>0, raw>=128)
            with Image.open(root/'label/test_1.png') as im:
                np.testing.assert_array_equal(np.array(im),raw)
            # Reject accidental split leakage before rewriting manifests.
            (root/'list/train.txt').write_text('test_1.png\n')
            with self.assertRaisesRegex(ValueError, 'overlap'):
                prepare_dataset(root, {'train': 1, 'val': 1, 'test': 1})

    def test_s2_crops_are_pixel_exact_and_labels_are_unioned(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/'source'
            out = Path(folder)/'prepared'
            y,x = np.indices((1024,1024))
            first = np.stack([x%256, y%256, (x//256+4*(y//256))*10], axis=-1).astype(np.uint8)
            second = 255-first
            a = np.zeros((1024,1024), np.uint8); a[260:270,510:520]=255
            b = np.zeros_like(a); b[0:8,0:8]=255
            for sub, data in [('Image1',first),('Image2',second),('label1',a),('label2',b)]:
                (root/'test'/sub).mkdir(parents=True)
                Image.fromarray(data).save(root/'test'/sub/'scene.png')
            prepare_test(root, out, expected_count=1)
            rows = read_manifest(out/'manifests/test_pairs.csv', ('t1','t2'))
            self.assertEqual(len(rows),16)
            labels = read_manifest(out/'manifests/test_labels.csv', ('mask',))
            for r,c in np.ndindex(4,4):
                name=f'scene_r{r}_c{c}.png'; sl=np.s_[r*256:(r+1)*256,c*256:(c+1)*256]
                for sub, data in [('A',first),('B',second),('label',np.maximum(a,b))]:
                    with Image.open(out/sub/name) as im:
                        np.testing.assert_array_equal(np.asarray(im),data[sl])
            self.assertEqual(len(labels),16)
            with self.assertRaisesRegex(ValueError,'empty'):
                prepare_test(root,out,expected_count=1)


if __name__ == '__main__':
    unittest.main()
