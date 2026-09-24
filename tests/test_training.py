import csv
import json
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from linyphiidae_id.model import GenusClassifier
from linyphiidae_id.training import (TrainingCancelled, prepare_dataset, select_threshold, train)


def fixture(root):
    rows = []
    for genus_index, genus in enumerate(['Genusa', 'Genusb']):
        (root / genus).mkdir()
        for specimen in range(5):
            for view in range(2):
                name = f'{genus}/{specimen}-{view}.png'
                rng = np.random.default_rng(100 * genus_index + 2 * specimen + view)
                Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(root / name)
                rows.append(dict(filename=name, specimen_id=f'{genus}-{specimen}', family='Linyphiidae', genus=genus))
    path = root / 'inventory.csv'
    write_inventory(path, rows)
    return path, rows


def write_inventory(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['filename', 'specimen_id', 'family', 'genus'])
        writer.writeheader()
        writer.writerows(rows)


class TrainingTests(unittest.TestCase):
    def test_specimen_splits_are_disjoint_and_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory, _ = fixture(root)
            plan = prepare_dataset(root, inventory)
            self.assertEqual(plan, prepare_dataset(root, inventory))
            groups = {s: {r['specimen_id'] for r in plan['records'] if r['split'] == s}
                      for s in ['train', 'validation', 'test']}
            self.assertFalse(groups['train'] & groups['test'])
            self.assertFalse(groups['train'] & groups['validation'])
            self.assertFalse(groups['test'] & groups['validation'])
            for split in groups:
                self.assertEqual({r['genus'] for r in plan['records'] if r['split'] == split}, {'Genusa', 'Genusb'})

    def test_missing_ids_conflicts_and_path_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory, rows = fixture(root)
            rows[0]['specimen_id'] = ''
            write_inventory(inventory, rows)
            with self.assertRaisesRegex(ValueError, 'specimen_id'):
                prepare_dataset(root, inventory)
            rows[0]['specimen_id'] = 'Genusb-0'
            write_inventory(inventory, rows)
            with self.assertRaisesRegex(ValueError, 'conflicting'):
                prepare_dataset(root, inventory)
            rows[0]['specimen_id'] = 'Genusa-0'
            rows[0]['filename'] = '../outside.jpg'
            write_inventory(inventory, rows)
            with self.assertRaisesRegex(ValueError, 'inside'):
                prepare_dataset(root, inventory)

    def test_duplicate_pixels_rejected_even_with_different_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory, rows = fixture(root)
            with Image.open(root / rows[0]['filename']) as image:
                image.save(root / rows[-1]['filename'])
            with self.assertRaisesRegex(ValueError, 'Duplicate image pixels'):
                prepare_dataset(root, inventory)

    def test_minimum_distinct_specimens(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory, rows = fixture(root)
            rows = [r for r in rows if r['specimen_id'].endswith(('-0', '-1'))]
            write_inventory(inventory, rows)
            with self.assertRaisesRegex(ValueError, '3 distinct specimens'):
                prepare_dataset(root, inventory)

    def test_threshold_can_reject_all(self):
        wrong = [dict(true=0, predicted=1, score=0.99)]
        self.assertTrue(select_threshold(wrong, 0.9)['reject_all'])
        mixed = wrong + [dict(true=1, predicted=1, score=0.999)]
        policy = select_threshold(mixed, 0.9)
        self.assertFalse(policy['reject_all'])
        self.assertEqual(policy['confidence_threshold'], 0.999)
        self.assertEqual(policy['validation_coverage'], 0.5)

    def test_cancel_publishes_no_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory, _ = fixture(root)
            cancel = threading.Event()
            cancel.set()
            with self.assertRaises(TrainingCancelled):
                train(root, inventory, root / 'output', pretrained=False, cancel=cancel)
            self.assertFalse(list(root.rglob('manifest.json')))

    def test_actual_resnet_training_export_and_reload(self):
        import torch
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                inventory, rows = fixture(root)
                result = train(root, inventory, root / 'output', epochs=1, batch_size=4,
                               pretrained=False, image_size=32)
                manifest = Path(result['manifest'])
                self.assertTrue(manifest.is_file())
                self.assertTrue(manifest.with_name('test_predictions.csv').is_file())
                report = json.loads(manifest.with_name('evaluation.json').read_text())
                self.assertEqual(report['test']['count'], 4)
                self.assertEqual(report['test_specimen_averaged']['count'], 2)
                self.assertEqual(report['architecture'], 'resnet18')
                self.assertFalse(report['pretrained'])
                model = GenusClassifier(manifest)
                with Image.open(root / rows[0]['filename']) as image:
                    ranked, _ = model.predict(image)
                self.assertEqual(len(ranked), 2)
                self.assertAlmostEqual(sum(score for _, score in ranked), 1, places=5)
                data = json.loads(manifest.read_text())
                data.update(reject_all=True, confidence_threshold=0.000001)
                manifest.write_text(json.dumps(data))
                with Image.open(root / rows[0]['filename']) as image:
                    self.assertFalse(GenusClassifier(manifest).predict(image)[1])
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == '__main__':
    unittest.main()
