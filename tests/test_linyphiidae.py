import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from linyphiidae_id.dataset import inspect_dataset, export_inventory
from linyphiidae_id.model import GenusClassifier, read_manifest


class WorkflowTests(unittest.TestCase):
    def test_inventory_preserves_images_and_flags_problems(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'Examplegenus').mkdir()
            Image.new('RGB', (20, 30)).save(root / 'Examplegenus' / 'one.jpg')
            Image.new('RGB', (20, 30)).save(root / 'unlabeled.jpg')
            (root / 'Examplegenus' / 'bad.jpg').write_text('not an image')
            rows = inspect_dataset(root)
            self.assertEqual(len(rows), 3)
            self.assertEqual(sum(bool(row['issue']) for row in rows), 2)
            self.assertTrue(all(row['specimen_id'] == '' for row in rows))
            export_inventory(rows, root / 'inventory.csv')
            with (root / 'inventory.csv').open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 3)

    def test_model_contract_and_uncertainty(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 2))
            torch.nn.init.zeros_(model[-1].weight)
            torch.nn.init.zeros_(model[-1].bias)
            torch.jit.trace(model.eval(), torch.zeros(1, 3, 32, 32)).save(str(root / 'test.ts'))
            metadata = dict(family='Linyphiidae', task='genus_classification', format='torchscript', weights='test.ts', genera=['Genusa', 'Genusb'], image_size=32, mean=[0, 0, 0], std=[1, 1, 1], confidence_threshold=0.8)
            path = root / 'model.json'
            path.write_text(json.dumps(metadata))
            ranked, accepted = GenusClassifier(path).predict(Image.new('RGB', (80, 40)))
            self.assertFalse(accepted)
            self.assertAlmostEqual(ranked[0][1], 0.5)
            metadata['family'] = 'OtherFamily'
            path.write_text(json.dumps(metadata))
            with self.assertRaises(ValueError):
                read_manifest(path)
            metadata['family'] = 'Linyphiidae'
            metadata['genera'] = ['Genusa', 'Genusa']
            path.write_text(json.dumps(metadata))
            with self.assertRaises(ValueError):
                read_manifest(path)


if __name__ == '__main__':
    unittest.main()
