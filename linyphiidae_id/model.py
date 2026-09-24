import json
import math
from pathlib import Path


def image_transform(metadata):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((metadata['image_size'], metadata['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(metadata['mean'], metadata['std']),
    ])


def read_manifest(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('family') != 'Linyphiidae' or data.get('task') != 'genus_classification':
        raise ValueError('Select a Linyphiidae genus classification manifest.')
    if data.get('format') != 'torchscript':
        raise ValueError('This mode requires a TorchScript classifier, not YOLOv7 weights.')
    genera = data.get('genera')
    if (not isinstance(genera, list) or len(genera) < 2 or
            any(not isinstance(g, str) or not g.isalpha() or not g[0].isupper() for g in genera)
            or len(set(genera)) != len(genera)):
        raise ValueError('The manifest must contain unique genus names in model output order.')
    size = data.get('image_size')
    threshold = data.get('confidence_threshold')
    if type(size) is not int or not 32 <= size <= 2048:
        raise ValueError('image_size must be between 32 and 2048.')
    if type(threshold) not in (int, float) or not 0 < threshold <= 1:
        raise ValueError('confidence_threshold must be greater than 0 and at most 1.')
    if type(data.get('reject_all', False)) is not bool:
        raise ValueError('reject_all must be a boolean.')
    for field in ('mean', 'std'):
        value = data.get(field)
        if (not isinstance(value, list) or len(value) != 3 or
                any(type(n) not in (int, float) or not math.isfinite(n) for n in value)):
            raise ValueError(field + ' must contain three finite numbers.')
    if any(n <= 0 for n in data['std']):
        raise ValueError('Standard deviations must be positive.')
    if not isinstance(data.get('weights'), str):
        raise ValueError('The manifest must specify its weights file.')
    weights = (path.parent / data['weights']).resolve()
    if not weights.is_file():
        raise ValueError('Model weights are missing: ' + str(weights))
    return data, weights


class GenusClassifier:
    def __init__(self, manifest):
        import torch

        self.metadata, weights = read_manifest(manifest)
        self.model = torch.jit.load(str(weights), map_location='cpu').eval()
        self.transform = image_transform(self.metadata)

    def predict(self, image):
        import torch
        from PIL import ImageOps

        with torch.inference_mode():
            logits = self.model(self.transform(ImageOps.exif_transpose(image).convert('RGB')).unsqueeze(0))
        genera = self.metadata['genera']
        if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (1, len(genera)):
            raise ValueError('Model output does not match the manifest genus list.')
        if not torch.isfinite(logits).all():
            raise ValueError('Model returned invalid scores.')
        scores = logits.softmax(dim=1)[0].tolist()
        ranked = sorted(zip(genera, scores), key=lambda item: item[1], reverse=True)
        accepted = not self.metadata.get('reject_all', False) and ranked[0][1] >= self.metadata['confidence_threshold']
        return ranked[:3], accepted
