"""Specimen-separated ResNet-18 training and portable classifier export."""

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class TrainingCancelled(Exception):
    pass


def check_cancel(cancel):
    if cancel is not None and cancel.is_set():
        raise TrainingCancelled('Training cancelled. No ready model was published.')


def prepare_dataset(root, inventory, seed=42, cancel=None):
    root = Path(root).resolve()
    with Path(inventory).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if not {'filename', 'specimen_id', 'family', 'genus'}.issubset(reader.fieldnames or []):
            raise ValueError('Inventory requires filename, specimen_id, family, and genus columns.')
        rows = list(reader)
    if not rows:
        raise ValueError('Inventory has no images.')
    records, filenames, pixels, specimen_labels = [], set(), {}, {}
    for number, row in enumerate(rows, 2):
        check_cancel(cancel)
        row = {key: (value or '').strip() for key, value in row.items() if key}
        genus, specimen = row['genus'], row['specimen_id']
        if row['family'] != 'Linyphiidae' or not genus.isalpha() or not genus[0].isupper():
            raise ValueError(f'Row {number}: confirmed Linyphiidae family and genus required.')
        if not specimen:
            raise ValueError(f'Row {number}: specimen_id is required; do not use a different ID per view.')
        if row.get('issue'):
            raise ValueError(f'Row {number}: resolve the inventory issue before training.')
        if specimen in specimen_labels and specimen_labels[specimen] != genus:
            raise ValueError(f'Specimen {specimen} has conflicting genus labels.')
        specimen_labels[specimen] = genus
        path = (root / row['filename']).resolve()
        if not row['filename'] or not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f'Row {number}: image must exist inside the dataset folder.')
        if path in filenames:
            raise ValueError(f'Duplicate image path: {row["filename"]}')
        filenames.add(path)
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
            digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        if digest in pixels:
            raise ValueError(f'Duplicate image pixels: {row["filename"]} and {pixels[digest]}. Remove duplicates.')
        pixels[digest] = row['filename']
        records.append(dict(filename=path.relative_to(root).as_posix(), specimen_id=specimen,
                            genus=genus, sha256=digest))
    genera = sorted(set(specimen_labels.values()))
    if len(genera) < 2:
        raise ValueError('Genus classification requires at least two genera.')
    rng = random.Random(seed)
    assignment = {}
    for genus in genera:
        specimens = sorted(s for s, label in specimen_labels.items() if label == genus)
        if len(specimens) < 3:
            raise ValueError(f'{genus}: at least 3 distinct specimens are needed for train/validation/test. More are recommended.')
        rng.shuffle(specimens)
        holdout = max(1, int(len(specimens) * 0.2))
        for i, specimen in enumerate(specimens):
            assignment[specimen] = 'test' if i < holdout else 'validation' if i < 2 * holdout else 'train'
    for record in records:
        record['split'] = assignment[record['specimen_id']]
    return dict(dataset_root=str(root), genera=genera, seed=seed, records=records)


def select_threshold(predictions, target_accuracy):
    # Select on validation only; maximize coverage while meeting the requested precision.
    candidates = sorted({row['score'] for row in predictions})
    for threshold in candidates:
        accepted = [row for row in predictions if row['score'] >= threshold]
        accuracy = sum(row['true'] == row['predicted'] for row in accepted) / len(accepted)
        if accuracy >= target_accuracy:
            return dict(confidence_threshold=max(threshold, 1e-8), reject_all=False,
                        validation_coverage=len(accepted) / len(predictions),
                        validation_accepted_accuracy=accuracy)
    return dict(confidence_threshold=1.0, reject_all=True,
                validation_coverage=0.0, validation_accepted_accuracy=None)


def metrics(predictions, genera, policy):
    matrix = [[0 for _ in genera] for _ in genera]
    for row in predictions:
        matrix[row['true']][row['predicted']] += 1
    recalls, f1s, per_genus = [], [], {}
    for index, genus in enumerate(genera):
        tp = matrix[index][index]
        support = sum(matrix[index])
        predicted = sum(row[index] for row in matrix)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(recall)
        f1s.append(f1)
        per_genus[genus] = dict(support=support, precision=precision, recall=recall, f1=f1)
    accepted = [row for row in predictions if not policy['reject_all'] and row['score'] >= policy['confidence_threshold']]
    return dict(count=len(predictions), accuracy=sum(r['true'] == r['predicted'] for r in predictions) / len(predictions),
                balanced_accuracy=sum(recalls) / len(genera), macro_f1=sum(f1s) / len(genera),
                coverage=len(accepted) / len(predictions),
                accepted_accuracy=sum(r['true'] == r['predicted'] for r in accepted) / len(accepted) if accepted else None,
                per_genus=per_genus, confusion_matrix=matrix, genus_order=genera)


def train(root, inventory, output, epochs=10, batch_size=16, learning_rate=0.001,
          seed=42, target_accuracy=0.9, pretrained=True, image_size=224,
          progress=lambda message: None, cancel=None):
    import torch
    from torchvision.models import resnet18, ResNet18_Weights
    from .model import image_transform

    if type(epochs) is not int or not 1 <= epochs <= 1000:
        raise ValueError('Epochs must be between 1 and 1000.')
    if type(batch_size) is not int or not 1 <= batch_size <= 256:
        raise ValueError('Batch size must be between 1 and 256.')
    if not math.isfinite(learning_rate) or not 0 < learning_rate <= 1:
        raise ValueError('Learning rate must be greater than 0 and at most 1.')
    if not math.isfinite(target_accuracy) or not 0 < target_accuracy <= 1:
        raise ValueError('Target accuracy must be greater than 0 and at most 1.')
    if type(image_size) is not int or not 32 <= image_size <= 512:
        raise ValueError('Image size must be between 32 and 512.')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Seed must be an integer from 0 to 4294967295.')
    progress('Validating images and specimen IDs...')
    plan = prepare_dataset(root, inventory, seed, cancel)
    check_cancel(cancel)
    run = Path(output).resolve() / ('run-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8])
    run.mkdir(parents=True, exist_ok=False)
    (run / 'split.json').write_text(json.dumps(plan, indent=2))
    torch.manual_seed(seed)
    progress('Loading ResNet-18' + (' ImageNet weights (downloaded if not cached)...' if pretrained else ' with random weights (pipeline testing only)...'))
    model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    check_cancel(cancel)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.fc = torch.nn.Linear(model.fc.in_features, len(plan['genera']))
    model.eval()
    transform = image_transform(dict(image_size=image_size, mean=MEAN, std=STD))
    classes = {genus: i for i, genus in enumerate(plan['genera'])}
    rows_by_split = {split: [r for r in plan['records'] if r['split'] == split] for split in ('train', 'validation', 'test')}

    def batches(rows, order=None):
        indices = list(range(len(rows))) if order is None else order
        for offset in range(0, len(indices), batch_size):
            check_cancel(cancel)
            selected = [rows[i] for i in indices[offset:offset + batch_size]]
            images = []
            for row in selected:
                with Image.open(Path(plan['dataset_root']) / row['filename']) as image:
                    images.append(transform(ImageOps.exif_transpose(image).convert('RGB')))
            yield selected, torch.stack(images), torch.tensor([classes[r['genus']] for r in selected])

    def evaluate(rows):
        predictions, losses = [], []
        model.eval()
        with torch.inference_mode():
            for selected, images, labels in batches(rows):
                logits = model(images)
                if not torch.isfinite(logits).all():
                    raise ValueError('Training produced non-finite predictions.')
                losses.extend(torch.nn.functional.cross_entropy(logits, labels, reduction='none').tolist())
                probabilities = logits.softmax(1)
                for row, label, probs in zip(selected, labels.tolist(), probabilities.tolist()):
                    prediction = max(range(len(probs)), key=probs.__getitem__)
                    predictions.append(dict(filename=row['filename'], specimen_id=row['specimen_id'],
                                            true=label, predicted=prediction, score=probs[prediction], probabilities=probs))
        return predictions, sum(losses) / len(losses)

    optimizer = torch.optim.Adam(model.fc.parameters(), lr=learning_rate)
    counts = Counter(r['specimen_id'] for r in rows_by_split['train'])
    genus_specimens = Counter({g: len({r['specimen_id'] for r in rows_by_split['train'] if r['genus'] == g}) for g in plan['genera']})
    sample_weights = [1 / (counts[r['specimen_id']] * genus_specimens[r['genus']]) for r in rows_by_split['train']]
    generator = torch.Generator().manual_seed(seed)
    history, best_loss, best_state, best_epoch = [], float('inf'), None, 0
    for epoch in range(1, epochs + 1):
        check_cancel(cancel)
        # Keep pretrained batch-normalization statistics fixed, including for batch size one.
        model.eval()
        sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True, generator=generator)
        running_loss, seen = 0.0, 0
        for selected, images, labels in batches(rows_by_split['train'], list(sampler)):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(images), labels)
            if not torch.isfinite(loss):
                raise ValueError('Training loss is non-finite; check the learning rate and images.')
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(selected)
            seen += len(selected)
        validation, validation_loss = evaluate(rows_by_split['validation'])
        if validation_loss < best_loss:
            best_loss, best_state, best_epoch = validation_loss, copy.deepcopy(model.state_dict()), epoch
        history.append(dict(epoch=epoch, train_loss=running_loss / seen, validation_loss=validation_loss))
        progress(f'Epoch {epoch}/{epochs}: training loss {running_loss / seen:.4f}; validation loss {validation_loss:.4f}')
    model.load_state_dict(best_state)
    validation, _ = evaluate(rows_by_split['validation'])
    policy = select_threshold(validation, target_accuracy)
    progress('Evaluating the selected model on held-out test specimens...')
    test, _ = evaluate(rows_by_split['test'])

    def specimen_predictions(predictions):
        groups = defaultdict(list)
        for row in predictions:
            groups[row['specimen_id']].append(row)
        result = []
        for specimen, views in groups.items():
            scores = [sum(view['probabilities'][i] for view in views) / len(views) for i in range(len(classes))]
            label = max(range(len(scores)), key=scores.__getitem__)
            result.append(dict(specimen_id=specimen, true=views[0]['true'], predicted=label, score=scores[label]))
        return result

    report = dict(architecture='resnet18', training_method='frozen backbone, trained linear head', pretrained=pretrained,
                  seed=seed, epochs=epochs, best_epoch=best_epoch, batch_size=batch_size, learning_rate=learning_rate,
                  target_validation_accuracy=target_accuracy, threshold_policy=policy, history=history,
                  split_counts={s: dict(images=len(rows), specimens=len({r['specimen_id'] for r in rows})) for s, rows in rows_by_split.items()},
                  validation=metrics(validation, plan['genera'], policy), test=metrics(test, plan['genera'], policy),
                  test_specimen_averaged=metrics(specimen_predictions(test), plan['genera'], policy),
                  limitations=['Family assumed, not verified.', 'Unknown genera were not evaluated.',
                               'Threshold is selected on validation images; scores are not calibrated probabilities.',
                               'Small specimen counts produce unreliable estimates.',
                               'Specimen-averaged metrics combine views; the app predicts one image at a time.'])
    check_cancel(cancel)
    (run / 'evaluation.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    with (run / 'test_predictions.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['filename', 'specimen_id', 'true_genus', 'predicted_genus', 'score', 'accepted'])
        for row in test:
            writer.writerow([row['filename'], row['specimen_id'], plan['genera'][row['true']], plan['genera'][row['predicted']],
                             row['score'], not policy['reject_all'] and row['score'] >= policy['confidence_threshold']])
    progress('Exporting model and manifest...')
    model.eval()
    traced = torch.jit.trace(model, torch.zeros(1, 3, image_size, image_size))
    traced.save(str(run / 'classifier.ts'))
    check_cancel(cancel)
    manifest = dict(family='Linyphiidae', task='genus_classification', format='torchscript', weights='classifier.ts',
                    genera=plan['genera'], image_size=image_size, mean=MEAN, std=STD,
                    confidence_threshold=policy['confidence_threshold'], reject_all=policy['reject_all'],
                    architecture='resnet18', evaluation='evaluation.json', family_verified=False)
    temporary = run / 'manifest.json.tmp'
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(run / 'manifest.json')
    progress('Training complete: ' + str(run))
    return dict(manifest=str(run / 'manifest.json'), report=report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--inventory', required=True)
    parser.add_argument('--output', default='models/linyphiidae')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--target-accuracy', type=float, default=0.9)
    parser.add_argument('--no-pretrained', action='store_true', help='Random frozen features: pipeline testing only.')
    args = parser.parse_args()
    try:
        result = train(args.dataset, args.inventory, args.output, args.epochs, args.batch_size,
                       args.learning_rate, args.seed, args.target_accuracy, not args.no_pretrained, progress=print)
        print('Load model:', result['manifest'])
    except (ValueError, OSError, TrainingCancelled) as error:
        parser.exit(1, str(error) + '\n')


if __name__ == '__main__':
    main()
