# Linyphiidae ID

Repository name: `linyphiidae-id`. Python package: `linyphiidae_id`.

A desktop workspace for preparing genus-labeled Linyphiidae images and running
a separately trained genus classifier. **No Linyphiidae classifier is bundled.**

## Run

Requires Python 3.11 and Tk 8.6 or newer. From a clone of this repository, create
the environment and install dependencies before launching. No trained model
is required for image preview and dataset preparation.

### macOS

```sh
source .venv/bin/activate
python main.py
```

On macOS, use Python 3.11 with Tk 8.6 or newer. For a fresh environment:

```sh
brew install python@3.11 python-tk@3.11
$(brew --prefix python@3.11)/bin/python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

### Windows (PowerShell)

Install Python 3.11 with Tcl/Tk support, then run:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe main.py
```

### Linux

Install Python 3.11, its venv support, and the matching Tk package using your
distribution's package manager. A graphical desktop is required to run the UI.

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

## Project Status

Implemented: image preview, genus-folder audit, inventory export, specimen-based
dataset splitting, ResNet-18 training, validation threshold selection, held-out
evaluation, TorchScript export/loading, candidate scores, and CSV results.

Not implemented: family recognition or validated unknown-genus rejection.
This is a research prototype, not a
validated taxonomic identification service.

## Dataset Preparation

Select a folder with one immediate subfolder per genus in the Genus Dataset tab.
The app counts images, flags unreadable files and missing folder labels, and
exports an inventory CSV. Folder spelling checks do not verify taxonomy.

```text
datasets/linyphiidae/
  ConfirmedGenusA/
    specimen001_top.jpg
    specimen001_side.jpg
  ConfirmedGenusB/
    specimen002_top.jpg
```

Use actual expert-confirmed genus names. Add specimen IDs, views, sex, and
location to the exported inventory. No specimen IDs are guessed from filenames.
Keep all photographs of one specimen together when splitting training and test
data. The training workflow enforces this using the specimen IDs in your CSV.

## Train Your Model

1. Open the dataset in **Genus Dataset** and export its inventory.
2. Fill in `specimen_id` for every image. Use one globally unique ID per specimen,
   shared by all its views. Review `family` and `genus`, fix flagged images, and
   clear resolved `issue` cells. Do not invent different IDs for multiple views.
3. In **Train Model**, select the dataset folder, reviewed CSV, and output folder.
4. Leave ImageNet weights enabled for the transfer-learning baseline. The first
   run downloads ResNet-18 weights from PyTorch if they are not cached.
5. Set epochs, batch size, learning rate, and target validation accuracy. Confirm
   label review, then choose **Train and Evaluate**. You can cancel between batches;
   model downloads and individual operations must finish before cancellation.
6. Review the held-out test report, then choose **Use Trained Model** or load the
   exported `manifest.json` in the identification tab.

The CPU baseline freezes ResNet-18's ImageNet backbone and trains its linear genus
head. Image preprocessing matches inference. Training samples are weighted by
genus and specimen so many views of one specimen do not dominate sampling.
This follows PyTorch's [transfer-learning approach](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial).

Each genus needs **at least 3 distinct specimens** to put at least one specimen
in each split. This is a technical minimum, not enough to establish useful accuracy.
Within each genus, approximately 20% of specimens go to validation and 20% to
test (at least one each); the rest go to training. Splits are deterministic for a
given seed. Conflicting labels, missing IDs, unreadable images, repeated paths,
and exact decoded-image duplicates are rejected. Near-duplicate images still
need human review. Only images listed in the inventory are used.

The lowest-validation-loss epoch is selected. A confidence threshold is chosen
on validation images to maximize coverage while meeting the requested accepted
accuracy. If none qualifies, the model rejects all predictions as Uncertain.
Test images are evaluated only after model and threshold selection. Do not tune
settings against the test set repeatedly and then report it as an unbiased test.

Each run creates a new folder containing:

- `split.json`: image hashes, labels, specimen IDs, split assignments, and seed.
- `evaluation.json`: settings, epoch losses, confusion matrix, per-genus precision,
  recall/F1, image-level accuracy, coverage, and specimen-averaged test metrics.
- `test_predictions.csv`: predictions for the held-out test images.
- `classifier.ts` and `manifest.json`: the portable model and its preprocessing.

The manifest is published last. Failed or cancelled runs may retain diagnostic
files but have no ready manifest. Scores are not calibrated probabilities, and
meeting a validation target does not guarantee test or future accuracy. Family
recognition and rejection of unseen genera remain outside this training task.

Command-line equivalent:

```sh
python -m linyphiidae_id.training --dataset datasets/linyphiidae \
  --inventory datasets/linyphiidae/specimen_inventory.csv \
  --output models/linyphiidae --epochs 10 --batch-size 16 \
  --learning-rate 0.001 --target-accuracy 0.9 --seed 42
```

`--no-pretrained` uses frozen random features for offline pipeline testing only;
it is not the recommended training mode. Automated tests use synthetic images
and this option, so they do not download weights or measure biological accuracy.

## Model Contract

Load a trusted JSON model manifest using Load Genus Model. Its TorchScript
weights must return finite raw logits of shape `[batch, number_of_genera]`.
Genus order must exactly match training. Images are EXIF-oriented, converted to
RGB, resized to a square, converted to tensors, and normalized using the manifest.

See [the manifest example](models/linyphiidae/manifest.example.json).
It is a format example, not a usable or validated model. Only load model files
from trusted sources. Manifest metadata is not proof of taxonomic accuracy.

Predictions below the manifest's threshold display Uncertain. This threshold
must be selected using validation data. A high score does not establish that an
unseen genus is supported. Unknown-genus rejection has not been validated.

**Family is assumed to be Linyphiidae, not predicted or verified.** Family
recognition requires additional training and labeled examples from other families.
The app exports this assumption with each prediction.

## Layout

```text
linyphiidae-id/
  main.py                   Source-checkout launcher
  pyproject.toml            Package metadata and desktop entry point
  linyphiidae_id/            Active application, dataset audit, model adapter
  datasets/linyphiidae/      Private genus-organized images (ignored)
  models/linyphiidae/        Private models plus public example manifest
  tests/                    Workflow tests
  .github/workflows/        GitHub Actions checks
```

Run the active app with `python main.py` or `python -m linyphiidae_id` from the
repository root. Optional editable installation adds the `linyphiidae-id` command:

```sh
python -m pip install -e .
linyphiidae-id
```

## Tests

```sh
python -m pip check
python -m unittest discover -s tests -v
```

GitHub Actions runs dependency, syntax, and non-GUI workflow checks on Python
3.11. GUI rendering requires a separate desktop check. See
[CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidance.

## Repository Contents

Virtual environments, private datasets, trained model files, and logs
are excluded from Git. The example model
manifest is included, but it cannot run without separately trained weights.

## Attribution and License

This project is derived from [SpiderID_APP](https://github.com/ThangLC304/SpiderID_APP)
by Luong Cao Thang, with the original project crediting Chih-Hsin Hung and
Chung-Der Hsiao. The previous general-spider application and its assets have
been removed; this acknowledgment preserves the project's provenance.

The original GNU GPL version 3 license is retained in [LICENSE](LICENSE).
Third-party dependencies and model/data assets remain subject to their respective
terms. No new ownership claim is made over upstream code or media.
