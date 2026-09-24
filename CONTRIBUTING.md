# Contributing to Linyphiidae ID

Use Python 3.11 and follow the setup instructions in README.md.

Before submitting a change:

```sh
python -m pip check
python -m unittest discover -s tests -v
python -m compileall -q linyphiidae_id main.py
```

Tests create synthetic images and model weights in temporary directories. They
do not require a trained Linyphiidae model or personal specimen photographs.
For GUI changes, also run `python main.py` with Tk 8.6 or newer and check image
preview, window resizing, dataset import, and error states.

Keep changes scoped and document model input/output changes. Include relevant
tests and describe checks performed in the pull request.

Do not commit virtual environments, trained weights, private images, specimen
metadata, logs, or prediction history. Dataset/model folders are ignored except
for their documentation and example manifest. Keep original author notices and
the existing license. Only contribute material you have permission to share.

Do not describe a classifier as taxonomically validated without a documented,
independent specimen-level evaluation. Family is currently assumed, not inferred.
