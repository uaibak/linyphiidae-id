# Contributing to Linyphiidae ID

Linyphiidae ID is a research prototype for preparing specimen datasets,
training genus classifiers, and reviewing predictions. Contributions should
support this Linyphiidae-focused workflow. No trained classifier is bundled;
family membership is assumed rather than inferred.

## Development Setup

Use Python 3.11 and Tk 8.6 or newer. Follow the platform-specific environment
setup in [README.md](README.md), then install dependencies inside your activated
virtual environment:

```sh
python -m pip install -r requirements.txt
python main.py
```

Create a separate branch for your changes. Keep changes scoped and follow the
existing code style. Discuss major model, dependency, or workflow changes in an
issue before implementing them.

## Checks Before Submission

Before submitting a change:

```sh
python -m pip check
python -m unittest discover -s tests -v
python -m compileall -q linyphiidae_id main.py
```

Tests create synthetic images and model weights in temporary directories. They
do not require a trained Linyphiidae model or personal specimen photographs.
For GUI changes, also run `python main.py` with Tk 8.6 or newer and check image
preview, window resizing, dataset audit and inventory export, model loading,
prediction export, and error states. For training UI changes, check progress,
cancellation, and loading a completed model without freezing the window.

Include regression tests for changed behavior. Document changes to inventory
columns, preprocessing, genus ordering, model manifests, or exported results.
GitHub Actions checks dependencies, syntax, and non-GUI tests; it does not
verify desktop rendering or biological accuracy.

## Dataset and Model Contributions

Do not commit virtual environments, trained weights, private images, specimen
metadata, logs, or prediction history. Dataset/model folders are ignored except
for their documentation and example manifest. Use small synthetic fixtures in
tests and temporary directories for generated outputs. Automated tests must
not require private datasets or model downloads.

Use expert-reviewed genus labels and consistent specimen IDs. Keep all views
of a specimen in the same split, prevent duplicate-image leakage, and select
model settings and confidence thresholds without using the held-out test set.
Preserve the distinction between a pipeline smoke test and a biological
evaluation. Three specimens per genus is only the current technical minimum,
not evidence of adequate training data.

Do not describe a classifier as taxonomically validated without a documented,
independent specimen-level evaluation. Family is currently assumed, not inferred.
Scores are not calibrated probabilities, and rejection of unseen genera has
not been validated. Report evaluation limitations alongside any accuracy claims.

## Issues and Pull Requests

For a bug report, include steps to reproduce, expected and actual behavior,
your operating system, Python and Tk versions, and the relevant error message.
Remove private paths, specimen metadata, credentials, and sensitive collection
locations from logs or screenshots. Use shareable or synthetic examples.

For a pull request, summarize the change, link related issues, list the checks
you ran, and note any remaining limitations. Include screenshots for visible
UI changes and update the README when the user workflow changes.

## License and Attribution

The project retains the GNU GPL version 3 license in [LICENSE](LICENSE).
Keep that standard license text and existing author notices intact. Upstream
SpiderID_APP attribution is recorded in the README's Attribution and License
section; do not remove it when refactoring or replacing code.

Only submit material you have permission to contribute under the project's
license. Identify the source and license of any third-party code or assets in
your pull request. Dependencies, datasets, and trained weights may have separate
terms; do not assume their redistribution is covered by this repository's
license.
