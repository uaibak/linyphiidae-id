import csv
from pathlib import Path

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp'}


def inspect_dataset(directory):
    from PIL import Image

    root = Path(directory)
    rows = []
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative = path.relative_to(root)
        genus = relative.parts[0] if len(relative.parts) > 1 else ''
        issue = '' if genus else 'Missing genus folder'
        if genus and (not genus.isalpha() or not genus[0].isupper()):
            issue = 'Check genus spelling and capitalization'
        try:
            with Image.open(path) as image:
                width, height = image.size
                image.verify()
        except (OSError, ValueError, SyntaxError) as error:
            width, height = 0, 0
            issue = 'Unreadable image: ' + str(error)
        rows.append(dict(filename=relative.as_posix(), specimen_id='',
                         family='Linyphiidae', genus=genus, view='', sex='',
                         location='', width=width, height=height, issue=issue))
    return rows


def export_inventory(rows, destination):
    columns = ['filename', 'specimen_id', 'family', 'genus', 'view', 'sex',
               'location', 'width', 'height', 'issue']
    with Path(destination).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
