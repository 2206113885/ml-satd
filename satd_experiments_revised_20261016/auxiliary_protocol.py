"""Shared protocol for explicitly invoked revised auxiliary runs (R14/R22/R30/R31/R37/R41).

Importing this module performs no dataset reads, downloads or model loading.
Historical result/data directories are never eligible output destinations.
"""
import csv
import hashlib
import importlib.metadata
import json
import platform
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / 'revised_outputs'
LABEL_ABBR = {
    'Machine Learning Knowledge': 'MLKnow', 'Model Interpretability': 'MLInter',
    'Model Code Comprehension': 'MCC', 'Duplicate Model Code': 'DMC',
    'Duplicate Feature Extraction Code': 'DFEC', 'Weight Configuration': 'Weight Conf',
    'Layer Configuration': 'Layer Conf', 'Hyper-parameter Configuration': 'HP Conf',
    'Code Dependency': 'MLDepend', 'Glue Code': 'GC', 'Custom Data Type': 'CDT',
    'Multiple Languages': 'ML', 'Unnecessary Model Code': 'UMC',
    'Data Processing Configuration': 'Data Conf', 'Plain Old Data Type': 'PODT',
    'Data Storage Configuration': 'DS Conf', 'Abstraction': 'Abs', 'Boundary Erosion': 'BE',
    'Model Code Modifiability': 'MCM', 'Model Code Reusability': 'MCR',
    'Prediction Quality': 'PQ', 'Machine Learning Reliability': 'MLRelia', 'Prototype': 'Proto',
}
ABBR2FULL = {abbr: full for full, abbr in LABEL_ABBR.items()}
ABBREVIATIONS = tuple(sorted(ABBR2FULL))
ALIASES = {re.sub(r'\s+', ' ', value).strip().casefold(): abbr
           for full, abbr in LABEL_ABBR.items() for value in (full, abbr)}
ALIASES.update({'hpconf': 'HP Conf', 'mdepend': 'MLDepend', 'mrelia': 'MLRelia',
                'ds configuration': 'DS Conf', 'machinelearning knowledge': 'MLKnow',
                'machine learning dependency': 'MLDepend'})


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_label(value):
    text = re.sub(r'\s+', ' ', str(value or '')).strip()
    if not text:
        return ''
    label = ALIASES.get(text.casefold())
    if label is None:
        raise ValueError('Gold is outside the 23 ML-SATD labels: ' + text)
    return label


def normalize_text(value):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(value))).strip().casefold()


def read_csv(path):
    csv.field_size_limit(20_000_000)
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = reader.fieldnames or []
    if not rows:
        raise ValueError('Empty input CSV')
    return fields, rows


def check_output_path(path):
    target = Path(path).expanduser().resolve()
    root = OUTPUT_ROOT.resolve()
    if target == root or not target.is_relative_to(root):
        raise ValueError('Choose a fresh child directory under ' + str(OUTPUT_ROOT))
    return target


def new_output_dir(path):
    target = check_output_path(path)
    target.mkdir(parents=True, exist_ok=False)
    return target


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def stable_sample_seed(seed, sample_id, repetition=0):
    raw = json.dumps([int(seed), str(sample_id), int(repetition)], ensure_ascii=False).encode()
    return int(hashlib.sha256(raw).hexdigest()[:8], 16)


def versions(names=()):
    packages = {}
    for name in names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = 'not-installed'
    return {'python': platform.python_version(), 'platform': platform.platform(), 'packages': packages}


def load_labeled(path, id_column='Dataset ID', text_column='Comment Text',
                 gold_column='ML TD Type', project_column='Repo Name', clone_column=None):
    fields, rows = read_csv(path)
    required = [id_column, text_column, gold_column, project_column]
    if clone_column:
        required.append(clone_column)
    if any(name not in fields for name in required):
        raise ValueError('Required columns missing: ' + ', '.join(name for name in required if name not in fields))
    ids = set()
    result = []
    for index, row in enumerate(rows, 2):
        sid, text, project = (row[id_column].strip(), row[text_column], row[project_column].strip())
        if not sid or sid in ids:
            raise ValueError('Missing/duplicate Dataset ID; run the explicit dedup pipeline first')
        if not text.strip() or not project:
            raise ValueError('Missing comment/project at CSV row ' + str(index))
        # [R35/R41 原代码保留]
        # gold = normalize_label(row[gold_column])
        # [R35/R41 修订]
        secondary_column = {'ML TD Type': 'ML TD Type 2', 'ml_satd_type': 'ml_satd_type_2'}.get(gold_column)
        if secondary_column in fields and row[secondary_column].strip():
            raise ValueError('Secondary label is nonempty; use explicit single-label preprocessing first')
        gold = normalize_label(row[gold_column])
        if not gold:
            raise ValueError('Missing gold at CSV row ' + str(index))
        clone = row[clone_column].strip() if clone_column else ''
        if clone_column and not clone:
            raise ValueError('Every row must have an adjudicated clone-group ID')
        ids.add(sid)
        result.append({'sample_id': sid, 'source_row_1based': index, 'comment': text,
                       'project': project, 'gold_label': gold, 'raw_gold': row[gold_column],
                       'filename': row.get('Filename', ''),
                       'introducing_revision': row.get('Comment-Introducing Revision', ''),
                       'clone_group': clone})
    return result
