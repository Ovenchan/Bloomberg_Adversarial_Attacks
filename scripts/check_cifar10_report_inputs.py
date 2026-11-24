"""Inspect report dependencies before writes; run from the repository root."""
import ast
import hashlib
import json
from pathlib import Path

from scripts.cifar10_explanation_reporting import Native32Report, RESULT

tree = ast.parse(Path('scripts/make_new_experiments_report.py').read_text(encoding='utf-8'))
paths = set()

def path_expr(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return Path(node.value)
    if isinstance(node, ast.Name) and node.id in ('R', 'DATA'):
        return {'R': Path('results'), 'DATA': Path('data/processed')}[node.id]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return path_expr(node.left) / path_expr(node.right)
    raise ValueError('dynamic path')

for node in ast.walk(tree):
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or not node.args:
        continue
    name = node.func.id
    try:
        value = path_expr(node.args[0])
    except ValueError:
        continue
    if name == 'rep_stage1':
        paths.update(Path('results') / value / n for n in ('stage1_detection.csv', 'stage1_noise.csv', 'evaluation.json'))
    elif name == 'expl_stage1':
        paths.update(value / n for n in ('stage1_detection.csv', 'stage1_noise.csv'))
    elif name == 'corpus_stats':
        paths.add(value / 'variants.csv')
    elif name == 's2rows':
        paths.add(Path('results') / value / 'combined_results.csv')
for directory in ('resnet_representation_cifar10_native32', 'dinov2_observer_cifar10_native32_stage2'):
    paths.update(Path('results') / directory / n for n in ('stage2_summary.csv', 'stage2_bootstrap.csv'))
paths.update(p for p in RESULT.rglob('*') if p.is_file() and p.suffix in ('.csv', '.json', '.xml')
             and p.name not in ('report_input_inventory.json', 'report_validation.json'))
missing = [p.as_posix() for p in sorted(paths) if not p.is_file()]
assert not missing, missing
report = Native32Report()
rendered = {name: len(getattr(report, name)()) for name in ('setup', 'explanation', 'representation', 'findings')}
record = {'status': 'passed', 'inputs': {p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)},
          'rendered_section_characters': rendered,
          'scope': 'Static generator input calls, corpus metadata, native result tables and validation artifacts checked before report writes'}
(RESULT / 'report_input_inventory.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print('Report preflight passed:', len(paths), 'input files;', rendered)
