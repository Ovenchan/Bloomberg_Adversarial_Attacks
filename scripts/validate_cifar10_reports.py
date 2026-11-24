"""Regenerate and verify native-32 reports, links and preserved result files.

Run after scripts.check_cifar10_report_inputs, from the repository root.
"""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

repo = Path.cwd()
result = repo / 'results/slice_belief_cifar10_native32_fixed1000'
docs = [repo / p for p in ('NEW_EXPERIMENTS_REPORT.md', 'COMBINED_EXPERIMENT_REPORT.md',
                            'results/slice_belief_cifar10_native32_fixed1000/MAIN_RESULTS.md')]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

before = {p.relative_to(repo).as_posix(): sha(p) for p in docs + [repo / 'results/corpus_summary_imagenette_cifar10.csv']}
subprocess.run([sys.executable, '-m', 'scripts.cifar10_explanation_reporting'], check=True)
subprocess.run([sys.executable, 'scripts/make_new_experiments_report.py', 'data/processed'], check=True)
after = {p: sha(repo / p) for p in before}
assert before == after, {'before': before, 'after': after}

links = []
for document in docs + [result / 'EXPERIMENT.md', repo / 'README.md']:
    for match in re.finditer(r'\[[^\]]+\]\(([^)]+)\)', document.read_text(encoding='utf-8')):
        target = match.group(1).strip('<>')
        if target.startswith(('http:', 'https:', 'mailto:', '#')):
            continue
        target = unquote(target.split('#', 1)[0])
        assert not Path(target).is_absolute(), (document, target)
        resolved = (document.parent / target).resolve()
        assert resolved.is_relative_to(repo) and resolved.exists(), (document, target)
        links.append({'document': document.relative_to(repo).as_posix(), 'target': target})

combined = (repo / 'COMBINED_EXPERIMENT_REPORT.md').read_text(encoding='utf-8')
assert re.findall(r'^## (.+)$', combined, re.M) == [
    '1. Overall setup', '2. Explanation-based pipeline', '3. Representation-based pipeline']
for name in ('setup', 'explanation', 'representation'):
    assert combined.count(f'<!-- cifar10-native32-{name}:start -->') == 1
    assert combined.count(f'<!-- cifar10-native32-{name}:end -->') == 1
assert 'have not yet been run on CIFAR-10' not in docs[0].read_text(encoding='utf-8')

table_checks = {}
for document in docs[:2]:
    old = subprocess.run(['git', 'show', f'HEAD:{document.name}'], capture_output=True, check=True).stdout.decode('utf-8')
    new = document.read_text(encoding='utf-8')
    parse = lambda text: [[c.strip() for c in line.strip().strip('|').split('|')]
                          for line in text.splitlines() if line.startswith('|') and not re.match(r'^\|[\s:|\-]+$', line)]
    previous, current = parse(old), parse(new)
    # The native CW representation table adds FPR columns; original cells remain a prefix.
    missing = [r for r in previous if not any(n[:len(r)] == r for n in current)]
    assert not missing, (document, missing)
    table_checks[document.name] = len(previous)

preserved = json.loads((result / 'preserved_results_sha256.json').read_text())
for name, expected in preserved.items():
    assert sha(repo / name) == expected, name
inputs = json.loads((result / 'report_input_inventory.json').read_text())['inputs']
for name, expected in inputs.items():
    assert sha(repo / name) == expected, name

proof = {'status': 'passed', 'idempotent_outputs': after, 'local_links_checked': len(links),
         'combined_main_sections': 3, 'preserved_original_table_rows': table_checks,
         'preserved_result_files': len(preserved), 'unchanged_report_inputs': len(inputs),
         'scope': 'Report regeneration, repository-relative links, original table cells, three-section structure and input/result preservation'}
(result / 'report_validation.json').write_text(json.dumps(proof, indent=2) + '\n', encoding='utf-8', newline='\n')
print(json.dumps(proof, indent=2))
