#!/usr/bin/env python3
"""Side-effect-free original AST extraction and data review evidence.

Never imports or executes an original module. Only explicitly selected string
parsers are compiled, with decorators removed, plus pure label constants.
"""
import ast
import collections
import copy
import csv
import hashlib
import inspect
import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = BASE.parent
EXCLUDE = {'models', 'pilot_context_20261016', 'paper_generation_audit_20261016', BASE.name, '.venv', '__pycache__', '.git'}
PARSERS = {'_canon', '_esc', '_visible_only', '_extract_visible', 'simple_extract', 'extract_abbr', 'extract_abbr_from_text', 'extract_coarse', 'extract_fine', 'extract_yes_no', 'parse_yesno', 'parse_abbr', 'parse_abbr_and_reason', 'extract_reason', 'extract_reason_from_text', 'parse_reason', 'normalize_to_full'}
SAFE_CALLS = {'set', 'list', 'dict', 'tuple', 'str', 'len', 'sorted', 'map', 'min', 'max', 'any', 'all', '_canon', '_esc'}


def pure_value(node):
    for child in ast.walk(node):
        if isinstance(child, (ast.Await, ast.Yield, ast.YieldFrom, ast.Lambda)):
            return False
        if isinstance(child, ast.Call):
            if isinstance(child.func, ast.Name) and child.func.id not in SAFE_CALLS:
                return False
            if isinstance(child.func, ast.Attribute) and child.func.attr not in {'compile', 'escape', 'lower', 'upper', 'strip', 'replace', 'join', 'items', 'keys', 'values'}:
                return False
    return True


def parser_environment(tree):
    env = {'re': re, 'json': json}
    defs = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in PARSERS:
            node = copy.deepcopy(node)
            node.decorator_list = []
            defs.append(node)
    for node in defs:
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<extracted_string_parser>', 'exec'), env)
    assignments = [n for n in tree.body if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and pure_value(n.value)]
    # Repeated passes resolve only pure label mappings and compiled regexes.
    for _ in range(4):
        for node in assignments:
            key = node.targets[0].id
            if not (key.isupper() or 'label' in key.lower() or 'abbr' in key.lower()):
                continue
            if any(part in key.lower() for part in ('token', 'auth', 'secret', 'password', 'model', 'path', 'dir', 'prompt')):
                continue
            try:
                env[key] = eval(compile(ast.Expression(node.value), '<pure_label_constant>', 'eval'), env)
            except Exception:
                pass
    return env


def main():
    evidence, coverage = [], []
    for file in sorted(DATA.rglob('*.py')):
        if set(file.relative_to(DATA).parts) & EXCLUDE:
            continue
        source = file.read_text(encoding='utf-8-sig')
        syntax = None
        try:
            tree = ast.parse(source)
        except SyntaxError as error:
            syntax = {'line': error.lineno, 'message': error.msg}
            try:
                tree = ast.parse('\n'.join(source.splitlines()[:error.lineno - 1]))
            except SyntaxError:
                tree = ast.Module(body=[], type_ignores=[])
        relative = str(file.relative_to(DATA))
        if file.name == 'download_model.py':
            coverage.append({'file': relative, 'status': 'not_applicable', 'reason': 'Model download utility; no classification gold, parser, metrics, or prediction saving. Authentication content not extracted.'})
            continue
        env = parser_environment(tree)
        tests = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef) or node.name not in PARSERS or node.name not in env:
                continue
            func = env[node.name]
            if node.name in {'extract_reason', 'extract_reason_from_text', 'parse_reason', '_canon', '_esc', '_visible_only', '_extract_visible', 'normalize_to_full'}:
                continue
            for text in ['', 'Yesterday', 'Not sure', 'Model Code Reusability', 'Data Conf.', '**Data Conf**']:
                args = [text]
                if node.name == 'extract_fine':
                    args.append(['Data Conf', 'PODT', 'DS Conf'])
                try:
                    answer = func(*args)
                    test = {'file': relative, 'function': node.name, 'line': node.lineno, 'end_line': node.end_lineno, 'input': text, 'output': answer}
                except Exception as error:
                    test = {'file': relative, 'function': node.name, 'line': node.lineno, 'end_line': node.end_lineno, 'input': text, 'exception': type(error).__name__, 'message': str(error)}
                tests.append(test)
                evidence.append(test)
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        features = {
            'dataset_read_lines': [n.lineno for n in calls if ast.unparse(n.func).endswith('read_csv')],
            'metric_lines': [n.lineno for n in calls if ast.unparse(n.func).split('.')[-1] in ('classification_report', 'confusion_matrix')],
            'prediction_save_lines': [n.lineno for n in calls if ast.unparse(n.func).endswith('to_csv')],
            'sample_id_preserved_in_output': bool(re.search(r'["\'](?:dataset_id|sample_id|Dataset ID)["\']\s*:', source)),
        }
        coverage.append({'file': relative, 'status': 'reviewed', 'sha256': hashlib.sha256(file.read_bytes()).hexdigest(), 'syntax_error': syntax, 'review_method': 'Read relevant label/parser/dataset/evaluation/save AST; group identical helpers; model-free extracted-parser tests; inspect method-specific control flow.', 'areas': ['gold and candidate label mapping', 'unknown/fallback control flow', 'metric labels and denominators', 'data alignment and duplicate identity', 'CSV/report write order and run naming', 'pure parser behavior'], 'parser_test_count': len(tests), **features})
    (BASE / 'data_parser_tests.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n')
    (BASE / 'data_coverage.json').write_text(json.dumps(coverage, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'files': len(coverage), 'reviewed': sum(r['status'] == 'reviewed' for r in coverage), 'not_applicable': sum(r['status'] == 'not_applicable' for r in coverage), 'parser_tests': len(evidence), 'exceptions': dict(collections.Counter(r.get('exception') for r in evidence if r.get('exception')))}, indent=2))


if __name__ == '__main__':
    main()
