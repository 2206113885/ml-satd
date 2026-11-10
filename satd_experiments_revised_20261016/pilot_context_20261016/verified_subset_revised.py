#!/usr/bin/env python3
"""R26: derive a NEW history-verified subset only after source-bound checks.

No network requests or inference. A stale summary, record, URL, context or local
evidence file causes rejection. The historical artifacts are read-only inputs.
"""
import argparse
import ast
import hashlib
import io
import json
import sys
import tokenize
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auxiliary_protocol import ABBREVIATIONS, file_sha256, new_output_dir, write_json


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


# [R26 修订] 只静态读取受支持的reference表达式；绝不import/exec历史runner或reference。
SUPPORTED_REFERENCE_SHA256 = {'b94cbdf8684412cd86b0411b9a9cabf1ae24197ec36cbb066cd39e0c2f5cb256': 'historical_frozen_v1', 'de362753003711370814da7f42a3df71c1bdbc81c70557e5011f88d14909e284': 'revised_json_v1'}
SUPPORTED_RUNNER_FUNCTION_AST_SHA256 = {'sha256': 'deacf46d1d88f6e9cb8a0074b87caf9996e28ed577fb69f2baa6216be7873103', 'prompt_for': '924fe4eb74f0b80a033c360152dc0cc1da2a55bad1caa5c5f4f9bc3322e884b0', 'row_seed': 'bd995dab730fda323aad273c1d238fcdc463e49689187f006646a7f31a49cfd2'}


def read_reference_string(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'strip' and not node.args and not node.keywords:
        return read_reference_string(node.func.value).strip()
    if isinstance(node, ast.JoinedStr):
        pieces = []
        expected_join = ast.dump(ast.parse("', '.join(sorted(ABBR_SET))", mode='eval').body, include_attributes=False)
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                pieces.append(value.value)
            elif isinstance(value, ast.FormattedValue) and value.conversion == -1 and value.format_spec is None and ast.dump(value.value, include_attributes=False) == expected_join:
                pieces.append(', '.join(ABBREVIATIONS))
            else:
                raise ValueError('Unsupported reference string expression; do not infer missing provenance')
        return ''.join(pieces)
    raise ValueError('Unsupported reference constant; no arbitrary source execution is allowed')


def expected_prompt(row, arm, reference_source, mode):
    tree = ast.parse(reference_source)
    constants = {target.id: node.value for node in tree.body if isinstance(node, ast.Assign)
                 for target in node.targets if isinstance(target, ast.Name)}
    head = read_reference_string(constants['PROMPT_HEAD'])
    if mode == 'historical_frozen_v1':
        tail = read_reference_string(constants['PROMPT_TAIL'])
        prompt = head + '\n\n' + tail + '\n\nCode Comment:' + row['comment'] + '\n"""\n\nAbbreviation:'
    elif mode == 'revised_json_v1':
        prompt = (head + '\n\nClassify the target ML-SATD comment. Return exactly ONE abbreviation: ' +
                  ', '.join(ABBREVIATIONS) + '\n\nCode Comment:\n' + json.dumps(row['comment'], ensure_ascii=False) + '\n\nAbbreviation:')
    else:
        raise ValueError('Unknown reference protocol')
    if arm == 'comment_only':
        return prompt
    suffix = '\n\nAbbreviation:'
    if not prompt.endswith(suffix):
        raise ValueError('Unsupported context insertion protocol')
    context = row['window_context' if arm == 'window' else 'function_context']
    return (prompt[:-len(suffix)] + '\n\nSource code context for the target comment above:\n```python\n' +
            context + '\n```\nUse this code only as evidence to disambiguate the target comment. '
            'Classify that comment, not other comments in the code.\n\nAbbreviation:')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', required=True, type=Path)
    parser.add_argument('--history-manifest', required=True, type=Path)
    parser.add_argument('--history-summary', required=True, type=Path)
    parser.add_argument('--evidence-root', required=True, type=Path,
                        help='Root against which saved_relative_path is resolved; explicit for old/new manifest conventions')
    parser.add_argument('--predictions', required=True, type=Path)
    # [R26 原代码保留]
    # parser.add_argument('--settings', required=True, type=Path)
    # [R26 修订]
    parser.add_argument('--settings', required=True, type=Path)
    parser.add_argument('--prompts', required=True, type=Path, help='Actual parent-run prompts.jsonl; reconstructed labels alone are insufficient')
    parser.add_argument('--runner-source', required=True, type=Path, help='Exact saved source snapshot whose hash matches settings.runner_sha256')
    parser.add_argument('--reference-source', required=True, type=Path, help='Exact supported reference snapshot matching settings.reference_prompt_sha256')
    parser.add_argument('--helper-source', type=Path, help='Required for revised settings that record helper_sha256; must match the actually imported helper')
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    # [R26 原代码保留]
    # initial_hashes = {str(p.resolve()): file_sha256(p) for p in
    #                   [args.cohort, args.history_manifest, args.history_summary, args.predictions, args.settings]}
    # [R26 修订]
    parent_paths = [args.cohort, args.history_manifest, args.history_summary, args.predictions,
                    args.settings, args.prompts, args.runner_source, args.reference_source]
    if args.helper_source is not None:
        parent_paths.append(args.helper_source)
    initial_hashes = {str(p.resolve()): file_sha256(p) for p in parent_paths}
    cohort_hash = initial_hashes[str(args.cohort.resolve())]
    summary = json.loads(args.history_summary.read_text())
    settings = json.loads(args.settings.read_text())
    if summary.get('cohort_sha256') != cohort_hash or summary.get('cohort_unchanged') is not True:
        raise ValueError('History summary does not verify the current cohort bytes; re-verify this cohort')
    # [R26 原代码保留]
    # if settings.get('cohort_sha256') != cohort_hash:
    # [R26 修订]
    if summary.get('manifest_sha256') is not None and summary['manifest_sha256'] != initial_hashes[str(args.history_manifest.resolve())]:
        raise ValueError('History summary checksum does not bind this manifest')
    if settings.get('cohort_sha256') != cohort_hash:
        raise ValueError('Prediction settings belong to a different cohort')
    # [R26 原代码保留]
    # cohort = read_jsonl(args.cohort)
    # [R26 修订]
    if settings.get('runner_sha256') != initial_hashes[str(args.runner_source.resolve())]:
        raise ValueError('Exact parent runner source snapshot is absent or different')
    reference_hash = initial_hashes[str(args.reference_source.resolve())]
    if settings.get('reference_prompt_sha256') != reference_hash or reference_hash not in SUPPORTED_REFERENCE_SHA256:
        raise ValueError('Parent reference snapshot is absent, mismatched or unsupported; do not infer its protocol')
    reference_mode = SUPPORTED_REFERENCE_SHA256[reference_hash]
    runner_tree = ast.parse(args.runner_source.read_text())
    observed_functions = {node.name: hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
                          for node in runner_tree.body if isinstance(node, ast.FunctionDef)
                          and node.name in SUPPORTED_RUNNER_FUNCTION_AST_SHA256}
    if observed_functions != SUPPORTED_RUNNER_FUNCTION_AST_SHA256:
        raise ValueError('Unsupported parent prompt/seed construction protocol; historical source is never executed')
    helper_expected = settings.get('helper_sha256')
    if helper_expected is not None:
        current_helper = Path(__file__).resolve().parents[1] / 'auxiliary_protocol.py'
        if args.helper_source is None or args.helper_source.resolve() != current_helper.resolve() or file_sha256(current_helper) != helper_expected:
            raise ValueError('Revised parent helper snapshot does not match the actually used protocol module')
    if settings.get('arms') != ['comment_only', 'window', 'function'] or settings.get('thinking') is not False:
        raise ValueError('Unsupported parent arm/thinking protocol')
    if type(settings.get('seed')) is not int or settings.get('seed_policy') != 'SHA256(seed:sample_id), same seed across arms':
        raise ValueError('Parent seed provenance is missing or unsupported')
    reference_source = args.reference_source.read_text()
    cohort = read_jsonl(args.cohort)
    ids = [r['sample_id'] for r in cohort]
    if len(set(ids)) != len(ids) or not ids or summary.get('cohort_size') != len(cohort):
        raise ValueError('Invalid cohort IDs/size')
    # [R26 原代码保留]
    # by_id = {r['sample_id']: r for r in cohort}
    # [R26 修订]
    if settings.get('selected_ids') != ids:
        raise ValueError('Settings selected_ids do not exactly bind the current complete cohort')
    by_id = {r['sample_id']: r for r in cohort}
    history = read_jsonl(args.history_manifest)
    history_ids = [r['sample_id'] for r in history]
    if len(set(history_ids)) != len(history_ids) or set(history_ids) != set(ids):
        raise ValueError('History manifest must account for each current cohort ID exactly once')
    selected_ids = set()
    evidence_root = args.evidence_root.resolve()
    evidence_hashes = {}
    for record in history:
        row = by_id[record['sample_id']]
        if (record.get('local_snapshot_sha256') != row['snapshot_sha256'] or
                record.get('introducing_url') != row['introducing_url'] or
                record.get('filename') != row['filename']):
            raise ValueError('Stale historical snapshot/URL/path for ' + row['sample_id'])
        if record.get('full_snapshot_identical') is not True:
            continue
        if record.get('status') != 'fetched':
            raise ValueError('Only fetched evidence can be verified')
        evidence = (evidence_root / record['saved_relative_path']).resolve()
        if not evidence.is_relative_to(evidence_root):
            raise ValueError('Evidence path escapes its declared root')
        raw = evidence.read_bytes()
        # [R26 原代码保留]
        # import hashlib
        # [R26 修订]
        # hashlib is imported at module scope; no source code is executed.
        digest = hashlib.sha256(raw).hexdigest()
        if digest != record.get('raw_bytes_sha256'):
            raise ValueError('Historical evidence bytes changed for ' + row['sample_id'])
        encoding = record.get('encoding') or tokenize.detect_encoding(io.BytesIO(raw).readline)[0]
        text = raw.decode(encoding).replace('\r\n', '\n')
        if hashlib.sha256(text.encode()).hexdigest() != row['snapshot_sha256']:
            raise ValueError('Historical full snapshot does not match current cohort')
        if row['function_context'] not in text or row['window_context'] not in text:
            raise ValueError('Current supplied contexts are absent from the verified snapshot')
        lines = text.splitlines()
        target = row['target_line_1based']
        if target < 1 or target > len(lines) or lines[target-1] != row['target_source_line']:
            raise ValueError('Current target line is absent from the verified snapshot')
        selected_ids.add(row['sample_id'])
        evidence_hashes[row['sample_id']] = digest
    if not selected_ids:
        raise ValueError('No verified snapshots; no subset written')
    predictions = read_jsonl(args.predictions)
    expected = {(sid, arm) for sid in ids for arm in ('comment_only', 'window', 'function')}
    keys = [(r['sample_id'], r['arm']) for r in predictions]
    if len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError('Predictions must contain each cohort/arm exactly once')
    # [R26 原代码保留]
    # if any(r['gold_label'] != by_id[r['sample_id']]['gold_label'] for r in predictions):
    #     raise ValueError('Prediction/cohort gold mismatch')
    # [R26 修订]
    if any(r['gold_label'] != by_id[r['sample_id']]['gold_label'] for r in predictions):
        raise ValueError('Prediction/cohort gold mismatch')
    prompts = read_jsonl(args.prompts)
    prompt_keys = [(r['sample_id'], r['arm']) for r in prompts]
    if len(set(prompt_keys)) != len(prompt_keys) or set(prompt_keys) != expected:
        raise ValueError('Parent prompt log must contain every sample/arm exactly once')
    prompts_by_key = {(r['sample_id'], r['arm']): r for r in prompts}
    for prediction in predictions:
        key = (prediction['sample_id'], prediction['arm'])
        logged = prompts_by_key[key]
        actual_prompt = logged.get('prompt')
        wanted = expected_prompt(by_id[key[0]], key[1], reference_source, reference_mode)
        if not isinstance(actual_prompt, str) or actual_prompt != wanted:
            raise ValueError('Actual parent prompt does not match source/cohort/arm protocol: ' + str(key))
        if prediction.get('prompt_sha256') != hashlib.sha256(actual_prompt.encode()).hexdigest():
            raise ValueError('Prediction is not bound to its actual parent prompt log: ' + str(key))
        expected_seed = int(hashlib.sha256((str(settings['seed']) + ':' + key[0]).encode()).hexdigest()[:8], 16)
        if prediction.get('seed') != expected_seed:
            raise ValueError('Prediction seed does not match parent protocol: ' + str(key))
        if logged.get('prompt_tokens') != prediction.get('prompt_tokens'):
            raise ValueError('Prediction/prompt token-count metadata mismatch: ' + str(key))
    if any(file_sha256(path) != digest for path, digest in initial_hashes.items()):
        raise ValueError('A parent artifact changed while deriving the verified subset')
    output = new_output_dir(args.output_dir)
    for name, rows in [('cohort.jsonl', [r for r in cohort if r['sample_id'] in selected_ids]),
                       ('predictions.jsonl', [r for r in predictions if r['sample_id'] in selected_ids])]:
        (output / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    write_json(output / 'lineage.json', {
        'parent_inputs_sha256': initial_hashes, 'cohort_sha256': cohort_hash,
        'selected_ids': sorted(selected_ids), 'evidence_sha256': evidence_hashes,
        'subset_cohort_sha256': file_sha256(output / 'cohort.jsonl'),
        'subset_predictions_sha256': file_sha256(output / 'predictions.jsonl'),
        # [R26 原代码保留]
        # 'script_sha256': file_sha256(Path(__file__)),
        # [R26 修订]
        'script_sha256': file_sha256(Path(__file__)),
        'reference_protocol': reference_mode,
        'parent_prompt_binding': 'Actual prompt bytes, prediction hashes, supported source snapshots, seed, arms and selected IDs verified without executing historical sources',
        'boundary': 'Existing records are reused only after current cohort and actual evidence-byte verification; no historical source/version claim is manufactured',
    })


if __name__ == '__main__':
    main()
