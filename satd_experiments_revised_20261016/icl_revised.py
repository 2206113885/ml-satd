"""R38-R44: independent prospective ICL experiment, never an automatic repair of old results.

Requires separate training/test CSVs and a clone-group provenance manifest. Old
satd_type, fold values and fixed examples are never imported. Retrieval is a NEW
TF-IDF protocol; only generation_tokenizer exists, eliminating tokenizer reuse.
"""
import argparse
import json
from pathlib import Path

from auxiliary_protocol import (ABBREVIATIONS, file_sha256, load_labeled, new_output_dir,
                                normalize_text, stable_sample_seed, write_json)
from qwen_auxiliary_runtime import LABEL_DEFINITIONS, QwenRuntime, exact_abbreviation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-csv', required=True, type=Path)
    parser.add_argument('--test-csv', required=True, type=Path)
    parser.add_argument('--clone-provenance', required=True, type=Path,
                        help='JSON with train_sha256, test_sha256 and nonempty method; externally adjudicated clone groups required')
    parser.add_argument('--clone-column', default='Clone Group')
    parser.add_argument('--id-column', default='Dataset ID')
    parser.add_argument('--text-column', default='Comment Text')
    parser.add_argument('--gold-column', default='ML TD Type', help='Use ML labels, never traditional satd_type')
    parser.add_argument('--project-column', default='Repo Name')
    parser.add_argument('--model-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--shots', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20261016)
    parser.add_argument('--input-cap', type=int, default=4096)
    parser.add_argument('--max-new-tokens', type=int, default=32)
    args = parser.parse_args()
    if args.gold_column.strip().casefold() == 'satd_type':
        raise ValueError('Traditional SATD gold is prohibited; provide a validated ML-SATD gold column')
    hashes = {'train_sha256': file_sha256(args.train_csv), 'test_sha256': file_sha256(args.test_csv)}
    provenance = json.loads(args.clone_provenance.read_text())
    if not str(provenance.get('method', '')).strip() or any(provenance.get(k) != v for k, v in hashes.items()):
        raise ValueError('Clone provenance missing or bound to different train/test bytes; supply independently adjudicated data first')
    columns = {'id_column': args.id_column, 'text_column': args.text_column,
               'gold_column': args.gold_column, 'project_column': args.project_column,
               'clone_column': args.clone_column}
    train, test = load_labeled(args.train_csv, **columns), load_labeled(args.test_csv, **columns)
    # ID, normalized comment, clone and project isolation are mandatory for this new cross-project protocol.
    for name, key in [('sample_id', lambda r: r['sample_id']),
                      ('normalized_comment', lambda r: normalize_text(r['comment'])),
                      ('project', lambda r: normalize_text(r['project'])),
                      ('clone_group', lambda r: r['clone_group'])]:
        if {key(r) for r in train} & {key(r) for r in test}:
            raise ValueError('Train/test leakage in ' + name + '; old fold values cannot bypass this check')
    if args.shots < 1 or args.shots > len(train):
        raise ValueError('shots must be between 1 and the number of independently supplied training rows')
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b')
    matrix = vectorizer.fit_transform([r['comment'] for r in train])
    output = new_output_dir(args.output_dir)
    runtime = QwenRuntime(args.model_dir, args.input_cap)
    write_json(output / 'settings.json', {
        'protocol': 'prospective ICL v1; TF-IDF on supplied training comments only',
        **hashes, 'clone_provenance_sha256': file_sha256(args.clone_provenance),
        'clone_provenance_external_claim': provenance, 'clone_detection_not_independently_revalidated': True,
        'columns': columns, 'shots': args.shots, 'seed': args.seed,
        'seed_policy': 'SHA256([seed, sample_id, repetition]); per-target independent sampling',
        'retrieval_tie_policy': 'descending similarity then stable training ID',
        'no_legacy_folds_or_fixed_examples': True,
        'model_identity': runtime.identity, 'loaded_generation_config': runtime.loaded_generation_config,
        'effective_generation_config': runtime.generation_config(args.max_new_tokens).to_dict(),
        'runner_sha256': file_sha256(Path(__file__)),
        'runtime_sha256': file_sha256(Path(__file__).with_name('qwen_auxiliary_runtime.py')),
        'helper_sha256': file_sha256(Path(__file__).with_name('auxiliary_protocol.py')),
        'failure_policy': 'strict failures count wrong; legacy Data Conf comparison is separate',
        'historical_R44_example_generator_provenance': 'UNRESOLVED; no GPT-5 seed/source history is inferred',
    })
    records = []
    with (output / 'prompts.jsonl').open('w', encoding='utf-8') as prompts, (output / 'predictions.jsonl').open('w', encoding='utf-8') as predictions:
        for row in test:
            scores = cosine_similarity(vectorizer.transform([row['comment']]), matrix)[0]
            order = sorted(range(len(train)), key=lambda index: (-float(scores[index]), train[index]['sample_id']))[:args.shots]
            examples = [train[index] for index in order]
            example_text = '\n\n'.join('Training comment: ' + json.dumps(e['comment'], ensure_ascii=False) + '\nLabel: ' + e['gold_label'] for e in examples)
            prompt = (LABEL_DEFINITIONS + '\n\nIndependently supplied labeled training examples:\n' + example_text +
                      '\n\nTarget comment:\n' + json.dumps(row['comment'], ensure_ascii=False) +
                      '\nReturn exactly ONE abbreviation and no other text. Allowed: ' + ', '.join(ABBREVIATIONS))
            seed = stable_sample_seed(args.seed, row['sample_id'])
            result = {**row, 'example_ids': [e['sample_id'] for e in examples], 'seed': seed}
            prompts.write(json.dumps({'sample_id': row['sample_id'], 'prompt': prompt, 'example_ids': result['example_ids']}, ensure_ascii=False) + '\n')
            try:
                generation = runtime.generate(prompt, args.max_new_tokens, seed)
                prediction = exact_abbreviation(generation['raw_response'])
                result.update(generation, predicted_label=prediction, parse_ok=prediction is not None,
                              fallback_reason='' if prediction is not None else 'unrecognized_output')
            except ValueError as exc:
                if not str(exc).startswith('input_budget_exceeded'):
                    raise
                # [R24 原代码保留]
                # result.update(predicted_label=None, parse_ok=False, raw_response='', fallback_reason='input_budget_exceeded')
                # [R24 修订]
                result.update(predicted_label=None, parse_ok=False, raw_response='', fallback_reason='input_budget_exceeded',
                              input_failure_detail=str(exc))
            result['legacy_policy_prediction'] = result['predicted_label'] or 'Data Conf'
            predictions.write(json.dumps(result, ensure_ascii=False) + '\n')
            predictions.flush()
            records.append(result)
    count = len(records)
    write_json(output / 'summary.json', {
        'n': count, 'strict_accuracy': sum(r['predicted_label'] == r['gold_label'] for r in records) / count,
        'legacy_fallback_accuracy': sum(r['legacy_policy_prediction'] == r['gold_label'] for r in records) / count,
        'format_or_input_failures': sum(not r['parse_ok'] for r in records),
        'interpretation': 'New independent protocol only. Not comparable to old ICL results without a separate controlled experiment.',
    })


if __name__ == '__main__':
    main()
