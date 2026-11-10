#!/usr/bin/env python3
"""R26: verify explicit cohort bytes into a NEW evidence directory.

Only the recorded introducing commit SHA/path is queried. No current HEAD,
removal revision or fallback source is used. This script was not run in revision.
"""
import argparse
import hashlib
import io
import json
import re
import sys
import tokenize
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auxiliary_protocol import file_sha256, new_output_dir, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--timeout', type=int, default=15)
    args = parser.parse_args()
    cohort_hash = file_sha256(args.cohort)
    cohort = [json.loads(line) for line in args.cohort.read_text().splitlines() if line.strip()]
    ids = [row['sample_id'] for row in cohort]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Empty cohort or duplicate IDs')
    output = new_output_dir(args.output_dir)
    records = []
    with (output / 'manifest.jsonl').open('w', encoding='utf-8') as handle:
        for row in cohort:
            record = {'sample_id': row['sample_id'], 'introducing_url': row['introducing_url'],
                      'filename': row['filename'], 'local_snapshot_sha256': row['snapshot_sha256']}
            match = re.fullmatch(r'https://github\.com/([^/]+/[^/]+)/commit/([0-9a-fA-F]{40})/?', row['introducing_url'])
            if not match:
                record['status'] = 'invalid_introducing_url'
            else:
                repo, revision = match.groups()
                url = 'https://raw.githubusercontent.com/' + repo + '/' + revision + '/' + urllib.parse.quote(row['filename'], safe='/')
                record.update(requested_url=url, requested_revision=revision)
                try:
                    request = urllib.request.Request(url, headers={'User-Agent': 'ML-SATD-source-verification/2.0'})
                    with urllib.request.urlopen(request, timeout=args.timeout) as response:
                        raw = response.read(5_000_001)
                    if len(raw) > 5_000_000:
                        raise ValueError('Source exceeds size budget')
                    encoding = tokenize.detect_encoding(io.BytesIO(raw).readline)[0]
                    text = raw.decode(encoding).replace('\r\n', '\n')
                    name = hashlib.sha256(row['sample_id'].encode()).hexdigest()[:24] + '.source'
                    (output / name).write_bytes(raw)
                    record.update(status='fetched', encoding=encoding, saved_relative_path=name,
                                  raw_bytes_sha256=hashlib.sha256(raw).hexdigest(),
                                  historical_text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                                  full_snapshot_identical=hashlib.sha256(text.encode()).hexdigest() == row['snapshot_sha256'])
                except urllib.error.HTTPError as exc:
                    record.update(status='http_error', http_status=exc.code)
                except Exception as exc:
                    record.update(status='network_or_decode_error', error_type=type(exc).__name__)
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            handle.flush()
            records.append(record)
    unchanged = file_sha256(args.cohort) == cohort_hash
    write_json(output / 'summary.json', {'cohort_sha256': cohort_hash, 'cohort_unchanged': unchanged,
                                        'cohort_size': len(cohort), 'attempted': len(records),
                                        'manifest_sha256': file_sha256(output / 'manifest.jsonl'),
                                        'script_sha256': file_sha256(Path(__file__)),
                                        'full_snapshot_identical': sum(r.get('full_snapshot_identical') is True for r in records)})
    if not unchanged:
        raise ValueError('Cohort changed during verification; evidence must not be reused')


if __name__ == '__main__':
    main()
