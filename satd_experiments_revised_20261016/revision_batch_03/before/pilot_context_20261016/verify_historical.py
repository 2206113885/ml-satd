#!/usr/bin/env python3
"""Verify frozen cohort snapshots using only their introducing commit URLs.

This is independent of inference. It never rewrites the cohort, its gold labels,
or model-facing contexts. Downloaded historical sources are evidence only.
"""

import argparse
import concurrent.futures
import hashlib
import io
import json
import re
import threading
import tokenize
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path


BASE = Path(__file__).resolve().parent
OUT = BASE / "data" / "historical_verification"
OUT.mkdir(exist_ok=True)
COHORT = BASE / "data" / "cohort.jsonl"
TIMEOUT_SECONDS = 15
WORKERS = 4
MAX_BYTES = 5_000_000


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(text):
    text = text.replace("\\/", "/").replace('\\"', '"')
    text = text.replace(" || ", "\n").replace(",", ";")
    text = re.sub(r"(?m)^\s*#+\s?", "", text)
    return re.sub(r"\s+", " ", text).strip()


def fetch(row):
    result = {"sample_id": row["sample_id"], "introducing_url": row["introducing_url"], "filename": row["filename"], "local_snapshot_sha256": row["snapshot_sha256"]}
    match = re.fullmatch(r"https://github\.com/([^/]+/[^/]+)/commit/([0-9a-fA-F]{40})/?", row["introducing_url"])
    if not match:
        result["status"] = "invalid_introducing_url"
        return result
    revision_repo, revision = match.groups()
    url = "https://raw.githubusercontent.com/" + revision_repo + "/" + revision + "/" + urllib.parse.quote(row["filename"], safe="/")
    result.update({"requested_url": url, "requested_revision": revision})
    relative = Path("data") / "historical_verification" / (row["sample_id"] + "__" + revision[:12] + ".source")
    path = BASE / relative
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "ML-SATD-research-context-verification/1.0"})
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            raw = response.read(MAX_BYTES + 1)
            result.update({"http_status": response.status, "final_url": response.url})
        if len(raw) > MAX_BYTES:
            result["status"] = "source_too_large"
            return result
        path.write_bytes(raw)
        encoding = "utf-8"
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
            text = raw.decode(encoding)
        text = text.replace("\r\n", "\n")
        lines = text.splitlines()
        historical_sha = sha256(text.encode("utf-8"))
        target = row["target_line_1based"]
        target_line_same = target <= len(lines) and lines[target - 1] == row["target_source_line"]
        exact_function = row["function_context"] in text
        exact_window = row["window_context"] in text
        normalized_comment = canonical(row["comment"]) in canonical(text)
        target_lines = [i + 1 for i, line in enumerate(lines) if canonical(row["target_source_line"]) in canonical(line)]
        result.update({
            "status": "fetched", "saved_relative_path": str(relative), "encoding": encoding,
            "bytes": len(raw), "raw_bytes_sha256": sha256(raw),
            "historical_text_sha256": historical_sha,
            "full_snapshot_identical": historical_sha == row["snapshot_sha256"],
            "target_line_identical_at_original_position": target_line_same,
            "exact_local_function_present": exact_function,
            "exact_local_window_present": exact_window,
            "serialized_comment_present_after_matching_normalization": normalized_comment,
            "matching_target_line_positions_1based": target_lines,
        })
    except urllib.error.HTTPError as exc:
        result.update({"status": "http_error", "http_status": exc.code})
    except Exception as exc:
        result.update({"status": "network_or_decode_error", "error_type": type(exc).__name__, "error": str(exc)[:250]})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retry-network-errors", action="store_true", help="Retry only transient failures recorded in the existing manifest, once")
    args = parser.parse_args()
    initial_hash = sha256(COHORT.read_bytes())
    cohort = [json.loads(line) for line in COHORT.read_text().splitlines()]
    prior_records = []
    if args.retry_network_errors:
        prior_records = [json.loads(line) for line in (OUT / "manifest.jsonl").read_text().splitlines()]
        retry_ids = {r["sample_id"] for r in prior_records if r["status"] == "network_or_decode_error"}
        requested = [r for r in cohort if r["sample_id"] in retry_ids]
        (OUT / "manifest_first_attempt.jsonl").write_text((OUT / "manifest.jsonl").read_text(), encoding="utf-8")
    else:
        requested = cohort
    records = []
    stopped = False
    # Four-request batches permit an error-rate stop without leaving a large
    # queue of requests in flight. HTTP 404 is a provenance finding, not a rate
    # limit; stop only for repeated network/rate-limit/server failures.
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
        for start in range(0, len(requested), WORKERS):
            batch = list(executor.map(fetch, requested[start:start + WORKERS]))
            records.extend(batch)
            errors = sum(r["status"] == "network_or_decode_error" or r.get("http_status") in (403, 429, 500, 502, 503, 504) for r in records)
            counts = Counter(r["status"] for r in records)
            print(json.dumps({"attempted": len(records), "status_counts": dict(counts)}, ensure_ascii=False), flush=True)
            if errors >= 3 and errors / len(records) >= .20:
                stopped = True
                break
    requests_this_run = len(records)
    if prior_records:
        updated = {r["sample_id"]: r for r in prior_records}
        for record in records:
            updated[record["sample_id"]] = record
        records = list(updated.values())
    records.sort(key=lambda r: r["sample_id"])
    with (OUT / "manifest.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    fetched = [r for r in records if r["status"] == "fetched"]
    summary = {
        "cohort_sha256": initial_hash, "cohort_unchanged": initial_hash == sha256(COHORT.read_bytes()),
        "cohort_size": len(cohort), "attempted": len(records), "workers": WORKERS,
        "requests_this_run": requests_this_run, "retry_network_errors_only": args.retry_network_errors,
        "timeout_seconds": TIMEOUT_SECONDS, "stopped_due_to_network_rate_errors": stopped,
        "status_counts": dict(Counter(r["status"] for r in records)),
        "http_error_counts": dict(Counter(str(r["http_status"]) for r in records if r["status"] == "http_error")),
        "fetched": len(fetched),
        "full_snapshot_identical": sum(r["full_snapshot_identical"] for r in fetched),
        "target_line_identical_at_original_position": sum(r["target_line_identical_at_original_position"] for r in fetched),
        "exact_local_function_present": sum(r["exact_local_function_present"] for r in fetched),
        "exact_local_window_present": sum(r["exact_local_window_present"] for r in fetched),
        "serialized_comment_present_after_matching_normalization": sum(r["serialized_comment_present_after_matching_normalization"] for r in fetched),
        "scope": "Only recorded introducing-commit SHA and recorded filename; no removal revision or current HEAD fallback; frozen cohort is never changed.",
        "interpretation": "Identical full snapshot or exact supplied function/window establishes that that input existed at the recorded introducing revision. A file fetch alone, or target-comment presence alone, does not validate all supplied context historically.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
