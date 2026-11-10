#!/usr/bin/env python3
"""Build a deterministic, aligned three-arm cohort without changing source data.

The CSV serializes commas as semicolons, newlines as `` || ``, and some
quotes/slashes with backslashes. These transformations are used for matching
only. Model-facing comments and code retain their respective original strings.
No commit messages, removal revisions, labels, or project names enter arm text.
"""

import argparse
import ast
import csv
import hashlib
import io
import json
import re
import textwrap
import tokenize
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path


BASE = Path(__file__).resolve().parent
DATASET = BASE.parent / "dataset"
FINAL = DATASET / "sampled_label_dataset_normalized_rmvtodo.csv"
CONTEXT = DATASET / "OBrien_789_v2.csv"
SELECTION_SEED = "ml-satd-context-pilot-20261016-v1"


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical(text):
    """Only undo known serialization differences and normalize whitespace."""
    text = text.replace("\\/", "/").replace('\\"', '"')
    text = text.replace(" || ", "\n").replace(",", ";")
    text = re.sub(r"(?m)^\s*#+\s?", "", text)
    return re.sub(r"\s+", " ", text).strip()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def ast_functions(source):
    """Return function spans from a complete parsable source snapshot."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return None
    result = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            first = min([node.lineno] + [d.lineno for d in node.decorator_list])
            result.append((first, node.end_lineno, node.name))
    return result


def extract_function(source, target, stored, parsed):
    """Use the innermost AST function, or an exactly located stored method."""
    lines = source.splitlines()
    if parsed is not None:
        candidates = [span for span in parsed if span[0] <= target <= span[1]]
        if candidates:
            first, last, name = min(candidates, key=lambda s: (s[1] - s[0], -s[0]))
            return first, last, "\n".join(lines[first - 1:last]), "full_file_ast", name
        # Complete AST proves this line is outside a function.
        return None
    if not stored.strip():
        return None
    stored = stored.replace("\r\n", "\n").rstrip("\n")
    if not re.match(r"\s*(?:async\s+)?def\s+", stored):
        return None
    offset = 0
    while True:
        index = source.find(stored, offset)
        if index < 0:
            return None
        first = source.count("\n", 0, index) + 1
        last = first + stored.count("\n")
        if first <= target <= last:
            # Reject partial stored methods when they are valid Python but do
            # not form a complete top-level function after dedenting.
            try:
                nodes = ast.parse(textwrap.dedent(stored)).body
                if len(nodes) != 1 or not isinstance(nodes[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
                    return None
                name = nodes[0].name
                kind = "stored_method_exact_source_match_ast_validated"
            except (SyntaxError, ValueError, RecursionError):
                # Python 2 source can fail the Python 3 parser. Its method is
                # accepted only as a literal source span covering target line.
                name = re.search(r"(?:async\s+)?def\s+(\w+)", stored).group(1)
                kind = "stored_method_exact_source_match_legacy_syntax"
            return first, last, "\n".join(lines[first - 1:last]), kind, name
        offset = index + 1


def comment_line_aligned(comment, source_line):
    """Require a meaningful serialized-comment segment on the recorded line."""
    segments = [canonical(s) for s in comment.split(" || ")]
    segments = [s for s in segments if len(re.sub(r"\W", "", s)) >= 5]
    line = canonical(source_line)
    return any(segment in line or (len(line) >= 12 and line in segment) for segment in segments)


def rank(row):
    return digest(SELECTION_SEED + ":" + row["sample_id"])


def target_is_comment_or_string(function, relative_line):
    try:
        return any(token.type in (tokenize.COMMENT, tokenize.STRING)
                   and token.start[0] <= relative_line <= token.end[0]
                   for token in tokenize.generate_tokens(io.StringIO(textwrap.dedent(function)).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return False


def token_key(text):
    return tuple(re.findall(r"\w+", canonical(text).lower()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--window-radius", type=int, default=10)
    args = parser.parse_args()
    csv.field_size_limit(20_000_000)
    final, context = read_csv(FINAL), read_csv(CONTEXT)
    id_counts = Counter(r["Dataset ID"] for r in final)
    source_by_id = defaultdict(list)
    for row in context:
        source_by_id[row["dataset_id"]].append(row)
    excluded, details, eligible = Counter(), [], []
    ast_cache = {}

    def reject(row, reason):
        excluded[reason] += 1
        details.append({"sample_id": row["Dataset ID"], "gold_label": row["ML TD Type"], "reason": reason})

    for row in final:
        sid, comment = row["Dataset ID"], row["Comment Text"]
        if id_counts[sid] > 1:
            reject(row, "duplicate_final_id_group")
            continue
        matches = source_by_id.get(sid, [])
        if not matches:
            reject(row, "no_source_id_match")
            continue
        matches = [s for s in matches if s["comment_text"] == comment
                   and s["repo_name"] == row["Repo Name"]
                   and s["filename"] == row["Filename"]]
        if len(matches) != 1:
            reject(row, "source_comment_project_path_not_unique_exact_match")
            continue
        src = matches[0]
        source = src["file_content"].replace("\r\n", "\n")
        lines = source.splitlines()
        try:
            zero_line = int(src["satd_line"])
        except ValueError:
            reject(row, "invalid_recorded_line")
            continue
        if zero_line < 0 or zero_line >= len(lines):
            reject(row, "recorded_line_out_of_bounds")
            continue
        target = zero_line + 1
        if not comment_line_aligned(comment, lines[zero_line]):
            reject(row, "comment_not_on_recorded_zero_based_line")
            continue
        source_hash = digest(source)
        if source_hash not in ast_cache:
            ast_cache[source_hash] = ast_functions(source)
        func = extract_function(source, target, src["containing_method"], ast_cache[source_hash])
        if not func:
            reject(row, "no_verified_enclosing_function")
            continue
        first, last, function, method, name = func
        if not target_is_comment_or_string(function, target - first + 1):
            reject(row, "target_line_not_lexically_comment_or_string")
            continue
        # This stronger full-comment check excludes multi-line comments whose
        # whole evidence cannot be represented by all three comparison arms.
        if canonical(comment) not in canonical(function):
            reject(row, "full_serialized_comment_not_in_function")
            continue
        w_first = max(1, target - args.window_radius)
        w_last = min(len(lines), target + args.window_radius)
        window = "\n".join(lines[w_first - 1:w_last])
        if canonical(comment) not in canonical(window):
            reject(row, "full_serialized_comment_not_in_window")
            continue
        eligible.append({
            "sample_id": sid, "dataset_id": sid,
            "gold_label": row["ML TD Type"], "gold": row["ML TD Type"],
            "comment": comment, "project": row["Repo Name"], "repo": row["Repo Name"],
            "filename": row["Filename"],
            "introducing_url": row["Comment-Introducing Revision"],
            "snapshot_sha256": source_hash,
            "snapshot_revision_verified": False,
            "source_recorded_satd_line_zero_based": zero_line,
            "target_line_1based": target,
            "target_source_line": lines[zero_line],
            "window_context": window, "window_text": window,
            "window_start_line_1based": w_first, "window_end_line_1based": w_last,
            "function_context": function, "function_text": function,
            "function_start_line_1based": first, "function_end_line_1based": last,
            "function_name": name, "function_extraction": method,
            "comment_source_alignment": "exact_dataset_comment_project_path_plus_serialization_normalized_source_line_and_full_context",
            "serialized_comment_normalization": "escaped slash and quote; CSV semicolon/comma; || newline; leading comment markers; whitespace",
            "ml_pipeline_stage": row["ML Pipeline Stage"],
            "repo_type": row["Repo Type"],
        })

    # Deduplicate before class allocation. The rule is label independent.
    pool, seen_comments, seen_functions, clone_rejections = [], set(), set(), []
    tokenized = []
    for row in sorted(eligible, key=rank):
        comment_key = canonical(row["comment"]).lower()
        function_key = digest(re.sub(r"\s+", "", row["function_context"]))
        tokens = token_key(row["comment"])
        reason, representative = None, None
        if comment_key in seen_comments:
            reason = "normalized_duplicate_comment"
        elif function_key in seen_functions:
            reason = "identical_function_ignoring_whitespace"
        elif len(tokens) >= 8:
            for previous, old_tokens in tokenized:
                if len(old_tokens) < 8 or min(len(tokens), len(old_tokens)) / max(len(tokens), len(old_tokens)) < .90:
                    continue
                if SequenceMatcher(None, tokens, old_tokens, autojunk=False).ratio() >= .90:
                    reason = "near_duplicate_comment_token_similarity_ge_0.90"
                    representative = previous["sample_id"]
                    break
        if reason:
            excluded[reason] += 1
            clone_rejections.append({"sample_id": row["sample_id"], "gold_label": row["gold_label"], "reason": reason, "representative_id": representative})
            continue
        pool.append(row)
        seen_comments.add(comment_key)
        seen_functions.add(function_key)
        tokenized.append((row, tokens))

    # Class-balanced round robin preserves every eligible label, then balances
    # additional examples without resampling rare labels or using predictions.
    buckets = defaultdict(list)
    for row in pool:
        buckets[row["gold_label"]].append(row)
    for bucket in buckets.values():
        bucket.sort(key=rank)
    selected, cursor = [], 0
    labels = sorted(buckets)
    while len(selected) < args.size:
        progressed = False
        for label in labels:
            if cursor < len(buckets[label]):
                selected.append(buckets[label][cursor])
                progressed = True
                if len(selected) == args.size:
                    break
        if not progressed:
            break
        cursor += 1
    if len(selected) != args.size:
        raise RuntimeError(f"Only {len(selected)} verified unique candidates; requested {args.size}")
    selected.sort(key=rank)
    output = BASE / "data"
    output.mkdir(exist_ok=True)
    with (output / "cohort.jsonl").open("w", encoding="utf-8") as stream:
        for row in selected:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output / "arms.jsonl").open("w", encoding="utf-8") as stream:
        for row in selected:
            for arm, key in (("comment_only", "comment"), ("window", "window_context"), ("function", "function_context")):
                stream.write(json.dumps({"sample_id": row["sample_id"], "arm": arm, "text": row[key], "gold_label": row["gold_label"]}, ensure_ascii=False) + "\n")
    with (output / "selection.csv").open("w", encoding="utf-8", newline="") as stream:
        columns = ["sample_id", "gold_label", "project", "filename", "target_line_1based", "function_start_line_1based", "function_end_line_1based", "function_extraction", "introducing_url"]
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)
    with (output / "excluded.jsonl").open("w", encoding="utf-8") as stream:
        for row in details + clone_rejections:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    audit = {
        "version": 1, "selection_seed": SELECTION_SEED,
        "source_files": {str(path): {"rows": count, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path, count in ((FINAL, len(final)), (CONTEXT, len(context)))},
        "target_size": args.size, "selected_size": len(selected),
        "original_classes": dict(sorted(Counter(r["ML TD Type"] for r in final).items())),
        "context_verified_before_dedup": len(eligible),
        "context_verified_unique_pool": len(pool),
        "eligible_class_counts": dict(sorted(Counter(r["gold_label"] for r in pool).items())),
        "selected_class_counts": dict(sorted(Counter(r["gold_label"] for r in selected).items())),
        "selected_project_count": len({r["project"] for r in selected}),
        "selected_project_counts": dict(sorted(Counter(r["project"] for r in selected).items())),
        "missing_original_labels": sorted({r["ML TD Type"] for r in final} - {r["gold_label"] for r in selected}),
        "excluded_counts": dict(sorted(excluded.items())),
        "not_selected_from_eligible_unique_pool": len(pool) - len(selected),
        "selected_function_extraction_counts": dict(Counter(r["function_extraction"] for r in selected)),
        "window_radius_lines": args.window_radius,
        "window_line_count_min_max": [min(r["window_end_line_1based"] - r["window_start_line_1based"] + 1 for r in selected), max(r["window_end_line_1based"] - r["window_start_line_1based"] + 1 for r in selected)],
        "function_line_count_min_max": [min(r["function_end_line_1based"] - r["function_start_line_1based"] + 1 for r in selected), max(r["function_end_line_1based"] - r["function_start_line_1based"] + 1 for r in selected)],
        "cohort_sha256": hashlib.sha256((output / "cohort.jsonl").read_bytes()).hexdigest(),
        "selection_rule": f"Stable SHA256 rank; label-independent exact/near-duplicate rejection; round-robin over sorted labels until {args.size}; no predictions used; no replacement sampling",
        "arms": ["comment_only", "window", "function"],
        "invariants": {"same_sample_ids_and_gold_across_arms": True, "all_target_comments_verified_in_each_context": True, "all_target_lines_lexically_comments_or_strings": True, "all_line_spans_one_based_in_outputs": True, "source_recorded_satd_line_is_zero_based": True, "full_functions_not_truncated": True, "original_csvs_unchanged": True},
        "limitations": [
            "Convenience cohort restricted to exact source-aligned Python comments in extractable functions; not a representative or held-out test set.",
            "Class-balanced sample deliberately differs from the 393-sample original prevalence; publish both macro metrics and per-class counts.",
            "No introducing commit was fetched or checked. The local snapshot contains the target comment, but its precise historical revision is unknown; contemporary introducing-code claims are not justified.",
            "Only code snapshots are used; neither removal changes nor commit messages enter input. Snapshot timestamps and unrelated code evolution cannot be independently established offline.",
            "Original gold labels are retained verbatim and may be ambiguous; this pilot does not adjudicate or relabel them.",
            "Legacy-syntax fallback uses only an exactly located stored method span covering the target; its source file cannot be parsed by the local Python 3 parser.",
            "Near duplicates use >=0.90 token-sequence similarity for comments with >=8 tokens and exact function whitespace hashes; this is a conservative heuristic, not a comprehensive clone detector.",
        ],
    }
    (output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: audit[key] for key in ("selected_size", "context_verified_before_dedup", "context_verified_unique_pool", "selected_class_counts", "selected_project_count", "missing_original_labels", "excluded_counts", "selected_function_extraction_counts", "window_line_count_min_max", "function_line_count_min_max")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
