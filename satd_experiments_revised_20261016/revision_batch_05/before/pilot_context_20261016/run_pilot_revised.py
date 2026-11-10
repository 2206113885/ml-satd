#!/usr/bin/env python3
# [R16/R25/R36 原代码保留]
# """Paired local Qwen3-8B context pilot, using frozen original prompt code.
# 
# Original datasets/scripts are never changed. Every completed prediction is
# flushed to disk; resuming requires identical protocol and input hashes.
# """
# [R16/R25/R36 修订]
"""Revised paired Qwen3 context pilot with an explicit finish/length protocol.

Historical frozen prompts and saved runs are evidence only. This entry uses a
new balanced prompt/parser and requires explicit inputs and a revised output
directory. No historical score or provenance is retroactively rewritten.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import time
from collections import Counter
from collections.abc import Mapping
from datetime import datetime
# [R16/R25/R36 原代码保留]
# from pathlib import Path
# [R16/R25/R36 修订]
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auxiliary_protocol import check_output_path, file_sha256

BASE = Path(__file__).resolve().parent
ARMS = ("comment_only", "window", "function")


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_reference():
    import importlib.util
    # [R16/R36 原代码保留]
    # spec = importlib.util.spec_from_file_location("pilot_frozen_prompt", BASE / "reference/frozen_prompt.py")
    # [R16/R36 修订]
    spec = importlib.util.spec_from_file_location("pilot_frozen_prompt", BASE / "reference_revised.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prompt_for(row, arm, reference):
    prompt = reference.build_prompt(row["comment"])
    if arm == "comment_only":
        return prompt
    suffix = "\n\nAbbreviation:"
    assert prompt.endswith(suffix)
    code = row["window_context" if arm == "window" else "function_context"]
    return prompt[:-len(suffix)] + (
        "\n\nSource code context for the target comment above:\n```python\n"
        + code
        + "\n```\nUse this code only as evidence to disambiguate the target comment. "
          "Classify that comment, not other comments in the code.\n\nAbbreviation:"
    )


def row_seed(sample_id, seed):
    # Common random numbers across arms, independent of execution/resume order.
    return int(sha256(f"{seed}:{sample_id}".encode())[:8], 16)


def as_token_ids(value):
    """Transformers 5 may return BatchEncoding rather than a bare list."""
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("Expected a single chat input")
        value = value[0]
    if not value or not all(isinstance(token, int) for token in value):
        raise ValueError("Expected nonempty integer input IDs")
    return list(value)


# [R16/R25 原代码保留]
# def parse_response(raw, reference, thinking=True, finish_reason=None):
#     thinking_closed = "</think>" in raw
#     visible = raw.rsplit("</think>", 1)[-1].strip()
#     if thinking and not thinking_closed:
#         return None, False, visible, "thinking_not_closed"
#     abbreviation = reference.simple_extract(visible)
#     if abbreviation == "<UNK>":
#         return None, False, visible, "unrecognized_output"
#     return abbreviation, True, visible, None
# [R16/R25 修订]
def parse_response(raw, reference, thinking=False, finish_reason=None, length_policy='accept_valid_label'):
    if not isinstance(raw, str):
        return None, False, '', 'empty_or_nontext_output'
    if finish_reason not in ('stop', 'length'):
        return None, False, raw.strip(), 'unconfirmed_finish_reason'
    visible = raw.strip()
    if thinking:
        if '</think>' not in raw:
            return None, False, visible, 'thinking_not_closed'
        visible = raw.rsplit('</think>', 1)[-1].strip()
    elif '<think>' in raw or '</think>' in raw:
        return None, False, visible, 'unexpected_native_thinking'
    if finish_reason == 'length' and length_policy == 'reject':
        return None, False, visible, 'token_limit_rejected_by_protocol'
    abbreviation = reference.simple_extract(visible)
    if abbreviation not in reference.ABBR_SET:
        return None, False, visible, 'unrecognized_output'
    return abbreviation, True, visible, None


def common_prefix(sequences):
    prefix = list(sequences[0])
    for sequence in sequences[1:]:
        k = 0
        while k < min(len(prefix), len(sequence)) and prefix[k] == sequence[k]:
            k += 1
        prefix = prefix[:k]
    # Keep at least one token to prefill for each independent generation.
    return prefix[:min(len(prefix), min(map(len, sequences)) - 1)]


def build_jobs(rows, tokenizer, reference, seed):
    jobs = []
    for row in rows:
        order = list(ARMS)
        random.Random(row_seed(row["sample_id"], seed)).shuffle(order)
        for arm in order:
            prompt = prompt_for(row, arm, reference)
            messages = [{"role": "user", "content": prompt}]
            ids = as_token_ids(tokenizer.apply_chat_template(messages, tokenize=True,
                                                add_generation_prompt=True,
                                                enable_thinking=False))
            jobs.append({"sample": row, "arm": arm, "prompt": prompt,
                         "ids": ids, "seed": row_seed(row["sample_id"], seed)})
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
# [R25/R26 原代码保留]
#     parser.add_argument("--cohort", type=Path, default=BASE / "data/cohort.jsonl")
# [R25/R26 修订]
    parser.add_argument("--cohort", type=Path, required=True)
# [R25 原代码保留]
#     parser.add_argument("--run-dir", type=Path, default=BASE / "runs/main")
# [R25 修订]
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--length-policy", choices=("accept_valid_label", "reject"), default="accept_valid_label")
    parser.add_argument("--limit", type=int, default=None, help="For separate smoke runs only")
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--prefill-step-size", type=int, default=512)
    parser.add_argument("--no-prefix-cache", action="store_true")
    parser.add_argument("--check-only", action="store_true")
# [R25 原代码保留]
#     args = parser.parse_args()
# [R25 修订]
    args = parser.parse_args()
    args.run_dir = check_output_path(args.run_dir)
    if args.max_input_tokens < 1 or args.max_new_tokens < 1:
        raise ValueError("Token budgets must be positive")
    reference = load_reference()
    manifest = json.loads((BASE / "model_manifest.json").read_text())
# [R24/R31 原代码保留]
#     model_dir = Path(manifest["local_dir"])
# [R24/R31 修订]
    model_dir = Path(manifest["local_dir"])
    config_data = json.loads((model_dir / "config.json").read_text())
    context_limit = int(config_data["max_position_embeddings"])
    args.max_input_tokens = min(args.max_input_tokens, context_limit - args.max_new_tokens - 64)
    if args.max_input_tokens < 1:
        raise ValueError("No input space remains after output reservation")
    local_model_hashes = {p.name: file_sha256(p) for p in sorted(model_dir.iterdir())
                          if p.is_file() and p.suffix in {".json", ".safetensors", ".model", ".jinja", ".txt"}}
    for weight in manifest["weights"]:
        expected = weight.get("downloaded_sha256") or weight.get("lfs_sha256")
        if not expected or local_model_hashes.get(weight["path"]) != expected:
            raise ValueError("Current model weight bytes do not match pinned manifest")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    rows = read_jsonl(args.cohort)
    if args.limit is not None:
        rows = rows[:args.limit]
    if not rows or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Empty cohort or duplicate IDs")
    if any(r["gold_label"] not in reference.ABBR_SET for r in rows):
        raise ValueError("Unknown gold label")
    jobs = build_jobs(rows, tokenizer, reference, args.seed)
    if any(len(job["ids"]) > args.max_input_tokens for job in jobs):
        longest = max(jobs, key=lambda job: len(job["ids"]))
        raise ValueError(f"No silent truncation: {longest['sample']['sample_id']} / "
                         f"{longest['arm']} has {len(longest['ids'])} tokens")
    prefix = [] if args.no_prefix_cache else common_prefix([job["ids"] for job in jobs])
    args.run_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "cohort": str(args.cohort.resolve()), "cohort_sha256": sha256(args.cohort.read_bytes()),
        "selected_ids": [r["sample_id"] for r in rows],
        "model_repo": manifest["repo_id"], "model_revision": manifest["revision"],
        "model_precision": manifest["precision"], "backend": "mlx-lm",
        "packages": {name: importlib.metadata.version(name) for name in
                     ["mlx", "mlx-lm", "transformers", "huggingface-hub", "numpy"]},
        "python": platform.python_version(), "platform": platform.platform(),
        # [R16/R36 原代码保留]
        # "reference_prompt_sha256": sha256((BASE / "reference/frozen_prompt.py").read_bytes()),
        # [R16/R36 修订]
        "reference_prompt_sha256": sha256((BASE / "reference_revised.py").read_bytes()),
        "runner_sha256": sha256(Path(__file__).read_bytes()),
        "arms": list(ARMS), "thinking": False,
        "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
        "sampling_explicit": True, "seed": args.seed,
        "seed_policy": "SHA256(seed:sample_id), same seed across arms",
        "execution_order": "Per-sample shuffled arms using deterministic independent seed",
        "max_input_tokens": args.max_input_tokens, "max_new_tokens": args.max_new_tokens,
        "prefill_step_size": args.prefill_step_size, "kv_quantization": None,
        "prefix_cache_tokens": len(prefix),
        "input_tokens_by_arm": {arm: {
            "min": min(len(job["ids"]) for job in jobs if job["arm"] == arm),
            "max": max(len(job["ids"]) for job in jobs if job["arm"] == arm)} for arm in ARMS},
        "truncation": "none; over-budget prompts fail before model load",
# [R16/R25/R31 原代码保留]
#         "parser": "Original nonthinking simple_extract on the first output line",
# [R16/R25/R31 修订]
        "parser": "Revised whole-output canonical case/whitespace match to one of 23 abbreviations",
        "length_policy": args.length_policy,
        "finish_policy": "Only stop/length accepted; length may score as valid iff the entire visible output is a legal abbreviation and length_policy=accept_valid_label",
        "model_local_file_sha256": local_model_hashes,
        "official_origin_boundary": "Pinned manifest is recorded; local byte hashes alone do not establish historical server identity",
        "model_manifest_sha256": sha256((BASE / "model_manifest.json").read_bytes()),
        "helper_sha256": file_sha256(BASE.parent / "auxiliary_protocol.py"),
# [R25 原代码保留]
#         "failure_policy": "Unrecognized/unfinished output counts as wrong; no majority fallback",
# [R25 修订]
        "failure_policy": "Invalid whole-output label or unsupported finish counts wrong; valid length-limited output follows explicit length_policy; no majority fallback",
        "changes_from_original": [
            "CUDA/Transformers model runtime replaced by MLX on Apple GPU, same official BF16 checkpoint",
            # [R24/R31 原代码保留]
            # "Original 4096 input cap retained; all actual prompts logged, no truncation",
            # [R24/R31 修订]
            "Input cap and model/output reservation explicitly recorded; all actual prompts logged, no truncation",
            "Sampling explicitly enabled with original stated nonthinking parameters; original omitted do_sample",
            "Per-sample fixed seed, paired contextual arms, independent prefix cache reuse",
# [R16/R25 原代码保留]
#             "Original parse helper retained; unrecognized outputs not replaced by Data Conf",
# [R16/R25 修订]
            "Revised whole-output parser and explicit finish/length protocol; historical run scores are unchanged",
        ],
    }
    settings = args.run_dir / "settings.json"
    output = args.run_dir / "predictions.jsonl"
    has_predictions = output.exists() and output.stat().st_size > 0
    if has_predictions and not settings.exists():
        raise ValueError("Existing predictions have no protocol; refusing to attach new settings")
    if has_predictions and json.loads(settings.read_text()) != protocol:
        raise ValueError("Protocol changed; create a new run directory instead of mixing results")
    settings.write_text(json.dumps(protocol, ensure_ascii=False, indent=2))
    with (args.run_dir / "prompts.jsonl").open("w") as handle:
        for job in jobs:
            handle.write(json.dumps({"sample_id": job["sample"]["sample_id"], "arm": job["arm"],
                                     "prompt": job["prompt"], "prompt_tokens": len(job["ids"]),
                                     "chat_template_token_sha256": sha256(json.dumps(job["ids"]).encode())},
                                    ensure_ascii=False) + "\n")
    print(json.dumps({"samples": len(rows), "predictions": len(jobs),
                      "prefix_tokens": len(prefix), "input_tokens": protocol["input_tokens_by_arm"]}), flush=True)
    if args.check_only:
        return
    if not manifest["complete"]:
        raise ValueError("Model download/checksum verification is not complete")
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_lm import load, stream_generate
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
    from mlx_lm.sample_utils import make_sampler

    completed = set()
    expected_jobs = {(job["sample"]["sample_id"], job["arm"]): job for job in jobs}
    if output.exists():
        # Never discard a partial record silently; each completed row is durable.
        for row in read_jsonl(output):
            key = (row["sample_id"], row["arm"])
            if key in completed:
                raise ValueError(f"Duplicate completed key: {key}")
            if key not in expected_jobs:
                raise ValueError(f"Unexpected completed key: {key}")
            expected = expected_jobs[key]
            if (row["gold_label"] != expected["sample"]["gold_label"]
                    or row["prompt_sha256"] != sha256(expected["prompt"].encode())
                    or row["seed"] != expected["seed"]):
                raise ValueError(f"Mismatched completed record: {key}")
            completed.add(key)
    if len(completed) == len(jobs):
        print("RUN_ALREADY_COMPLETE", flush=True)
        return
    print("Loading pinned official BF16 model …", flush=True)
    model, loaded_tokenizer, config = load(str(model_dir), return_config=True)
    for job in jobs:
        loaded_ids = as_token_ids(loaded_tokenizer.apply_chat_template(
            [{"role": "user", "content": job["prompt"]}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False))
        if loaded_ids != job["ids"]:
            raise ValueError("HF and MLX tokenizer wrappers generated different chat template IDs")
    if config.get("quantization") or config.get("quantization_config"):
        raise ValueError("Expected unquantized checkpoint")
    parameters = tree_flatten(model.parameters())
    dtype_counts = Counter(str(value.dtype) for _, value in parameters)
    if any(isinstance(module, nn.QuantizedLinear) for _, module in model.named_modules()):
        raise ValueError("Unexpected quantized layer")
    runtime = {"device": mx.device_info(), "parameter_tensor_dtypes": dict(dtype_counts),
               "model_weight_bytes": sum(value.nbytes for _, value in parameters),
               "loaded_at": datetime.now().astimezone().isoformat()}
    (args.run_dir / "runtime.json").write_text(json.dumps(runtime, indent=2))
    sampler = make_sampler(temp=0.7, top_p=0.8, top_k=20, min_p=0.0)
    cache = make_prompt_cache(model)
    if prefix:
        start = time.monotonic()
        for _ in generate_step(mx.array(prefix), model, max_tokens=0,
                               prompt_cache=cache, prefill_step_size=args.prefill_step_size):
            pass
        mx.eval([entry.state for entry in cache])
        assert all(entry.offset == len(prefix) for entry in cache)
        runtime["common_prefix_prefill_seconds"] = time.monotonic() - start
        (args.run_dir / "runtime.json").write_text(json.dumps(runtime, indent=2))
    with output.open("a", buffering=1) as handle:
        for job in jobs:
            row, arm = job["sample"], job["arm"]
            key = (row["sample_id"], arm)
            if key in completed:
                continue
            mx.random.seed(job["seed"])
            current_cache = cache if prefix else make_prompt_cache(model)
            if prefix:
                assert all(entry.offset == len(prefix) for entry in current_cache)
            start = time.monotonic()
            pieces, last = [], None
            print(f"START {len(completed)+1}/{len(jobs)} {key[0]} {arm}", flush=True)
            for response in stream_generate(
                    model, loaded_tokenizer, job["ids"][len(prefix):],
                    max_tokens=args.max_new_tokens, sampler=sampler, prompt_cache=current_cache,
                    prefill_step_size=args.prefill_step_size, kv_bits=None):
                pieces.append(response.text)
                last = response
            raw = "".join(pieces)
# [R25 原代码保留]
#             pred, ok, visible, error = parse_response(raw, reference, False, last.finish_reason)
# [R25 修订]
            finish_reason = last.finish_reason if last is not None else None
            pred, ok, visible, error = parse_response(raw, reference, False, finish_reason, args.length_policy)
            result = {"sample_id": row["sample_id"], "arm": arm,
                      "gold_label": row["gold_label"], "predicted_label": pred,
                      "parse_ok": ok, "parse_error": error,
                      "raw_response": raw, "visible_response": visible,
                      "elapsed_seconds": round(time.monotonic() - start, 4),
                      "prompt_tokens": len(job["ids"]),
                      "uncached_prompt_tokens": len(job["ids"]) - len(prefix),
                      # [R25 原代码保留]
                      # "completion_tokens": last.generation_tokens,
                      # [R25 修订]
                      "completion_tokens": (last.generation_tokens if last is not None else 0),
                      # [R25 原代码保留]
                      # "generation_tokens_per_second": last.generation_tps,
                      # [R25 修订]
                      "generation_tokens_per_second": (last.generation_tps if last is not None else None),
# [R25 原代码保留]
#                       "finish_reason": last.finish_reason, "seed": job["seed"],
# [R25 修订]
                      "finish_reason": finish_reason, "seed": job["seed"],
                      "hit_limit": finish_reason == "length",
                      "accepted_at_limit": finish_reason == "length" and ok,
                      # [R25 原代码保留]
                      # "peak_memory_gb": last.peak_memory,
                      # [R25 修订]
                      "peak_memory_gb": (last.peak_memory if last is not None else None),
                      "prompt_sha256": sha256(job["prompt"].encode()),
                      "legacy_fallback_prediction": pred if ok else "Data Conf"}
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            completed.add(key)
            (args.run_dir / "status.json").write_text(json.dumps({
                "completed": len(completed), "total": len(jobs), "last_sample_id": key[0],
                "last_arm": arm, "updated_at": datetime.now().astimezone().isoformat()}))
            print(f"DONE {len(completed)}/{len(jobs)} {key[0]} {arm} gold={row['gold_label']} "
                  # [R25 原代码保留]
                  # f"pred={pred} tokens={last.generation_tokens} seconds={result['elapsed_seconds']}", flush=True)
                  # [R25 修订]
                  f"pred={pred} tokens={(last.generation_tokens if last is not None else 0)} seconds={result['elapsed_seconds']}", flush=True)
            if prefix:
                actual = current_cache[0].offset
                trim_prompt_cache(current_cache, actual - len(prefix))
                assert all(entry.offset == len(prefix) for entry in current_cache)
            mx.clear_cache()
    print("RUN_COMPLETE", len(completed), flush=True)


if __name__ == "__main__":
    main()
