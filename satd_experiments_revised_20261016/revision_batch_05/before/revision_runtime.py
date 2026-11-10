"""Shared, explicit runtime support for the revised ML-SATD experiments.

Importing this module performs no model loading or experiment execution.
Historical predictions are never read or rewritten here. Runtime generation
events are recorded before downstream parsing or fallback can discard them.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import sys


INVALID = "<UNK>"


def _jsonable(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()
                if not re.search(r"(?:^|_)(?:token|password|secret|api_key)$", str(k), re.I)}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except (ValueError, TypeError):
            pass
    return str(value)


def _digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path):
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def answer_view(raw, native_thinking=False):
    if not isinstance(raw, str) or not raw.strip():
        return {"text": "", "status": "empty_output", "thinking_closed": None}
    closes = list(re.finditer(r"</think>", raw, re.I))
    opens = list(re.finditer(r"<think>", raw, re.I))
    if native_thinking and not closes:
        return {"text": "", "status": "unfinished_thinking", "thinking_closed": False}
    if opens and (not closes or opens[-1].start() > closes[-1].start()):
        return {"text": "", "status": "unfinished_thinking", "thinking_closed": False}
    if closes:
        text = raw[closes[-1].end():].strip()
    else:
        text = raw.strip()
    return {"text": text, "status": "ok" if text else "missing_final_answer",
            "thinking_closed": bool(closes) if native_thinking or opens else None}


def visible_answer(raw, native_thinking=False):
    return answer_view(raw, native_thinking)["text"]


def _canonical(text):
    return re.sub(r"[^a-z0-9]", "", text.casefold())


def parse_label(raw, label_abbr, allowed=None, native_thinking=False):
    view = answer_view(raw, native_thinking)
    result = {"label": INVALID, "parse_ok": False, "status": view["status"],
              "answer_text": view["text"]}
    if view["status"] != "ok":
        return result
    text = view["text"].strip()
    tagged = re.findall(r"(?mi)^\s*#Answer\s*[:：]\s*([^\r\n]*)$", text)
    if tagged:
        if len(tagged) != 1:
            result["status"] = "multiple_answer_fields"
            return result
        text = tagged[0].strip()
    elif len(text.splitlines()) != 1:
        result["status"] = "multiple_output_lines"
        return result
    text = re.sub(r"(?i)^\s*#?\s*(?:abbreviation|label|final answer|answer)\s*[:：]\s*", "", text)
    permitted = set(label_abbr.values()) if allowed is None else set(allowed)
    matches = {abbr for full, abbr in label_abbr.items() if abbr in permitted
               and _canonical(text) in {_canonical(abbr), _canonical(full)}}
    if len(matches) == 1:
        result.update(label=next(iter(matches)), parse_ok=True, status="ok")
    else:
        result["status"] = "ambiguous_label" if matches else "invalid_label"
    return result


def parse_classification(raw, label_abbr, with_reason=False):
    """Handle the explicitly requested label/reason contract without searching reasoning."""
    if not with_reason:
        return {**parse_label(raw, label_abbr), "reason": ""}
    if not isinstance(raw, str) or not raw.strip():
        return {**parse_label(raw, label_abbr), "reason": ""}
    lines = [line.strip() for line in raw.strip().splitlines() if line.strip()]
    fields = re.findall(r"(?mi)^\s*#?\s*(?:Abbreviation|Answer|Label)\s*[:：]\s*([^\r\n]*)$", raw)
    if len(fields) > 1:
        return {"label": INVALID, "parse_ok": False, "status": "multiple_answer_fields", "reason": ""}
    candidate = fields[0] if fields else lines[0]
    parsed = parse_label(candidate, label_abbr)
    reason_match = re.search(r"(?mis)^\s*(?:#Reason|Reason|REASON)\s*[:：]\s*(.+)", raw)
    reason = reason_match.group(1).strip() if reason_match else ""
    if reason:
        reason = re.split(r"(?mi)^\s*#?\s*(?:Answer|Abbreviation|Label)\s*[:：]", reason)[0].strip()
    # Extra answer-like lines cannot hide behind a valid first label. The
    # remaining lines must be the requested explanation field.
    if not fields and len(lines) > 1 and not re.match(r"(?i)^(?:#Reason|Reason)\s*[:：]", lines[1]):
        return {"label": INVALID, "parse_ok": False, "status": "unexpected_extra_output", "reason": reason}
    return {**parsed, "reason": reason}


def extract_gpt_final(response):
    """Extract final from structured assistant messages or raw Harmony tokens.

    A bare decoded string without final-channel evidence is deliberately not
    searched for category names. Structured content is the final field under
    the official GPT-OSS chat template; a thinking/analysis field is not used.
    """
    if isinstance(response, list):
        if response and isinstance(response[0], dict) and "generated_text" in response[0]:
            return extract_gpt_final(response[0]["generated_text"])
        assistants = [m for m in response if isinstance(m, dict) and m.get("role") == "assistant"]
        if not assistants:
            return {"answer_text": "", "parse_status": "missing_assistant_final"}
        last = assistants[-1]
        if last.get("channel") in {"analysis", "commentary"}:
            return {"answer_text": "", "parse_status": "missing_assistant_final"}
        content = last.get("content")
        if isinstance(content, str) and content.strip():
            if "<|" in content:
                return extract_gpt_final(content)
            return {"answer_text": content.strip(), "parse_status": "ok"}
        return {"answer_text": "", "parse_status": "missing_assistant_final"}
    if isinstance(response, dict):
        if "generated_text" in response:
            return extract_gpt_final(response["generated_text"])
        return extract_gpt_final([response])
    if not isinstance(response, str):
        return {"answer_text": "", "parse_status": "unsupported_response_structure"}
    pattern = r"(?:<\|start\|>assistant<\|meta_sep\|>final<\|message\|>|^<\|meta_sep\|>final<\|message\|>)"
    starts = list(re.finditer(pattern, response))
    if not starts:
        return {"answer_text": "", "parse_status": "missing_assistant_final"}
    final = response[starts[-1].end():]
    final = re.split(r"<\|(?:fim_suffix|im_end|end|start|ghissue)\|>", final, maxsplit=1)[0].strip()
    return {"answer_text": final, "parse_status": "ok" if final else "missing_assistant_final"}


def _finite_context(model, tokenizer):
    cfg = model.config
    configs = [cfg, getattr(cfg, "text_config", None)]
    values = [getattr(c, name, None) for c in configs if c is not None
              for name in ("max_position_embeddings", "max_sequence_length", "n_positions")]
    values.append(getattr(tokenizer, "model_max_length", None))
    valid = [v for v in values if isinstance(v, int) and 256 <= v <= 10_000_000]
    if not valid:
        raise ValueError("No verified finite model/tokenizer context limit; configure the checkpoint correctly.")
    return min(valid)


def full_prompt_inputs(text, tokenizer, model, max_new_tokens, **tokenizer_kwargs):
    """Encode the entire rendered input; refuse overflow rather than truncating.

    The full comment, optional code context, all demonstrations and template
    are included. The old per-script max_length is metadata, not a truncation
    command. The actual checkpoint context remains the hard limit.
    """
    legacy_limit = tokenizer_kwargs.pop("max_length", None)
    tokenizer_kwargs.pop("truncation", None)
    tokenizer_kwargs.pop("add_special_tokens", None)
    inputs = tokenizer(text, return_tensors="pt", truncation=False,
                       add_special_tokens=False, **tokenizer_kwargs)
    count = int(inputs["input_ids"].shape[-1])
    context = _finite_context(model, tokenizer)
    available_output = context - count - 64
    effective_output = min(int(max_new_tokens), max(0, available_output))
    budget = context - effective_output - 64
    info = {"input_tokens": count, "context_limit": context,
            "input_token_ids_sha256": _digest(json.dumps(inputs["input_ids"][0].tolist())),
            "requested_output_tokens": int(max_new_tokens),
            "reserved_output_tokens": effective_output,
            "effective_output_tokens": effective_output,
            "output_budget_adjusted": effective_output != int(max_new_tokens),
            "output_budget_adjustment_reason": "full_input_and_checkpoint_context" if effective_output != int(max_new_tokens) else "",
            "safety_tokens": 64,
            "input_budget": budget, "legacy_input_limit": legacy_limit,
            "truncation": False, "prompt_sha256": _digest(text)}
    if available_output <= 0 or count > budget:
        raise InputBudgetError(info)
    return inputs.to(model.device), info


class InputBudgetError(ValueError):
    def __init__(self, info):
        self.info = info
        super().__init__("Full prompt exceeds checkpoint context after output reserve: " + json.dumps(info))


def row_identity(row, index):
    values = {str(k): _jsonable(v) for k, v in dict(row).items()}
    lookup = {re.sub(r"[^a-z0-9]", "", k.casefold()): v for k, v in values.items()}
    def first(*names):
        return next((lookup[n] for n in names if n in lookup), None)
    record_hash = _digest(json.dumps(values, sort_keys=True, ensure_ascii=False, default=str))
    return {"source_row": int(index), "sample_uid": _digest(str(index) + ":" + record_hash),
            "dataset_id": first("datasetid", "sampleid", "id"),
            "project": first("project", "projectname", "repository", "repo", "reponame"),
            "file_path": first("filepath", "path", "filename"),
            "introducing_commit": first("introducingcommit", "commit", "commithash", "introducingurl", "commentintroducingrevision", "commitintroducingrevision"),
            "gold_original": first("mltdtype", "mlsatdtype", "label"),
            "source_record_sha256": record_hash,
            "source_metadata_json": json.dumps(values, ensure_ascii=False, default=str)}


class RunAudit:
    def __init__(self, run_dir, model_path, csv_path, seed=42):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = str(model_path)
        self.csv_path = str(csv_path)
        self.seed = int(os.environ.get("SATD_SEED", seed))
        self.events = []
        self.identity = {}
        self.call_index = 0
        self._settings = {"model_path": self.model_path, "source_csv": self.csv_path,
                          "source_csv_sha256": file_sha256(csv_path), "base_seed": self.seed,
                          "python_version": platform.python_version(),
                          "entry_script_sha256": file_sha256(sys.argv[0]),
                          "revision_runtime_sha256": file_sha256(__file__),
                          "checkpoint_small_file_hashes": {
                              name: file_sha256(Path(model_path) / name)
                              for name in ("config.json", "generation_config.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja", "chat_template.json")},
                          "checkpoint_revision_policy": "Record commit hash where exposed and local config/template hashes; unavailable revision is not inferred.",
                          "created_utc": datetime.now(timezone.utc).isoformat(),
                          "seed_protocol": "sha256(base_seed,sample_uid,stage,call_index); one configured seed per run",
                          "history": "New runtime observation; historical server configuration is not reconstructed."}
        weight_files = sorted(p for p in Path(model_path).glob("*")
                              if p.is_file() and (p.suffix in {".safetensors", ".gguf"}
                                                   or p.name.startswith("pytorch_model") and p.suffix == ".bin"))
        hash_weights = os.environ.get("SATD_HASH_WEIGHTS", "1") == "1"
        self._settings["checkpoint_weight_files"] = {
            p.name: {"bytes": p.stat().st_size,
                     "sha256": file_sha256(p) if hash_weights else None}
            for p in weight_files}
        self._settings["weight_identity_policy"] = "Local weight SHA256 is enabled by default; SATD_HASH_WEIGHTS=0 records size only and does not establish weight-byte identity. Remote checkpoints use exposed commit metadata in each generation event."
        for package in ("transformers", "torch", "tokenizers", "accelerate", "pandas", "scikit-learn"):
            try:
                self._settings[package + "_version"] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                self._settings[package + "_version"] = None
        lock_path = os.environ.get("SATD_ENV_LOCK")
        if lock_path:
            locked = json.loads(Path(lock_path).read_text())
            lock_keys = [k for k in self._settings if k.endswith("_version")]
            lock_keys += ["source_csv_sha256", "checkpoint_small_file_hashes", "checkpoint_weight_files"]
            mismatches = [k for k in lock_keys if locked.get(k) != self._settings[k]]
            if mismatches:
                raise ValueError("Environment/data/checkpoint differs from SATD_ENV_LOCK: " + ", ".join(mismatches))
            self._settings["environment_lock_sha256"] = file_sha256(lock_path)
        self._settings["environment_lock_policy"] = "SATD_ENV_LOCK optionally enforces a previous runtime manifest before inference; this manifest records the first run's actual versions."
        pinned = [package + "==" + self._settings[package + "_version"]
                  for package in ("transformers", "torch", "tokenizers", "accelerate", "pandas", "scikit-learn")
                  if self._settings[package + "_version"] is not None]
        (self.run_dir / "requirements-observed.txt").write_text("\n".join(pinned) + "\n")
        (self.run_dir / "runtime_manifest.json").write_text(json.dumps(self._settings, ensure_ascii=False, indent=2))

    def set_sample(self, source_row, row):
        self.identity = row_identity(row, source_row)
        self.events = []
        self.call_index = 0

    def sample_fields(self):
        return {**self.identity, "source_csv_sha256": self._settings["source_csv_sha256"]}

    def settings(self):
        return deepcopy(self._settings)

    def seed_for_call(self, stage, sample_key=None):
        key = sample_key if sample_key is not None else self.identity.get("sample_uid", "unassigned")
        seed = int(_digest(f"{self.seed}:{key}:{stage}:{self.call_index}")[:8], 16) % (2**31)
        self.call_index += 1
        return seed

    # R45 OFFICIAL PROFILE: previous event recorder retained.
    # def record_event(self, event):
    #     event = _jsonable(event)
    #     self.events.append(event)
    #     with (self.run_dir / "generation_events.jsonl").open("a", encoding="utf-8") as stream:
    #         stream.write(json.dumps({**self.sample_fields(), **event}, ensure_ascii=False) + "\n")

    def record_event(self, event):
        event = _jsonable(event)
        profile = event.get("official_generation_profile")
        if profile is not None:
            # Task/stage caps stay on individual events, rather than becoming
            # a model sampling policy or overwriting another stage's budget.
            summary = {key: value for key, value in profile.items()
                       if key != "requested_task_max_new_tokens"}
            profiles = self._settings.setdefault("official_generation_profiles", {})
            previous = profiles.get(profile["profile_id"])
            if previous is not None and previous != summary:
                raise ValueError("Official generation profile changed within the same run")
            profiles[profile["profile_id"]] = summary
            self._settings["generation_profile_policy"] = "R45: author recommendations; otherwise fresh local checkpoint defaults; caps/seeds unchanged"
            (self.run_dir / "runtime_manifest.json").write_text(
                json.dumps(self._settings, ensure_ascii=False, indent=2), encoding="utf-8")
        self.events.append(event)
        with (self.run_dir / "generation_events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({**self.sample_fields(), **event}, ensure_ascii=False) + "\n")


# R45 OFFICIAL PROFILE: previous function retained in place.
# def audited_generate(model, tokenizer, messages, generation_kwargs, *, audit=None,
#                      max_input_tokens=None, native_thinking=False,
#                      reasoning_effort=None, stage="classification", sample_key=None):
#     """One HF generation call, preserving raw output and effective settings."""
#     from transformers import set_seed
#     kwargs = dict(generation_kwargs)
#     template = {}
#     model_type = str(getattr(model.config, "model_type", "")).lower()
#     name = str(getattr(model.config, "_name_or_path", "")).lower()
#     is_gpt = "gpt_oss" in model_type or "gpt-oss" in name
#     is_qwen3 = "qwen3" in model_type or "qwen3" in name
#     if is_qwen3:
#         template["enable_thinking"] = bool(native_thinking)
#     elif native_thinking and not is_gpt:
#         raise ValueError("Native thinking requested for a checkpoint without a supported native-thinking template.")
#     if is_gpt:
#         effort = reasoning_effort or "medium"
#         if effort not in {"low", "medium", "high"}:
#             raise ValueError("reasoning_effort must be low, medium or high")
#         template["reasoning_effort"] = effort
#     elif reasoning_effort is not None:
#         raise ValueError("GPT-OSS reasoning_effort requested for another checkpoint")
#     text = tokenizer.apply_chat_template(messages, tokenize=False,
#                                          add_generation_prompt=True, **template)
#     if is_gpt and "Reasoning: " + template["reasoning_effort"] not in text:
#         raise ValueError("The loaded GPT-OSS template did not render its requested reasoning effort.")
#     config = deepcopy(kwargs.pop("generation_config", model.generation_config))
#     base_config = deepcopy(config.to_dict())
#     loaded_config = deepcopy(model.generation_config.to_dict())
#     unused = config.update(**kwargs)
#     if unused:
#         raise ValueError("Unsupported generation overrides: " + ", ".join(sorted(unused)))
#     max_new = getattr(config, "max_new_tokens", None)
#     if not isinstance(max_new, int) or max_new <= 0:
#         raise ValueError("An explicit positive max_new_tokens is required")
#     seed = audit.seed_for_call(stage, sample_key) if audit is not None else int(os.environ.get("SATD_SEED", "42"))
#     set_seed(seed)
#     event = {"stage": stage, "seed": seed, "native_thinking": bool(native_thinking or is_gpt),
#              "qwen_enable_thinking": bool(native_thinking) if is_qwen3 else None,
#              "reasoning_mode": "gpt_oss_native_reasoning" if is_gpt else "qwen3_native_thinking" if native_thinking else "prompt_or_non_native_reasoning",
#              "reasoning_effort": template.get("reasoning_effort"), "template_kwargs": template,
#              "prompt_sha256": _digest(text), "messages_sha256": _digest(json.dumps(messages, ensure_ascii=False)),
#              "rendered_prompt": text, "legacy_max_input_tokens": max_input_tokens,
#              "loaded_generation_config": loaded_config,
#              "base_generation_config": base_config,
#              "generation_call_overrides": _jsonable(kwargs),
#              "effective_generation_config": config.to_dict(),
#              "model_type": model_type, "model_commit_hash": getattr(model.config, "_commit_hash", None),
#              "tokenizer_commit_hash": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
#              "chat_template_sha256": _digest(str(getattr(tokenizer, "chat_template", ""))),
#              "device": str(model.device), "raw_text": "", "answer_text": ""}
#     try:
#         inputs, budget = full_prompt_inputs(text, tokenizer, model, max_new)
#     except InputBudgetError as exc:
#         event.update(exc.info, parse_status="input_budget_exceeded", stop_status="not_generated", output_tokens=0)
#         if audit is not None:
#             audit.record_event(event)
#         return event
#     event.update(budget)
#     # A configured generation maximum is an upper bound. Retain the complete
#     # input, then cap that upper bound by the remaining real context; this is
#     # explicit in each event, rather than silently truncating demonstrations.
#     max_new = budget["effective_output_tokens"]
#     config.max_new_tokens = max_new
#     event["effective_generation_config"] = config.to_dict()
#     import torch
#     with torch.inference_mode():
#         output = model.generate(**inputs, generation_config=config)
#     sequences = output.sequences if hasattr(output, "sequences") else output
#     generated = sequences[0][inputs["input_ids"].shape[-1]:]
#     ids = generated.tolist()
#     raw = tokenizer.decode(generated, skip_special_tokens=False)
#     decoded = tokenizer.decode(generated, skip_special_tokens=True)
#     eos = getattr(config, "eos_token_id", None)
#     eos_set = set(eos if isinstance(eos, (list, tuple)) else [eos]) - {None}
#     saw_eos = bool(ids and ids[-1] in eos_set)
#     stop = "eos" if saw_eos else "token_limit" if len(ids) >= max_new else "other_stop"
#     if is_gpt:
#         final = extract_gpt_final(raw)
#     else:
#         # Check native thinking against raw protocol text, even when a
#         # tokenizer regards think markers as special and omits them on decode.
#         view = answer_view(raw if native_thinking else decoded, native_thinking)
#         answer = view["text"]
#         for special in getattr(tokenizer, "all_special_tokens", []):
#             answer = answer.replace(special, "")
#         final = {"answer_text": answer.strip(), "parse_status": view["status"]}
#     event.update(raw_text=raw, decoded_text=decoded, output_tokens=len(ids),
#                  output_token_ids_sha256=_digest(json.dumps(ids)),
#                  stop_status=stop, reached_output_limit=len(ids) >= max_new,
#                  stopped_on_configured_eos=saw_eos, **final)
#     if audit is not None:
#         audit.record_event(event)
#     return event

# R45 OFFICIAL PROFILE: model policy opt-in; excluded legacy callers retain previous behavior.
def audited_generate(model, tokenizer, messages, generation_kwargs, *, audit=None,
                     max_input_tokens=None, native_thinking=False,
                     reasoning_effort=None, stage="classification", sample_key=None,
                     official_profile=False):
    """One HF generation call, preserving raw output and effective settings."""
    from transformers import set_seed
    kwargs = dict(generation_kwargs)
    template = {}
    model_type = str(getattr(model.config, "model_type", "")).lower()
    name = str(getattr(model.config, "_name_or_path", "")).lower()
    is_gpt = "gpt_oss" in model_type or "gpt-oss" in name
    is_qwen3 = "qwen3" in model_type or "qwen3" in name
    profile_metadata = None
    generate_options = {}
    loaded_config = deepcopy(model.generation_config.to_dict())
    if official_profile:
        from official_generation import build_official_generation
        if set(kwargs) != {"max_new_tokens"}:
            raise ValueError("Official-profile calls accept only the task max_new_tokens; sampling is model policy")
        if reasoning_effort is not None:
            raise ValueError("GPT-OSS effort is fixed by the official profile; remove the per-script override")
        model_path = audit.model_path if audit is not None else getattr(model.config, "_name_or_path", "")
        config, base_config, template, profile_metadata, generate_options = build_official_generation(
            model, model_path, bool(native_thinking), kwargs["max_new_tokens"])
    else:
        if is_qwen3:
            template["enable_thinking"] = bool(native_thinking)
        elif native_thinking and not is_gpt:
            raise ValueError("Native thinking requested for a checkpoint without a supported native-thinking template.")
        if is_gpt:
            effort = reasoning_effort or "medium"
            if effort not in {"low", "medium", "high"}:
                raise ValueError("reasoning_effort must be low, medium or high")
            template["reasoning_effort"] = effort
        elif reasoning_effort is not None:
            raise ValueError("GPT-OSS reasoning_effort requested for another checkpoint")
        config = deepcopy(kwargs.pop("generation_config", model.generation_config))
        base_config = deepcopy(config.to_dict())
        loaded_config = deepcopy(model.generation_config.to_dict())
        unused = config.update(**kwargs)
        if unused:
            raise ValueError("Unsupported generation overrides: " + ", ".join(sorted(unused)))
    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True, **template)
    if is_gpt and "Reasoning: " + template["reasoning_effort"] not in text:
        raise ValueError("The loaded GPT-OSS template did not render its requested reasoning effort.")
    max_new = getattr(config, "max_new_tokens", None)
    if not isinstance(max_new, int) or max_new <= 0:
        raise ValueError("An explicit positive max_new_tokens is required")
    seed = audit.seed_for_call(stage, sample_key) if audit is not None else int(os.environ.get("SATD_SEED", "42"))
    set_seed(seed)
    event = {"stage": stage, "seed": seed, "native_thinking": bool(native_thinking or is_gpt),
             "qwen_enable_thinking": bool(native_thinking) if is_qwen3 else None,
             "reasoning_mode": "gpt_oss_native_reasoning" if is_gpt else "qwen3_native_thinking" if native_thinking else "prompt_or_non_native_reasoning",
             "reasoning_effort": template.get("reasoning_effort"), "template_kwargs": template,
             "prompt_sha256": _digest(text), "messages_sha256": _digest(json.dumps(messages, ensure_ascii=False)),
             "rendered_prompt": text, "legacy_max_input_tokens": max_input_tokens,
             "loaded_generation_config": loaded_config,
             "loaded_generation_config_used_as_base": not official_profile,
             "official_generation_profile": profile_metadata,
             "generate_options": generate_options,
             "base_generation_config": base_config,
             "generation_call_overrides": _jsonable(kwargs),
             "effective_generation_config": config.to_dict(),
             "model_type": model_type, "model_commit_hash": getattr(model.config, "_commit_hash", None),
             "tokenizer_commit_hash": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
             "chat_template_sha256": _digest(str(getattr(tokenizer, "chat_template", ""))),
             "device": str(model.device), "raw_text": "", "answer_text": ""}
    try:
        inputs, budget = full_prompt_inputs(text, tokenizer, model, max_new)
    except InputBudgetError as exc:
        event.update(exc.info, parse_status="input_budget_exceeded", stop_status="not_generated", output_tokens=0)
        if audit is not None:
            audit.record_event(event)
        return event
    event.update(budget)
    # A configured generation maximum is an upper bound. Retain the complete
    # input, then cap that upper bound by the remaining real context; this is
    # explicit in each event, rather than silently truncating demonstrations.
    max_new = budget["effective_output_tokens"]
    config.max_new_tokens = max_new
    event["effective_generation_config"] = config.to_dict()
    import torch
    # R45 OFFICIAL PROFILE: previous invocation retained; newer Transformers
    # can fill None fields from model.generation_config without a disable flag.
    # with torch.inference_mode():
    #     output = model.generate(**inputs, generation_config=config, **generate_options)
    if official_profile:
        previous_generation_reference = model.generation_config
        try:
            # Isolate that fallback reference as well as the explicit config.
            # Core experiment scripts call generation sequentially.
            model.generation_config = deepcopy(config)
            with torch.inference_mode():
                output = model.generate(**inputs, generation_config=config, **generate_options)
        finally:
            model.generation_config = previous_generation_reference
    else:
        with torch.inference_mode():
            output = model.generate(**inputs, generation_config=config, **generate_options)
    sequences = output.sequences if hasattr(output, "sequences") else output
    generated = sequences[0][inputs["input_ids"].shape[-1]:]
    ids = generated.tolist()
    raw = tokenizer.decode(generated, skip_special_tokens=False)
    decoded = tokenizer.decode(generated, skip_special_tokens=True)
    eos = getattr(config, "eos_token_id", None)
    eos_set = set(eos if isinstance(eos, (list, tuple)) else [eos]) - {None}
    saw_eos = bool(ids and ids[-1] in eos_set)
    stop = "eos" if saw_eos else "token_limit" if len(ids) >= max_new else "other_stop"
    if is_gpt:
        final = extract_gpt_final(raw)
    else:
        # Check native thinking against raw protocol text, even when a
        # tokenizer regards think markers as special and omits them on decode.
        view = answer_view(raw if native_thinking else decoded, native_thinking)
        answer = view["text"]
        for special in getattr(tokenizer, "all_special_tokens", []):
            answer = answer.replace(special, "")
        final = {"answer_text": answer.strip(), "parse_status": view["status"]}
    event.update(raw_text=raw, decoded_text=decoded, output_tokens=len(ids),
                 output_token_ids_sha256=_digest(json.dumps(ids)),
                 stop_status=stop, reached_output_limit=len(ids) >= max_new,
                 stopped_on_configured_eos=saw_eos, **final)
    if audit is not None:
        audit.record_event(event)
    return event


def write_failure_metrics(pred_df, labels, output_dir, pred_col="pred", gold_col="real",
                          parse_ok_col="parse_ok", prefix="failure_as_error"):
    """Additional sensitivity metrics; the established fallback report stays primary."""
    from sklearn.metrics import accuracy_score, classification_report
    output_dir = Path(output_dir)
    values = pred_df[pred_col].where(pred_df[parse_ok_col].astype(bool), "<INVALID>")
    text = classification_report(pred_df[gold_col], values, labels=labels, zero_division=0, digits=3)
    report = classification_report(pred_df[gold_col], values, labels=labels, zero_division=0, output_dict=True)
    report["accuracy_all_rows"] = float(accuracy_score(pred_df[gold_col], values))
    report["invalid_count"] = int((~pred_df[parse_ok_col].astype(bool)).sum())
    (output_dir / (prefix + ".txt")).write_text(text, encoding="utf-8")
    (output_dir / (prefix + ".json")).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
