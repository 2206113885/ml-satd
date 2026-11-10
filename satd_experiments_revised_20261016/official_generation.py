# R46 STRICT POLICY: previous module description retained.
# """R45: one source of generation policy per model and native runtime mode.
#
# No model/tokenizer is loaded on import. Profiles distinguish author advice,
# checkpoint values, framework defaults, and reference-implementation inference.
# Task output caps and audit seeds are deliberately outside the model policy.
# """
"""R46: explicit author advice > checkpoint config > framework defaults.

No model/tokenizer is loaded on import. Usage examples and inferred reference
sampler settings are not promoted to general author recommendations.
Task output caps and audit seeds remain outside the model policy.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path


REGISTRY_PATH = Path(__file__).with_name("official_generation_profiles.json")
SAMPLING_KEYS = (
    "do_sample", "temperature", "top_p", "top_k", "min_p",
    "repetition_penalty", "num_beams", "typical_p", "epsilon_cutoff", "eta_cutoff",
)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _read_json(path):
    data = path.read_bytes()
    return json.loads(data), {"path": str(path), "sha256": _sha(data)}


def _model_entry(registry, model_path):
    basename = Path(str(model_path).rstrip("/")).name.lower()
    matches = [(key, entry) for key, entry in registry["models"].items()
               if basename in entry["aliases"]]
    if len(matches) != 1:
        raise ValueError("No unambiguous official model profile for: " + str(model_path))
    return matches[0]


def _fresh_model_generation(GenerationConfig, model, raw):
    # Transformers 5.20 accepts the raw dictionary and preserves legacy
    # generation fields that a PretrainedConfig constructor can discard.
    # Older 4.x requires a configuration object instead.
    try:
        return GenerationConfig.from_model_config(deepcopy(raw)), "raw_config_dict"
    except (AttributeError, TypeError):
        fresh = type(model.config).from_dict(deepcopy(raw))
        return GenerationConfig.from_model_config(fresh), "fresh_config_object_legacy_api"


def _resolve_framework_defaults(config):
    """Transformers 5 can defer defaults; materialize its own defaults for logs.

    On Transformers 4 these are already populated by GenerationConfig. On 5,
    resolve only fields for which the installed library supplies a non-None
    default; None values with an intentional disabled/optional meaning remain.
    """
    get_defaults = getattr(config, "_get_default_generation_params", None)
    resolved = {}
    if callable(get_defaults):
        defaults = get_defaults()
        for key, value in defaults.items():
            if getattr(config, key, None) is None and value is not None:
                setattr(config, key, deepcopy(value))
                resolved[key] = deepcopy(value)
    unresolved = [key for key in ("do_sample", "temperature", "top_p", "top_k",
                                  "repetition_penalty", "num_beams")
                  if getattr(config, key, None) is None]
    if unresolved:
        raise ValueError("Installed Transformers defaults cannot be resolved: " + ", ".join(unresolved))
    return resolved


# R46 STRICT POLICY: previous function preserved in place.
# def build_official_generation(model, model_path, native_thinking, max_new_tokens):
#     """Read a clean checkpoint config, then apply only the selected model policy.
#
#     This deliberately never copies model.generation_config, which pipelines or
#     legacy scripts may have mutated. No network request is made here. A gated
#     checkpoint's unknown values are read from the user's existing local files.
#     """
#     from transformers import GenerationConfig
#
#     if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens <= 0:
#         raise ValueError("An explicit positive task max_new_tokens is required")
#     registry, registry_source = _read_json(REGISTRY_PATH)
#     model_key, entry = _model_entry(registry, model_path)
#     if model_key.startswith("Qwen3-"):
#         mode = "thinking" if native_thinking else "nonthinking"
#     elif model_key == "gpt-oss-20b":
#         mode = "reasoning"
#     else:
#         if native_thinking:
#             raise ValueError("This checkpoint has no supported native thinking switch")
#         mode = "instruct"
#     profile = entry["modes"][mode]
#     checkpoint_dir = Path(model_path)
#     generation_path = checkpoint_dir / "generation_config.json"
#     model_config_path = checkpoint_dir / "config.json"
#     if generation_path.is_file():
#         raw, config_source = _read_json(generation_path)
#         config = GenerationConfig.from_dict(deepcopy(raw))
#         config_origin = "fresh_local_checkpoint_generation_config"
#     elif model_config_path.is_file():
#         raw, config_source = _read_json(model_config_path)
#         # Reconstruct a configuration object only; this does not load weights.
#         # R45 OFFICIAL PROFILE: initial object-only implementation retained.
#         # fresh_model_config = type(model.config).from_dict(deepcopy(raw))
#         # config = GenerationConfig.from_model_config(fresh_model_config)
#         config, model_config_api = _fresh_model_generation(GenerationConfig, model, raw)
#         config_source["generation_config_api"] = model_config_api
#         config_origin = "fresh_local_model_config_and_transformers_defaults"
#     elif entry.get("public_generation_config") is not None:
#         raw = deepcopy(entry["public_generation_config"])
#         config = GenerationConfig.from_dict(raw)
#         config_source = {"url": entry["generation_config_url"],
#                          "pinned_revision": entry["official_revision"],
#                          "embedded_json_sha256": _sha(json.dumps(raw, sort_keys=True).encode())}
#         config_origin = "bundled_pinned_official_public_generation_config"
#     else:
#         raise FileNotFoundError("No clean local generation_config.json/config.json for "
#                                 + model_key + "; gated defaults must not be guessed")
#
#     # R45 OFFICIAL PROFILE: initial base snapshot retained below as history.
#     # base_config = deepcopy(config.to_dict())
#     # Special-token fallback must also use disk data, never the script-mutated
#     # model.generation_config reference. GenerationConfig.from_model_config
#     # handles nested text configs (including Gemma 3) in the installed library.
#     protocol_fallback = {}
#     protocol_source = None
#     protocol_keys = ("bos_token_id", "eos_token_id", "pad_token_id", "decoder_start_token_id")
#     if model_config_path.is_file() and any(getattr(config, key, None) is None for key in protocol_keys):
#         model_raw, protocol_source = _read_json(model_config_path)
#         # R45 OFFICIAL PROFILE: object-only implementation retained.
#         # disk_model_config = type(model.config).from_dict(deepcopy(model_raw))
#         # disk_generation = GenerationConfig.from_model_config(disk_model_config)
#         disk_generation, protocol_api = _fresh_model_generation(GenerationConfig, model, model_raw)
#         protocol_source["generation_config_api"] = protocol_api
#         for key in protocol_keys:
#             value = getattr(disk_generation, key, None)
#             if getattr(config, key, None) is None and value is not None:
#                 setattr(config, key, deepcopy(value))
#                 protocol_fallback[key] = deepcopy(value)
#     base_config = deepcopy(config.to_dict())
#     if getattr(config, "eos_token_id", None) is None or getattr(config, "eos_token_id", None) == []:
#         raise ValueError("Clean checkpoint has no EOS token ID; do not guess an instruction model's stopping protocol")
#     overrides = deepcopy(profile["parameters"])
#     unused = config.update(**overrides)
#     if unused:
#         raise ValueError("Installed Transformers does not support profile fields: "
#                          + ", ".join(sorted(unused)))
#     default_resolution = _resolve_framework_defaults(config)
#     # HF's decoder-only pad fallback is EOS when a checkpoint has no pad ID.
#     # Materialize that technical fallback so the saved config is accurate.
#     pad_resolution = None
#     if getattr(config, "pad_token_id", None) is None and getattr(config, "eos_token_id", None) is not None:
#         eos = config.eos_token_id
#         config.pad_token_id = eos[0] if isinstance(eos, (list, tuple)) else eos
#         pad_resolution = {"pad_token_id": config.pad_token_id,
#                           "origin": "Transformers technical fallback to first EOS; not an author sampling recommendation"}
#     config.max_new_tokens = max_new_tokens
#     config._from_model_config = False
#
#     # 4.50+ may otherwise merge defaults back from a mutated model config even
#     # when a separate GenerationConfig was supplied. This is a generate() kwarg,
#     # not a GenerationConfig field. Older versions lack that merge switch.
#     generate_options = {}
#     parameters = inspect.signature(model.generate).parameters
#     if "use_model_defaults" in parameters:
#         generate_options["use_model_defaults"] = False
#     else:
#         # A wrapper hiding this keyword must not silently enable inheritance.
#         prepare = getattr(model, "_prepare_generation_config", None)
#         if prepare is not None and "use_model_defaults" in inspect.signature(prepare).parameters:
#             generate_options["use_model_defaults"] = False
#
#     metadata = {
#         "profile_id": profile["profile_id"], "model_key": model_key, "mode": mode,
#         "policy": registry["policy"], "registry": registry_source,
#         "helper_sha256": _sha(Path(__file__).read_bytes()),
#         "official_repo": entry["repo_id"], "official_source_revision": entry["official_revision"],
#         "official_model_card_url": entry["model_card_url"],
#         "official_generation_config_url": entry["generation_config_url"],
#         "author_sampling_recommendation_url": profile.get("sampling_recommendation_url"),
#         "reference_sampler_url": profile.get("reference_sampler_url"),
#         "official_source_revision_is_local_weight_revision": False,
#         "config_origin": config_origin, "config_source": config_source,
#         "fresh_model_config_protocol_source": protocol_source,
#         "fresh_model_config_protocol_fallback": protocol_fallback,
#         "technical_pad_fallback": pad_resolution,
#         "remote_generation_config_status": entry.get("remote_generation_config_status", "public_snapshot_verified"),
#         "parameter_evidence": deepcopy(profile["parameter_evidence"]),
#         "official_profile_overrides": overrides,
#         "framework_version": importlib.metadata.version("transformers"),
#         "materialized_framework_defaults": default_resolution,
#         "effective_sampler": {key: deepcopy(getattr(config, key, None)) for key in SAMPLING_KEYS},
#         "generate_options": generate_options,
#         "model_default_isolation": "During this sequential call, runtime uses a fresh config as model.generation_config fallback and restores the previous reference in finally; covers Transformers 4 special-token fallback and Transformers 5 unconditional None-field fallback.",
#         "input_output_budget_policy": registry["output_budget_policy"],
#         "requested_task_max_new_tokens": max_new_tokens,
#     }
#     return config, base_config, deepcopy(profile["chat_template_kwargs"]), metadata, generate_options

# R46 STRICT POLICY: only explicit recommendations override clean defaults.
def build_official_generation(model, model_path, native_thinking, max_new_tokens):
    """R46: author recommendation > clean checkpoint fields > framework defaults.

    This deliberately never copies model.generation_config, which pipelines or
    legacy scripts may have mutated. No network request is made here. A gated
    checkpoint's unknown values are read from the user's existing local files.
    """
    from transformers import GenerationConfig

    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens <= 0:
        raise ValueError("An explicit positive task max_new_tokens is required")
    registry, registry_source = _read_json(REGISTRY_PATH)
    model_key, entry = _model_entry(registry, model_path)
    if model_key.startswith("Qwen3-"):
        mode = "thinking" if native_thinking else "nonthinking"
    elif model_key == "gpt-oss-20b":
        mode = "reasoning"
    else:
        if native_thinking:
            raise ValueError("This checkpoint has no supported native thinking switch")
        mode = "instruct"
    profile = entry["modes"][mode]
    checkpoint_dir = Path(model_path)
    generation_path = checkpoint_dir / "generation_config.json"
    model_config_path = checkpoint_dir / "config.json"
    # R46 STRICT POLICY: never treat a usage example or a pipeline's modified
    # in-memory config as a checkpoint default. Prefer actual generation JSON.
    if generation_path.is_file():
        raw, config_source = _read_json(generation_path)
        config = GenerationConfig.from_dict(deepcopy(raw))
        config_origin = "fresh_local_checkpoint_generation_config"
    elif entry.get("public_generation_config") is not None:
        raw = deepcopy(entry["public_generation_config"])
        config = GenerationConfig.from_dict(raw)
        # R46 STRICT POLICY: first fallback provenance retained; a user-supplied
        # main-branch download must not be reported as a verified pinned file.
        # config_source = {"url": entry["generation_config_url"],
        #                  "pinned_revision": entry["official_revision"],
        #                  "embedded_json_sha256": _sha(json.dumps(raw, sort_keys=True).encode())}
        # config_origin = "bundled_pinned_official_generation_config"
        config_source = deepcopy(entry.get("generation_config_snapshot_source", {
            "url": entry["generation_config_url"],
            "pinned_revision": entry["official_revision"],
            "evidence_kind": "direct_official_repository_snapshot",
        }))
        config_source["embedded_json_sha256"] = _sha(json.dumps(raw, sort_keys=True).encode())
        config_origin = "bundled_official_source_generation_config_snapshot"
    elif model_config_path.is_file():
        raw, config_source = _read_json(model_config_path)
        config, model_config_api = _fresh_model_generation(GenerationConfig, model, raw)
        config_source["generation_config_api"] = model_config_api
        config_origin = "fresh_local_model_config_and_transformers_defaults"
    else:
        raise FileNotFoundError("No clean checkpoint configuration for " + model_key
                                + "; EOS and unknown gated defaults must not be guessed")

    # R45 OFFICIAL PROFILE: initial base snapshot retained below as history.
    # base_config = deepcopy(config.to_dict())
    # Special-token fallback must also use disk data, never the script-mutated
    # model.generation_config reference. GenerationConfig.from_model_config
    # handles nested text configs (including Gemma 3) in the installed library.
    protocol_fallback = {}
    protocol_source = None
    protocol_keys = ("bos_token_id", "eos_token_id", "pad_token_id", "decoder_start_token_id")
    if model_config_path.is_file() and any(getattr(config, key, None) is None for key in protocol_keys):
        model_raw, protocol_source = _read_json(model_config_path)
        # R45 OFFICIAL PROFILE: object-only implementation retained.
        # disk_model_config = type(model.config).from_dict(deepcopy(model_raw))
        # disk_generation = GenerationConfig.from_model_config(disk_model_config)
        disk_generation, protocol_api = _fresh_model_generation(GenerationConfig, model, model_raw)
        protocol_source["generation_config_api"] = protocol_api
        for key in protocol_keys:
            value = getattr(disk_generation, key, None)
            if getattr(config, key, None) is None and value is not None:
                setattr(config, key, deepcopy(value))
                protocol_fallback[key] = deepcopy(value)
    base_config = deepcopy(config.to_dict())
    if getattr(config, "eos_token_id", None) is None or getattr(config, "eos_token_id", None) == []:
        raise ValueError("Clean checkpoint has no EOS token ID; do not guess an instruction model's stopping protocol")
    overrides = deepcopy(profile["parameters"])
    unused = config.update(**overrides)
    if unused:
        raise ValueError("Installed Transformers does not support profile fields: "
                         + ", ".join(sorted(unused)))
    # Constructor defaults in 4.x and deferred defaults in 5.x both come
    # from the installed framework, never from fixed experiment-level knobs.
    default_resolution = _resolve_framework_defaults(config)
    configured_fields = set(raw)
    if config_origin == "fresh_local_model_config_and_transformers_defaults":
        for name in ("decoder", "generator", "text_config"):
            nested = raw.get(name)
            if isinstance(nested, dict):
                configured_fields.update(nested)
    parameter_sources = {}
    for key in SAMPLING_KEYS:
        if key in overrides:
            origin = "explicit_author_recommendation"
        elif key in default_resolution:
            # A checkpoint field explicitly set to None can still be resolved
            # by Transformers 5; its effective value is a framework default.
            origin = "transformers_framework_default"
        elif key in configured_fields:
            origin = "checkpoint_generation_config" if "model_config_and" not in config_origin else "legacy_model_config_generation_field"
        else:
            origin = "transformers_framework_default"
        parameter_sources[key] = {
            "origin": origin,
            "value": deepcopy(getattr(config, key, None)),
            "operative_for_sampling": bool(getattr(config, "do_sample", False))
            if key in {"temperature", "top_p", "top_k", "min_p", "typical_p", "epsilon_cutoff", "eta_cutoff"}
            else None,
        }

    # HF's decoder-only pad fallback is EOS when a checkpoint has no pad ID.
    # Materialize that technical fallback so the saved config is accurate.
    pad_resolution = None
    if getattr(config, "pad_token_id", None) is None and getattr(config, "eos_token_id", None) is not None:
        eos = config.eos_token_id
        config.pad_token_id = eos[0] if isinstance(eos, (list, tuple)) else eos
        pad_resolution = {"pad_token_id": config.pad_token_id,
                          "origin": "Transformers technical fallback to first EOS; not an author sampling recommendation"}
    config.max_new_tokens = max_new_tokens
    config._from_model_config = False

    # 4.50+ may otherwise merge defaults back from a mutated model config even
    # when a separate GenerationConfig was supplied. This is a generate() kwarg,
    # not a GenerationConfig field. Older versions lack that merge switch.
    generate_options = {}
    parameters = inspect.signature(model.generate).parameters
    if "use_model_defaults" in parameters:
        generate_options["use_model_defaults"] = False
    else:
        # A wrapper hiding this keyword must not silently enable inheritance.
        prepare = getattr(model, "_prepare_generation_config", None)
        if prepare is not None and "use_model_defaults" in inspect.signature(prepare).parameters:
            generate_options["use_model_defaults"] = False

    metadata = {
        "profile_id": profile["profile_id"], "model_key": model_key, "mode": mode,
        "policy": registry["policy"], "registry": registry_source,
        "helper_sha256": _sha(Path(__file__).read_bytes()),
        "official_repo": entry["repo_id"], "official_source_revision": entry["official_revision"],
        "official_model_card_url": entry["model_card_url"],
        "official_generation_config_url": entry["generation_config_url"],
        "author_sampling_recommendation_url": profile.get("sampling_recommendation_url"),
        "reference_sampler_url": profile.get("reference_sampler_url"),
        "official_source_revision_is_local_weight_revision": False,
        "config_origin": config_origin, "config_source": config_source,
        "fresh_model_config_protocol_source": protocol_source,
        "fresh_model_config_protocol_fallback": protocol_fallback,
        "technical_pad_fallback": pad_resolution,
        "remote_generation_config_status": entry.get("remote_generation_config_status", "public_snapshot_verified"),
        "parameter_evidence": deepcopy(profile["parameter_evidence"]),
        "official_profile_overrides": overrides,
        "generation_parameter_priority": registry["parameter_priority"],
        "parameter_sources": parameter_sources,
        "explicit_checkpoint_fields": sorted(configured_fields),
        "author_recommendation_urls": profile.get("recommendation_urls", []),
        "framework_version": importlib.metadata.version("transformers"),
        "framework_do_sample_fallback": "No forced True. Omitted checkpoint field uses installed GenerationConfig default (basic HF default False).",
        "materialized_framework_defaults": default_resolution,
        "effective_sampler": {key: deepcopy(getattr(config, key, None)) for key in SAMPLING_KEYS},
        "generate_options": generate_options,
        "model_default_isolation": "During this sequential call, runtime uses a fresh config as model.generation_config fallback and restores the previous reference in finally; covers Transformers 4 special-token fallback and Transformers 5 unconditional None-field fallback.",
        "input_output_budget_policy": registry["output_budget_policy"],
        "requested_task_max_new_tokens": max_new_tokens,
    }
    return config, base_config, deepcopy(profile["chat_template_kwargs"]), metadata, generate_options
