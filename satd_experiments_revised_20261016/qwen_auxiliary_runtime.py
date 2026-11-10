"""Local-only Qwen3 auxiliary runtime; no loading occurs on import (R24/R31/R43)."""
LABEL_DEFINITIONS = "There are 23 types of Machine Learning (ML) SATD (Self-Admitted Technical Debt):\n\nMACHINE LEARNING KNOWLEDGE (MLKnow): Machine-learning software carries plenty of unique challenges. Uneducated solutions by unaware developers may have to be revisited.\n\nMODEL INTERPRETABILITY (MLInter): Machine-learning models are a black box, causing poor understanding of model functionality; this can lead to unknown behaviour.\n\nMODEL CODE COMPREHENSION (MCC): Model code carries extra legibility concerns not present in traditional software (e.g., poorly-named temporary matrix variables).\n\nDUPLICATE MODEL CODE (DMC): Code duplication frequently occurs in model code.\n\nDUPLICATE FEATURE EXTRACTION CODE (DFEC): Code duplication frequently occurs in feature-extraction code.\n\nWEIGHT CONFIGURATION (Weight Conf): Editing code that involves the weights of a model, or configuring a model's weights directly.\n\nLAYER CONFIGURATION (Layer Conf): Editing code that deals with model layers, or configuring layers themselves.\n\nHYPER-PARAMETER CONFIGURATION (HP Conf): Configuring hyper-parameters of a model, or editing default values of off-the-shelf models.\n\nCODE DEPENDENCY (MLDepend): A needed change is blocked by an external dependency (library / subsystem).\n\nGLUE CODE (GC): Supporting code written to interface with other code, inhibiting improvements due to dependent peculiarities.\n\nCUSTOM DATA TYPE (CDT): Using data types provided by general-purpose packages can cause extensive inter-operability burden.\n\nMULTIPLE LANGUAGES (ML): Components written in other languages may introduce difficulties in ML development.\n\nUNNECESSARY MODEL CODE (UMC): Model code that bottlenecks execution, is unreachable / deprecated, or is unnecessary and should be removed.\n\nDATA PROCESSING CONFIGURATION (Data Conf): Configuring how data is processed, by editing the data directly or adding new processing steps.\n\nPLAIN OLD DATA TYPE (PODT): Using raw data types in ML causes confusion when interpreting processes.\n\nDATA STORAGE CONFIGURATION (DS Conf): Configuring how data is represented internally or stored externally.\n\nABSTRACTION (Abs): Lack of abstractions in ML systems causes cascading changes when a component is modified.\n\nBOUNDARY EROSION (BE): Lack of boundaries between subsystems complicates maintenance and isolates changes poorly.\n\nMODEL CODE MODIFIABILITY (MCM): Model code should enable easy maintenance and future modifications.\n\nMODEL CODE REUSABILITY (MCR): Model code should be generalised for reuse in varying situations.\n\nPREDICTION QUALITY (PQ): Workflow changes may affect performance.\n\nMACHINE LEARNING RELIABILITY (MLRelia): Model functionality depends on data quality; robustness measures should be in place.\n\nPROTOTYPE (Proto): Deploying small-scale prototypes into full systems can be dangerous."

import hashlib
import json
import random
from copy import deepcopy
from pathlib import Path
from auxiliary_protocol import ABBREVIATIONS, file_sha256, versions


def exact_abbreviation(raw):
    # A single full abbreviation only; do not search explanations for the first candidate.
    normalized = ' '.join(str(raw).split()).casefold()
    return {label.casefold(): label for label in ABBREVIATIONS}.get(normalized)


class QwenRuntime:
    def __init__(self, model_dir, input_cap=4096):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        self.torch, self.set_seed = torch, set_seed
        self.path = Path(model_dir).resolve()
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.path, local_files_only=True, device_map='auto', torch_dtype='auto')
        if self.model.config.model_type != 'qwen3':
            raise ValueError('This new protocol requires Qwen3 and explicit enable_thinking=False')
        self.model.eval()
        self.input_cap = input_cap
        self.context = int(self.model.config.max_position_embeddings)
        self.loaded_generation_config = self.model.generation_config.to_dict()
        # Hash actual local artifacts; a supplied path or revision string alone does not verify official origin.
        allowed_suffixes = {'.json', '.safetensors', '.bin', '.model', '.jinja', '.txt'}
        files = [p for p in self.path.iterdir() if p.is_file() and p.suffix in allowed_suffixes]
        self.identity = {
            'local_dir': str(self.path), 'files_sha256': {p.name: file_sha256(p) for p in sorted(files)},
            'config_commit_hash_unverified': getattr(self.model.config, '_commit_hash', None),
            'origin_status': 'Local bytes hashed; official repository/commit ancestry is NOT established by this alone',
            'template_sha256': hashlib.sha256(str(self.tokenizer.chat_template).encode()).hexdigest(),
            # [R31 原代码保留]
            # 'versions': versions(['torch', 'transformers', 'accelerate', 'safetensors']),
            # [R31 修订]
            'versions': versions(['torch', 'transformers', 'accelerate', 'safetensors']),
            'model_device': str(self.model.device), 'model_dtype': str(self.model.dtype),
            'device_map': {str(k): str(v) for k, v in getattr(self.model, 'hf_device_map', {}).items()},
            'torch_deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cuda_devices': [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        }

    def generation_config(self, max_new_tokens, greedy=False):
        config = deepcopy(self.model.generation_config)
        config.max_new_tokens = int(max_new_tokens)
        config.num_beams = 1
        config.do_sample = not greedy
        if not greedy:
            config.temperature, config.top_p, config.top_k = 0.7, 0.8, 20
        return config

    def generate(self, prompt, max_new_tokens, seed, greedy=False):
        ids = self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False,
            return_tensors='pt', return_dict=True)
        # Full templated input is encoded once. No special tokens are appended a second time.
        input_tokens = ids['input_ids'].shape[1]
        budget = min(self.input_cap, self.context - int(max_new_tokens) - 64)
        if budget <= 0 or input_tokens > budget:
            raise ValueError('input_budget_exceeded: full prompt=' + str(input_tokens) + ', allowed=' + str(budget))
        token_list = ids['input_ids'][0].tolist()
        ids = ids.to(self.model.device)
        random.seed(seed)
        self.set_seed(seed)
        config = self.generation_config(max_new_tokens, greedy)
        with self.torch.inference_mode():
            output = self.model.generate(**ids, generation_config=config)
        generated = output[0][input_tokens:]
        raw = self.tokenizer.decode(generated, skip_special_tokens=True)
        eos = config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        stopped = bool(len(generated) and int(generated[-1]) in eos_ids)
        hit_limit = len(generated) >= max_new_tokens and not stopped
        return {'raw_response': raw, 'input_tokens': input_tokens,
                'input_budget': budget, 'completion_tokens': len(generated),
                'finish_reason': 'stop' if stopped else ('length' if hit_limit else 'other'),
                'hit_limit': hit_limit, 'seed': seed,
                'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                'input_token_sha256': hashlib.sha256(json.dumps(token_list).encode()).hexdigest(),
                'effective_generation_config': config.to_dict()}
