#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# [R16/R23/R31 原代码保留]
# """
# run_satd_gptoss20b_two_stage.py
# --------------------------------
# 模型：/data/yangzhifei/pro/ML_SATD/models/gpt-oss-20b
# 方法：Two-stage（Coarse 9 → Fine 23）
#   - Stage-1：输出 9 粗类缩写（AWR/RDB/.../SCL）
#   - Stage-2：在对应粗类的细类候选中输出唯一一个 23 缩写之一
# 解析：兼容 "**AWR**" / "Abbreviation: AWR" / 全称等格式；保留 assistantfinal 之后的文本
# 输出：results/run-YYYYmmdd-HHMMSS-模型名 / predictions.csv, confusion_matrix_fine.csv,
#       confusion_matrix_coarse.csv, report_fine.txt, report_coarse.txt, settings.json
# """
# [R16/R23/R31 修订]
"""Two-stage ML-SATD classification with audited full prompts, exact canonical
label parsing, retained policy fallbacks, and separate stage-1/coarse reports.
This revision changes future execution only; historical results are not re-evaluated.
"""

import re, os, gc, json
from pathlib import Path
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import torch
from transformers import pipeline
# [R16/R22/R24/R31/R33 修订]
import sys
import hashlib
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from revision_runtime import RunAudit, audited_generate, parse_label, write_failure_metrics

# ========== 0. 基础路径（仅此处改） ==========
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU 号
CSV_PATH   = "/data/yangzhifei/pro/ML_SATD/dataset/sampled_label_dataset_normalized_rmvtodo.csv"  # ← 修改
MODEL_PATH = "/data/yangzhifei/pro/ML_SATD/models/gpt-oss-20b"                                     # ← 修改（或 "openai/gpt-oss-20b"）

# 目录名包含时间戳 + 模型名
MODEL_NAME = Path(MODEL_PATH).name
# [R31 原代码保留]
# RUN_DIR    = Path("results") / f"run-{datetime.now():%Y%m%d-%H%M%S}-{MODEL_NAME}"
# RUN_DIR.mkdir(parents=True, exist_ok=True)
# [R31 修订] Use a fresh run directory; a rare collision fails instead of overwriting results.
# [R31 修订] 将新产物固定在修订目录，避免依赖启动工作目录。
# RUN_DIR = Path("results") / f"run-{datetime.now():%Y%m%d-%H%M%S-%f}-{MODEL_NAME}"
RUN_DIR = Path(__file__).resolve().parents[1] / "revised_outputs" / f'run-{datetime.now():%Y%m%d-%H%M%S-%f}-{MODEL_NAME}'
RUN_DIR.mkdir(parents=True, exist_ok=False)
# [R22/R29/R31 修订]
SEED = int(os.environ.get("SATD_SEED", "42"))
AUDIT = RunAudit(RUN_DIR, MODEL_PATH, CSV_PATH, seed=SEED)
# R29: configure the native template metadata, rather than a prose-only instruction.
GPT_OSS_REASONING_EFFORT = os.environ.get("SATD_REASONING_EFFORT", "medium")
if GPT_OSS_REASONING_EFFORT not in {"low", "medium", "high"}:
    raise ValueError("SATD_REASONING_EFFORT must be low, medium, or high")

# ========== 1. 其它参数 ==========
PRINT_UNPARSED = True
MAX_NEW_TOKENS = 32768

# ========== 2. 细粒度标签（23 类） ==========
FINE_LABEL_ABBR = {
    "Machine Learning Knowledge":"MLKnow","Model Interpretability":"MLInter",
    "Model Code Comprehension":"MCC","Duplicate Model Code":"DMC",
    "Duplicate Feature Extraction Code":"DFEC","Weight Configuration":"Weight Conf",
    "Layer Configuration":"Layer Conf","Hyper-parameter Configuration":"HP Conf",
    "Code Dependency":"MLDepend","Glue Code":"GC","Custom Data Type":"CDT",
    "Multiple Languages":"ML","Unnecessary Model Code":"UMC",
    "Data Processing Configuration":"Data Conf","Plain Old Data Type":"PODT",
    "Data Storage Configuration":"DS Conf","Abstraction":"Abs","Boundary Erosion":"BE",
    "Model Code Modifiability":"MCM","Model Code Reusability":"MCR",
    "Prediction Quality":"PQ","Machine Learning Reliability":"MLRelia","Prototype":"Proto",
}
FINE_ABBR2FULL  = {v:k for k,v in FINE_LABEL_ABBR.items()}
FINE_LABELS     = list(FINE_LABEL_ABBR.keys())
FINE_ABBR_SET   = set(FINE_ABBR2FULL.keys())

# ========== 3. 粗粒度（9 类）与细↔粗映射 ==========
COARSE_LABELS = ["AWR","RDB","DCE","CFO","CDD","DTD","MDL","PRF","SCL"]
COARSE_ABBR_SET = set(COARSE_LABELS)

# 细(缩写) → 粗
FINE2COARSE_ABBR = {
    "MLKnow":"AWR","MLInter":"AWR",
    "MCC":"RDB",
    "DMC":"DCE","DFEC":"DCE",
    "Weight Conf":"CFO","Layer Conf":"CFO","HP Conf":"CFO",
    "MLDepend":"CDD","GC":"CDD","CDT":"CDD","ML":"CDD","UMC":"CDD",
    "Data Conf":"DTD","PODT":"DTD","DS Conf":"DTD",
    "Abs":"MDL","BE":"MDL","MCM":"MDL","MCR":"MDL",
    "PQ":"PRF","MLRelia":"PRF",
    "Proto":"SCL",
}

# 粗 → 细(缩写) 候选集合
COARSE2FINE_ABBR = {
    "AWR":  ["MLKnow","MLInter"],
    "RDB":  ["MCC"],
    "DCE":  ["DMC","DFEC"],
    "CFO":  ["Weight Conf","Layer Conf","HP Conf"],
    "CDD":  ["MLDepend","GC","CDT","ML","UMC"],
    "DTD":  ["Data Conf","PODT","DS Conf"],
    "MDL":  ["Abs","BE","MCM","MCR"],
    "PRF":  ["PQ","MLRelia"],
    "SCL":  ["Proto"],
}

# 细(全称) → 粗（用于评估映射）
FINE2COARSE_FULL = {
    "Machine Learning Knowledge":"AWR", "Model Interpretability":"AWR",
    "Model Code Comprehension":"RDB",
    "Duplicate Model Code":"DCE", "Duplicate Feature Extraction Code":"DCE",
    "Weight Configuration":"CFO", "Layer Configuration":"CFO", "Hyper-parameter Configuration":"CFO",
    "Code Dependency":"CDD", "Glue Code":"CDD", "Custom Data Type":"CDD", "Multiple Languages":"CDD", "Unnecessary Model Code":"CDD",
    "Data Processing Configuration":"DTD", "Plain Old Data Type":"DTD", "Data Storage Configuration":"DTD",
    "Abstraction":"MDL", "Boundary Erosion":"MDL", "Model Code Modifiability":"MDL", "Model Code Reusability":"MDL",
    "Prediction Quality":"PRF", "Machine Learning Reliability":"PRF",
    "Prototype":"SCL",
}

# 各粗类下细类的回退（细类解析失败时使用）
FALLBACK_FINE_BY_COARSE = {
    "AWR":"MLKnow", "RDB":"MCC", "DCE":"DMC", "CFO":"HP Conf", "CDD":"MLDepend",
    "DTD":"Data Conf", "MDL":"MCM", "PRF":"PQ", "SCL":"Proto",
}

# ========== 4. Prompt ==========
COARSE_PROMPT_HEAD = r"""
There are 9 coarse-grained types of ML SATD:

Awareness (AWR): Lack of developer understanding or certainty about ML algorithms, APIs, data, or model behavior. Often arises due to “black box” nature of ML models.
Example Symptoms:Doubts on correctness, API usage uncertainty, unclear design decisions, unknown error causes, lack of domain knowledge.

Readability (RDB):Model code has poor legibility or unclear logic, especially in ML contexts (e.g., poorly named variables, dense math code).
Example Symptoms:Refactoring for clarity, simplifying complex statements, improving naming.

Duplicate Code Elimination (DCE): Identical or near-identical ML model or preprocessing logic appears in multiple places.
Example Symptoms:Repeated feature extraction, repeated model evaluation loops.

Configurable Options (CFO): Missing or poorly implemented configuration points for weights, network layers, or model parameters.
Example Symptoms:Hard-coded parameters, unimplemented config flags, missing flexibility for experiments.

Code Dependency (CDD): Code depends on other code or libraries in a way that blocks maintenance, requires workarounds, or is outdated.
Example Symptoms:Waiting for library updates, integration hacks, dependency-induced bugs, deprecated model code.

Data Dependency (DTD): Issues in how data is processed, represented, or stored that hinder maintainability or correctness.
Example Symptoms:Missing preprocessing steps, poor choice of data formats, unoptimized storage.

Modularity (MDL): Poor modular boundaries or design make the system harder to evolve, reuse, or maintain.
Example Symptoms:Missing abstractions, test parameters used in production, model-specific code that should be generalized.

Performance (PRF): Performance or robustness is insufficient, requiring evaluation, optimization, or reliability improvements.
Example Symptoms:Model accuracy concerns, robustness checks, performance comparisons.

Scalability (SCL): Prototype-level code is used in production without scaling optimizations.
Example Symptoms:Inefficient loops, naive algorithms, lack of batching/parallelism.
""".strip()

def build_prompt_coarse(comment: str) -> str:
    tail = (
        "Please classify the following Code Comment into exactly ONE of these coarse abbreviations:\n"
        + ", ".join(COARSE_LABELS) +
        "\nReturn the abbreviation only (e.g., AWR)."
    )
    return f"{COARSE_PROMPT_HEAD}\n\n{tail}\n\nCode Comment:\n\"\"\"\n{comment}\n\"\"\"\n\nAnswer:"

FINE_PROMPT_HEAD = r"""
There are 23 types of Machine Learning (ML) SATD (Self-Admitted Technical Debt):

MACHINE LEARNING KNOWLEDGE (MLKnow): Machine-learning software carries plenty of unique challenges. Uneducated solutions by unaware developers may have to be revisited.
examples:
1. `TODO: Confirm whether layer-norm before attention actually boosts convergence for small-batch training on this dataset`
2. `TODO: Figure out if label smoothing distorts precision-recall when classes are highly imbalanced`
3. `XXX: Revisit our choice of cosine annealing—does it still make sense once we switch to AdamW?`

MODEL INTERPRETABILITY (MLInter): Machine-learning models are a black box, causing poor understanding of model functionality; this can lead to unknown behaviour.
examples:
1. `TODO: Generate SHAP summary plot so users can see feature impact`
2. `TODO: Log attention weights for each prediction to aid debugging`
3. `XXX: Provide saliency-map overlay in evaluation notebook for misclassified images`

MODEL CODE COMPREHENSION (MCC): Model code carries extra legibility concerns not present in traditional software (e.g., poorly-named temporary matrix variables).
examples:
1. `TODO: break up monolithic function train_eval_predict() into clearer, single-purpose helpers`
2. `FIXME: variable “h” is reused for both hidden state and Hessian—pick distinct, descriptive names`
3. `XXX: replace magic constant 512 (actually EMBED_SIZE) with a config parameter or enum`

DUPLICATE MODEL CODE (DMC): Code duplication frequently occurs in model code.
examples:
1. `TODO: remove copy-pasted ConvBlock definition—reuse common_layers.ConvBlock instead`
2. `FIXME: forward() in MyResNet duplicates logic from BaseResNet; refactor to call super()`
3. `XXX: data_augmentation_v2() is 95 % identical to data_augmentation(); consolidate into one helper`

DUPLICATE FEATURE EXTRACTION CODE (DFEC): Code duplication frequently occurs in feature-extraction code.
examples:
1. `TODO: merge extract_mfcc() and extract_mfcc_fast(); only differ in hop_length`
2. `FIXME: pixel_stats_v2() repeats almost all logic from pixel_stats(); refactor into shared util`
3. `XXX: text_features_v1 and text_features_v2 are copy-pasted—create one configurable pipeline`

WEIGHT CONFIGURATION (Weight Conf): Editing code that involves the weights of a model, or configuring a model's weights directly.
examples:
1. `TODO: expose weight_decay as configurable argument instead of hard-coding 0.01`
2. `FIXME: pretrained checkpoint loads fp16 weights—add auto-cast to ensure dtype consistency`
3. `XXX: add toggle to freeze first two Transformer layers during fine-tune via gradient mask`

LAYER CONFIGURATION (Layer Conf): Editing code that deals with model layers, or configuring layers themselves.
examples:
1. `TODO: swap BatchNorm for LayerNorm when depth > 24`
2. `FIXME: insert positional-embedding layer before first Transformer block`
3. `XXX: make dropout rate per-layer configurable instead of using global 0.1`

HYPER-PARAMETER CONFIGURATION (HP Conf): Configuring hyper-parameters of a model, or editing default values of off-the-shelf models.
examples:
1. `TODO: make learning_rate schedule (cosine vs. linear) selectable via CLI flag`
2. `FIXME: batch_size is hard-coded to 32; derive from GPU memory or user input`
3. `XXX: expose early_stopping_patience in config.yaml instead of magic value 10`

CODE DEPENDENCY (MLDepend): A needed change is blocked by an external dependency (library / subsystem).
examples:
1. `TODO: bump scikit-learn to ≥1.6 once conda forge wheels for arm64 become available`
2. `FIXME: delete custom tokenizer patch after huggingface/transformers merges PR #42812`
3. `XXX: keep legacy CUDA 11 build until upstream faiss releases official 12.x binaries`

GLUE CODE (GC): Supporting code written to interface with other code, inhibiting improvements due to dependent peculiarities.
examples:
1. `TODO: remove pandas-to-torch bridge once DataLoader accepts DataFrame directly`
2. `FIXME: adapter.py only exists to map old Predictor API to new one—rewrite consumers instead`
3. `XXX: consolidate fetch_from_s3_and_cache(); purely glue between boto3 and our local cache manager`

CUSTOM DATA TYPE (CDT): Using data types provided by general-purpose packages can cause extensive inter-operability burden.
examples:
1. `TODO: replace custom SparseTensorDict with standard torch.sparse_coo_tensor before exporting`
2. `FIXME: graph_utils.Node stores metadata in AttrDict; breaks ONNX serialization—use plain dict`
3. `XXX: pipeline returns dask DataFrame, but Keras expects NumPy array—add type-conversion layer`

MULTIPLE LANGUAGES (ML): Components written in other languages may introduce difficulties in ML development.
examples:
1. `TODO: port custom attention kernel from CUDA C++ to Metal for macOS deployment`
2. `FIXME: keep tokenizer logic in sync between Rust CLI and Python training script—generate bindings instead`
3. `XXX: consolidate duplicate data_cleaning.{scala,py}; two implementations drift out of spec easily`

UNNECESSARY MODEL CODE (UMC): Model code that bottlenecks execution, is unreachable / deprecated, or is unnecessary and should be removed.
examples:
1. `TODO: delete legacy compute_grad_stats(); superseded by torch.autograd.profiler`
2. `FIXME: remove flag use_old_preprocess once new pipeline is stable`
3. `XXX: drop debug print statements inside training loop before release`

DATA PROCESSING CONFIGURATION (Data Conf): Configuring how data is processed, by editing the data directly or adding new processing steps.
examples:
1. `TODO: add configurable pipeline step to down-sample majority class before batching`
2. `FIXME: current JSON parser drops NaNs—update to preserve missing values for later imputation`
3. `XXX: expose window_size and stride in config.yaml so sliding-window generator isn’t hard-coded`

PLAIN OLD DATA TYPE (PODT): Using raw data types in ML causes confusion when interpreting processes.
examples:
1. `TODO: replace raw list of logits with a NamedTuple so callers know the order`
2. `FIXME: function returns plain dict; define a TypedConfig dataclass instead for clarity`
3. `XXX: passing labels as bare NumPy array—wrap in LabelsTensor to preserve shape metadata`

DATA STORAGE CONFIGURATION (DS Conf): Configuring how data is represented internally or stored externally.
examples:
1. `TODO: migrate checkpoint format from .pth to safetensors for safer serialization`
2. `FIXME: results_writer currently appends CSV rows one by one—buffer and batch-flush to reduce I/O`
3. `XXX: store intermediate feature maps in mmap file to avoid exceeding GPU memory`

ABSTRACTION (Abs): Lack of abstractions in ML systems causes cascading changes when a component is modified.
examples:
1. `TODO: extract common scoring logic into BaseMetric so each model wrapper isn’t re-implementing it`
2. `FIXME: decouple data_loader from Trainer—should accept any iterable, not hard-coded DataLoader`
3. `XXX: replace direct calls to torch.cuda.* with a HardwareBackend interface to support TPU`

BOUNDARY EROSION (BE): Lack of boundaries between subsystems complicates maintenance and isolates changes poorly.
examples:
1. `TODO: migrate training-time logging out of Trainer class; should live in metrics.logger`
2. `FIXME: data_loader currently triggers augmentations inline—move to separate preprocess pipeline`
3. `XXX: remove direct DB writes from evaluate.py; expose results through results_api instead`

MODEL CODE MODIFIABILITY (MCM): Model code should enable easy maintenance and future modifications.
examples:
1. `TODO: move hyper-parameter defaults into config file so changing them doesn’t require code edits`
2. `FIXME: hard-coded embedding_dim (256) sprinkled across modules—define once in constants.py`
3. `XXX: split run_experiment.py into cli entrypoint and reusable Experiment class for easier reuse`

MODEL CODE REUSABILITY (MCR): Model code should be generalised for reuse in varying situations.
examples:
1. `TODO: abstract make_encoder() so VisionTransformer and ResNet builders share the same backbone factory`
2. `FIXME: refactor evaluate_batch() into mix-in class so both Trainer and HyperTune reuse identical logic`
3. `XXX: remove hard-coded Softmax loss; accept any callable loss_fn to support regression and ranking tasks`

PREDICTION QUALITY (PQ): Workflow changes may affect performance.
examples:
1. `TODO: rerun validation after enabling mixed-precision; confirm accuracy isn’t degraded`
2. `FIXME: measure F1 change when we swap tokenizer to WordPiece—may hurt rare-word recall`
3. `XXX: compare BLEU before/after removing dropout layer to ensure translation quality holds`

MACHINE LEARNING RELIABILITY (MLRelia): Model functionality depends on data quality; robustness measures should be in place.
examples:
1. `TODO: add sanity-check to reject examples with >20 % missing features before prediction`
2. `FIXME: wrap model call in try/except and fall back to last-known good output when decoder crashes`
3. `XXX: create robustness test suite with Gaussian noise and FGSM perturbations to monitor accuracy drift`

PROTOTYPE (Proto): Deploying small-scale prototypes into full systems can be dangerous.
examples:
1. `TODO: replace the single-threaded demo server with a production-ready Gunicorn setup`
2. `FIXME: temp JSON files used for state—migrate to proper Redis cache before rollout`
3. `XXX: remove hard-coded API key left from prototype; load securely from env vars instead`
""".strip()

# [R28 原代码保留]
# def build_prompt_fine(comment: str, coarse_abbr: str) -> str:
#     allowed = COARSE2FINE_ABBR.get(coarse_abbr, sorted(FINE_ABBR_SET))
#     tail = (
#         "Please determine the fine-grained ML SATD type. You MUST answer with only ONE abbreviation from:\n"
#         + ", ".join(allowed) + "\nReturn the abbreviation only (e.g., HP Conf)."
#     )
#     return f"{FINE_PROMPT_HEAD}\n\n{tail}\n\nCode Comment:\n\"\"\"\n{comment}\n\"\"\"\n\nAnswer:"
# [R28 修订]
def build_prompt_fine(comment: str, coarse_abbr: str) -> str:
    allowed = COARSE2FINE_ABBR.get(coarse_abbr, sorted(FINE_ABBR_SET))
    tail = (
        "Please determine the fine-grained ML SATD type. You MUST answer with only ONE abbreviation from:\n"
        + ", ".join(allowed)
        + f"\nReturn the abbreviation only (e.g., {allowed[0]})."
    )
    return f'{FINE_PROMPT_HEAD}\n\n{tail}\n\nCode Comment:\n"""\n{comment}\n"""\n\nAnswer:'

# ========== 5. 解析函数（健壮版） ==========
COARSE_REGEX = re.compile(r'\b(AWR|RDB|DCE|CFO|CDD|DTD|MDL|PRF|SCL)\b', re.I)
ABBR_SET = FINE_ABBR_SET
ABBR_LOWER = {a.lower(): a for a in ABBR_SET}
ABBR_REGEX = re.compile(r"\b(" + "|".join(re.escape(a) for a in ABBR_SET) + r")\b", re.I)

# [R16 原代码保留]
# def extract_coarse(raw: str) -> str:
#     """从任意首行文本里提取粗类缩写（容忍 '**AWR**' / 'Abbreviation: AWR' 等）。"""
#     # [R04 原代码保留]
#     # if not isinstance(raw, str):
#     #     return "<UNK>"
#     # [R04 修订]
#     if not isinstance(raw, str) or not raw.strip():
#         return "<UNK>"
#     s = raw.splitlines()[0].strip()
#
#     m = COARSE_REGEX.search(s)
#     if m:
#         return m.group(1).upper()
#
#     m = re.search(r'\*\*\s*(AWR|RDB|DCE|CFO|CDD|DTD|MDL|PRF|SCL)\s*\*\*', s, flags=re.I)
#     if m:
#         return m.group(1).upper()
#
#     letters = re.sub(r'[^A-Za-z]', '', s).lower()
#     for abbr in COARSE_LABELS:
#         if abbr.lower() in letters:
#             return abbr
#     return "<UNK>"
# [R16 修订]
def extract_coarse(raw: str) -> str:
    """Accept one complete canonical coarse abbreviation; empty/ambiguous output fails."""
    return parse_label(raw, {abbr: abbr for abbr in COARSE_LABELS}, allowed=COARSE_LABELS)["label"]

# [R16 原代码保留]
# def extract_fine(raw: str, allowed_abbr_list) -> str:
#     """从 raw 中提取细类缩写，仅允许 allowed_abbr_list 内的候选。兼容空格差异与 markdown。"""
#     # [R04 原代码保留]
#     # if not isinstance(raw, str):
#     #     return "<UNK>"
#     # [R04 修订]
#     if not isinstance(raw, str) or not raw.strip():
#         return "<UNK>"
#     s = raw.splitlines()[0].strip()
#
#     # 允许 “HP\s*Conf” 这种空白差异
#     pat_choices = [re.escape(x).replace(r'\ ', r'\s*') for x in allowed_abbr_list]
#     pat = r'\b(' + '|'.join(pat_choices) + r')\b'
#     m = re.search(pat, s, flags=re.I)
#     if m:
#         hit = re.sub(r'\s+', ' ', m.group(1)).strip().lower()
#         for a in allowed_abbr_list:
#             if re.sub(r'\s+', ' ', a).strip().lower() == hit:
#                 return a
#
#     # **HP Conf** 之类
#     m = re.search(r'\*\*\s*(' + '|'.join(pat_choices) + r')\s*\*\*', s, flags=re.I)
#     if m:
#         hit = re.sub(r'\s+', ' ', m.group(1)).strip().lower()
#         for a in allowed_abbr_list:
#             if re.sub(r'\s+', ' ', a).strip().lower() == hit:
#                 return a
#
#     # 兜底：letters-only 包含匹配
#     letters = re.sub(r'[^A-Za-z]', '', s).lower()
#     for a in allowed_abbr_list:
#         if re.sub(r'[^A-Za-z]', '', a).lower() in letters:
#             return a
#     return "<UNK>"
# [R16 修订]
def extract_fine(raw: str, allowed_abbr_list) -> str:
    """Normalize full names/abbreviations consistently and enforce the candidate set."""
    return parse_label(raw, FINE_LABEL_ABBR, allowed=allowed_abbr_list)["label"]

# ========== 6. 加载模型（沿用你之前的 pipeline 方式） ==========
print("[INFO] Loading model …")

pipe = pipeline(
    "text-generation",
    model=MODEL_PATH,
    torch_dtype="auto",
    device_map="auto",
)



@torch.inference_mode()
# [R23/R24/R29/R31 原代码保留]
# def generate_first_line(messages, max_new_tokens: int) -> str:
#     outputs = pipe(
#         messages,
#         max_new_tokens=max_new_tokens,
#     )
#     text = outputs[0]["generated_text"][-1]['content']
#     # print(text)
#     raw_answer = text.split('assistantfinal')[-1]
#     return raw_answer
# [R23/R24/R29/R31 修订]
def generate_first_line(messages, max_new_tokens: int, stage: str = "classification"):
    """Keep complete generated tokens and extract a validated native GPT-OSS final channel."""
    return audited_generate(
        pipe.model, pipe.tokenizer, messages,
        {"generation_config": getattr(pipe, "generation_config", pipe.model.generation_config),
         "max_new_tokens": max_new_tokens},
        audit=AUDIT, native_thinking=False,
        reasoning_effort=GPT_OSS_REASONING_EFFORT, stage=stage,
    )

# ========== 7. 两阶段推理 ==========
@torch.inference_mode()
# [R10/R16/R31/R33 原代码保留]
# def classify_two_stage(comment: str):
#     # Stage-1: Coarse
#     coarse_prompt = build_prompt_coarse(comment)
#     messages_coarse = [
#         {"role": "system", "content": "Return exactly one coarse abbreviation (AWR/RDB/DCE/CFO/CDD/DTD/MDL/PRF/SCL)."},
#         {"role": "user",   "content": coarse_prompt},
#     ]
#     coarse_raw = generate_first_line(messages_coarse, MAX_NEW_TOKENS)
#     coarse_abbr = extract_coarse(coarse_raw)
#
#     # Stage-2: Fine（限制在该粗类的候选集合）
#     allowed = COARSE2FINE_ABBR.get(coarse_abbr, sorted(FINE_ABBR_SET))
#     fine_prompt = build_prompt_fine(comment, coarse_abbr)
#     messages_fine = [
#         {"role": "system", "content": "Return exactly one fine abbreviation from the allowed candidates."},
#         {"role": "user",   "content": fine_prompt},
#     ]
#     fine_raw = generate_first_line(messages_fine, MAX_NEW_TOKENS)
#     fine_abbr = extract_fine(fine_raw, allowed)
#
#     if fine_abbr == "<UNK>":
#         fine_abbr = FALLBACK_FINE_BY_COARSE.get(coarse_abbr, "Data Conf")
#     print(fine_abbr)
#     return coarse_abbr, coarse_raw, fine_abbr, fine_raw
# [R10/R16/R31/R33 修订]
def classify_two_stage(comment: str):
    coarse_prompt = build_prompt_coarse(comment)
    coarse_result = generate_first_line([
        {"role": "system", "content": "Return exactly one coarse abbreviation (AWR/RDB/DCE/CFO/CDD/DTD/MDL/PRF/SCL)."},
        {"role": "user", "content": coarse_prompt},
    ], MAX_NEW_TOKENS, stage="coarse")
    coarse_parse = parse_label(
        coarse_result["answer_text"], {abbr: abbr for abbr in COARSE_LABELS}, allowed=COARSE_LABELS
    )
    coarse_abbr = coarse_parse["label"]
    # Preserve the original policy: an invalid coarse answer opens all fine candidates.
    allowed = COARSE2FINE_ABBR.get(coarse_abbr, sorted(FINE_ABBR_SET))
    fine_prompt = build_prompt_fine(comment, coarse_abbr)
    fine_result = generate_first_line([
        {"role": "system", "content": "Return exactly one fine abbreviation from the allowed candidates."},
        {"role": "user", "content": fine_prompt},
    ], MAX_NEW_TOKENS, stage="fine")
    fine_parse = parse_label(fine_result["answer_text"], FINE_LABEL_ABBR, allowed=allowed)
    fine_before_fallback = fine_parse["label"]
    fallback_used = not fine_parse["parse_ok"]
    # Retain the user-approved fallback as a policy output, with its failure visible.
    fine_abbr = (FALLBACK_FINE_BY_COARSE.get(coarse_abbr, "Data Conf")
                 if fallback_used else fine_before_fallback)
    return {
        "coarse_abbr": coarse_abbr,
        "coarse_parse_ok": coarse_parse["parse_ok"],
        "coarse_parse_status": coarse_parse["status"],
        "coarse_generation": coarse_result,
        "fine_abbr": fine_abbr,
        "fine_before_fallback": fine_before_fallback,
        "fine_parse_ok": fine_parse["parse_ok"],
        "fine_parse_status": fine_parse["status"],
        "fine_generation": fine_result,
        "fallback_used": fallback_used,
        "fallback_reason": ("fine_parse_" + fine_parse["status"]) if fallback_used else "",
        "allowed_fine": allowed,
    }

# ========== 8. 主流程 ==========
# [R10/R22/R31/R32/R33 原代码保留]
# df = pd.read_csv(CSV_PATH)
# rows = []
# unk_coarse_cnt = 0
# unk_fine_cnt   = 0
#
# print("[INFO] Classifying (two-stage, gpt-oss-20b) …")
# for cmt, gold in tqdm(zip(df["Comment Text"], df["ML TD Type"]),
#                       total=len(df), ncols=80):
#     coarse_abbr, coarse_raw, fine_abbr, fine_raw = classify_two_stage(str(cmt))
#
#     if coarse_abbr not in COARSE_ABBR_SET:
#         unk_coarse_cnt += 1
#     if fine_abbr not in FINE_ABBR_SET:
#         unk_fine_cnt += 1
#
#     rows.append({
#         "comment": cmt,
#         "real": FINE_ABBR2FULL.get(gold, gold),                 # 细类全称（真实）
#         "pred": FINE_ABBR2FULL.get(fine_abbr, fine_abbr),       # 细类全称（预测）
#         "coarse_pred": coarse_abbr,                              # 粗类缩写（预测）
#         "coarse_raw_output": coarse_raw,                         # 粗类原始输出
#         "fine_abbr": fine_abbr,                                  # 细类缩写（预测）
#         "raw_output": fine_raw,                                  # 细类原始输出
#     })
#
# pred_df = pd.DataFrame(rows)
#
# # ========== 9. 评估 ==========
# # 细（23）
# fine_report_text = classification_report(
#     pred_df["real"], pred_df["pred"],
#     labels=FINE_LABELS, zero_division=0, digits=3
# )
# print("\n[INFO] Fine-grained metrics (23):\n", fine_report_text)
#
# # 粗（9）
# pred_df["real_coarse"] = pred_df["real"].map(FINE2COARSE_FULL)
# pred_df["pred_coarse"] = pred_df["pred"].map(FINE2COARSE_FULL)
# coarse_report_text = classification_report(
#     pred_df["real_coarse"], pred_df["pred_coarse"],
#     labels=COARSE_LABELS, zero_division=0, digits=3
# )
# print("\n[INFO] Coarse-grained metrics (9):\n", coarse_report_text)
#
# # ========== 10. 保存 ==========
# pred_df.to_csv(RUN_DIR/"predictions.csv", index=False)
#
# pd.DataFrame(
#     confusion_matrix(pred_df["real"], pred_df["pred"], labels=FINE_LABELS),
#     index=FINE_LABELS, columns=FINE_LABELS
# ).to_csv(RUN_DIR/"confusion_matrix_fine.csv")
#
# pd.DataFrame(
#     confusion_matrix(pred_df["real_coarse"], pred_df["pred_coarse"], labels=COARSE_LABELS),
#     index=COARSE_LABELS, columns=COARSE_LABELS
# ).to_csv(RUN_DIR/"confusion_matrix_coarse.csv")
#
# (Path(RUN_DIR/"report_fine.txt")).write_text(fine_report_text, encoding="utf-8")
# (Path(RUN_DIR/"report_coarse.txt")).write_text(coarse_report_text, encoding="utf-8")
#
# (Path(RUN_DIR/"settings.json")).write_text(json.dumps({
#     "csv": CSV_PATH,
#     "model": MODEL_PATH,
#     "model_name": MODEL_NAME,
#     "run_dir": str(RUN_DIR),
#     "max_new_tokens": MAX_NEW_TOKENS,
#     "two_stage": True,
#     "unknown_coarse": unk_coarse_cnt,
#     "unknown_fine": unk_fine_cnt
# }, indent=2))
#
# print(f"\n[INFO] Files saved in {RUN_DIR}")
# print(f"[INFO] Unknown coarse predictions: {unk_coarse_cnt}")
# print(f"[INFO] Unknown fine predictions:   {unk_fine_cnt}")
#
# gc.collect(); torch.cuda.empty_cache()
# [R10/R22/R31/R32/R33 修订]
df = pd.read_csv(CSV_PATH)
rows = []
unk_coarse_cnt = 0
unk_fine_cnt = 0
print("[INFO] Classifying (two-stage; audited full prompts and explicit fallback state) …")
for source_index, source_row in tqdm(df.iterrows(), total=len(df), ncols=80):
    source_record = source_row.to_dict()
    AUDIT.set_sample(int(source_index), source_record)
    cmt = str(source_row["Comment Text"])
    gold = source_row["ML TD Type"]
    gold_full = FINE_ABBR2FULL.get(gold, gold)
    if gold_full not in FINE_LABELS:
        raise ValueError(f"Unknown gold label at source row {source_index}: {gold!r}")
    result = classify_two_stage(cmt)
    unk_coarse_cnt += int(not result["coarse_parse_ok"])
    unk_fine_cnt += int(not result["fine_parse_ok"])
    coarse_generation = result["coarse_generation"]
    fine_generation = result["fine_generation"]
    rows.append({
        **AUDIT.sample_fields(),
        "source_record_json": json.dumps(source_record, ensure_ascii=False, default=str),
        "raw_gold": gold,
        "comment": cmt,
        "real": gold_full,
        # Policy prediction remains the original primary evaluation convention.
        "pred": FINE_ABBR2FULL.get(result["fine_abbr"], result["fine_abbr"]),
        "pred_before_fallback": FINE_ABBR2FULL.get(result["fine_before_fallback"], "<UNK>"),
        "parse_ok": result["fine_parse_ok"],
        "parse_status": result["fine_parse_status"],
        "fallback_used": result["fallback_used"],
        "fallback_reason": result["fallback_reason"],
        "coarse_pred": result["coarse_abbr"],
        "stage1_coarse_pred": result["coarse_abbr"],
        "coarse_parse_ok": result["coarse_parse_ok"],
        "coarse_parse_status": result["coarse_parse_status"],
        "coarse_raw_output": coarse_generation["raw_text"],
        "coarse_answer_output": coarse_generation["answer_text"],
        "coarse_generation_parse_status": coarse_generation["parse_status"],
        "fine_abbr": result["fine_abbr"],
        "fine_abbr_before_fallback": result["fine_before_fallback"],
        "allowed_fine_json": json.dumps(result["allowed_fine"], ensure_ascii=False),
        "raw_output": fine_generation["raw_text"],
        "answer_output": fine_generation["answer_text"],
        "generation_parse_status": fine_generation["parse_status"],
        "generation_calls_json": json.dumps(AUDIT.events, ensure_ascii=False, default=str),
    })

pred_df = pd.DataFrame(rows)
pred_df["real_coarse"] = pred_df["real"].map(FINE2COARSE_FULL)
pred_df["final_fine_aggregated_coarse"] = pred_df["pred"].map(FINE2COARSE_FULL)
# Preserve the legacy alias but explicitly identify what it measures.
pred_df["pred_coarse"] = pred_df["final_fine_aggregated_coarse"]

fine_report_text = classification_report(
    pred_df["real"], pred_df["pred"], labels=FINE_LABELS, zero_division=0, digits=3
)
coarse_report_text = classification_report(
    pred_df["real_coarse"], pred_df["final_fine_aggregated_coarse"],
    labels=COARSE_LABELS, zero_division=0, digits=3,
)
stage1_coarse_report_text = classification_report(
    pred_df["real_coarse"], pred_df["stage1_coarse_pred"],
    labels=COARSE_LABELS, zero_division=0, digits=3,
)
print("\n[INFO] Fine policy metrics (23):\n", fine_report_text)
print("\n[INFO] Final fine aggregated coarse metrics (9):\n", coarse_report_text)
print("\n[INFO] Stage-1 coarse metrics (9; invalid answers count as errors):\n", stage1_coarse_report_text)

pred_df.to_csv(RUN_DIR / "predictions.csv", index=False)
pd.DataFrame(
    confusion_matrix(pred_df["real"], pred_df["pred"], labels=FINE_LABELS),
    index=FINE_LABELS, columns=FINE_LABELS,
).to_csv(RUN_DIR / "confusion_matrix_fine.csv")
pd.DataFrame(
    confusion_matrix(pred_df["real_coarse"], pred_df["final_fine_aggregated_coarse"], labels=COARSE_LABELS),
    index=COARSE_LABELS, columns=COARSE_LABELS,
).to_csv(RUN_DIR / "confusion_matrix_coarse.csv")
# Include the explicit failure column so failed stage-1 predictions cannot disappear.
stage1_matrix_labels = COARSE_LABELS + ["<UNK>"]
pd.DataFrame(
    confusion_matrix(pred_df["real_coarse"], pred_df["stage1_coarse_pred"], labels=stage1_matrix_labels),
    index=stage1_matrix_labels, columns=stage1_matrix_labels,
).to_csv(RUN_DIR / "confusion_matrix_stage1_coarse.csv")
(RUN_DIR / "report_fine.txt").write_text(fine_report_text, encoding="utf-8")
(RUN_DIR / "report_coarse.txt").write_text(coarse_report_text, encoding="utf-8")
(RUN_DIR / "report_coarse_final_aggregated.txt").write_text(coarse_report_text, encoding="utf-8")
(RUN_DIR / "report_coarse_stage1.txt").write_text(stage1_coarse_report_text, encoding="utf-8")
# Additional metrics never replace the retained policy reports above.
write_failure_metrics(pred_df, FINE_LABELS, RUN_DIR, prefix="failure_as_error_fine")
write_failure_metrics(
    pred_df, COARSE_LABELS, RUN_DIR,
    pred_col="stage1_coarse_pred", gold_col="real_coarse", parse_ok_col="coarse_parse_ok",
    prefix="failure_as_error_stage1_coarse",
)
settings = {'csv': CSV_PATH, 'model': MODEL_PATH, 'model_name': MODEL_NAME, 'run_dir': str(RUN_DIR), 'max_new_tokens': MAX_NEW_TOKENS, 'two_stage': True, 'unknown_coarse': unk_coarse_cnt, 'unknown_fine': unk_fine_cnt}
settings.update({
    "runtime_audit": AUDIT.settings(),
    "seed": SEED,
    "script": str(Path(__file__).resolve()),
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    # [R24 原代码保留]
    # "input_length_policy": "full untruncated chat; actual model context minus original output cap and shared safety reserve; over-budget inputs are recorded failures",
    # [R24 修订] The configured output maximum is an upper bound, not a fixed full reservation.
    "input_length_policy": "count full untruncated chat first; effective output cap=min(requested cap, actual context-input tokens-64); record requested/effective caps and adjustment; fail only when no output space remains",
    "coarse_max_input_tokens": None,
    "fine_max_input_tokens": None,
    "legacy_stage_input_caps": None,
    "primary_metric_policy": "original fallback prediction retained in pred/report_fine/report_coarse; failure-as-error is additional",
    "coarse_report_policy": "report_coarse is final_fine_aggregated_coarse; report_coarse_stage1 independently evaluates stage1_coarse_pred",
    "fallback_count": int(pred_df["fallback_used"].sum()),
    "historical_results": "unchanged; no re-evaluation executed by this revision",
})
settings["gpt_oss_reasoning_effort"] = GPT_OSS_REASONING_EFFORT
(RUN_DIR / "settings.json").write_text(json.dumps(settings, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
print(f"\n[INFO] Files saved in {RUN_DIR}")
print(f"[INFO] Unknown coarse predictions before fallback: {unk_coarse_cnt}")
print(f"[INFO] Unknown fine predictions before fallback: {unk_fine_cnt}")
gc.collect()
torch.cuda.empty_cache()
