#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_satd_qwen3_yn_polling.py
----------------------------
模型：/data/yangzhifei/pro/ML_SATD/models/Qwen3-14B
方法：对每条评论，对 23 标签逐一做 Yes/No，汇总 Yes 候选；如有多个，再让模型在候选中只选 1 个缩写；
      若没有 Yes，则做一次全标签强制单标签输出；仍解析失败则回退到 FALLBACK_ABBR。
输出：results/run-YYYYmmdd-HHMMSS-模型名 / predictions.csv, confusion_matrix.csv, report.csv, settings.json
注意：raw_output 只写缩写；predictions.csv 列顺序固定为 comment, real, pred, raw_output, reason（reason 为第 5 列）。
"""

import re, os, gc, json
from pathlib import Path
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# -------- 0. 基础路径（按需修改） --------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # ← 改 GPU 号
CSV_PATH   = "/data/yangzhifei/pro/ML_SATD/dataset/sampled_label_dataset_normalized_rmvtodo.csv"  # ← 改数据
MODEL_PATH = "/data/yangzhifei/pro/ML_SATD/models/Qwen3-8B"                                      # ← 改模型

# 目录名包含时间戳 + 模型名
MODEL_NAME = Path(MODEL_PATH).name
# [R09-R44 原代码保留]
# RUN_DIR    = Path("results") / f"run-{datetime.now():%Y%m%d-%H%M%S}-{MODEL_NAME}-ynpoll"
# [R09-R44 修订]
import sys
import uuid
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from revision_runtime import RunAudit, audited_generate, parse_label, visible_answer, row_identity, write_failure_metrics
REVISION_SEED = int(os.environ.get('SATD_SEED', '42'))
# [R31 修订] 将新产物固定在修订目录，避免依赖启动工作目录。
# RUN_DIR = Path('results') / f'run-{datetime.now():%Y%m%d-%H%M%S}-{MODEL_NAME}-ynpoll'
RUN_DIR = Path(__file__).resolve().parents[1] / "revised_outputs" / f'run-{datetime.now():%Y%m%d-%H%M%S}-{MODEL_NAME}-ynpoll'
RUN_DIR = RUN_DIR.with_name(RUN_DIR.name + '-revised-' + datetime.now().strftime('%f') + '-' + uuid.uuid4().hex[:8])
# [R09-R44 原代码保留]
# RUN_DIR.mkdir(parents=True, exist_ok=True)
# [R09-R44 修订]
RUN_DIR.mkdir(parents=True, exist_ok=False)
audit = RunAudit(RUN_DIR, MODEL_PATH, CSV_PATH, seed=REVISION_SEED)

# -------- 1. 其它参数（可按需微调；不改变采样策略） --------
PRINT_UNPARSED       = True
FALLBACK_ABBR        = "Data Conf"
MAX_NEW_TOKENS_YN    = 2048      # Yes/No 轮询
MAX_NEW_TOKENS_CHOOSE= 2048      # 候选内裁决
MAX_NEW_TOKENS_FULL  = 2048     # 全标签强制单标签
MAX_NEW_TOKENS_REASON= 96      # 简短 reason（单句）
TEMPERATURE          = 0.6
TOP_P                = 0.95
TOP_K                = 20

# ----------------- 标签映射 -----------------
LABEL_ABBR = {
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
ABBR2FULL  = {v:k for k,v in LABEL_ABBR.items()}
FULL2ABBR  = LABEL_ABBR
LABELS     = list(LABEL_ABBR.keys())
ABBR_SET   = set(ABBR2FULL.keys())
ABBR_LOWER = {a.lower():a for a in ABBR_SET}

# ============ 2. 23 类定义（用于判别上下文） ============
PROMPT_HEAD = r"""
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

# [R06 原代码保留] 原共享尾段混合推理规则和统一缩写输出要求。
# PROMPT_TAIL = f"""
# You are a strict ML-SATD classifier.
# Do all reasoning internally and do NOT reveal it.
# Decide the single best label by following this internal decision process, then output exactly ONE abbreviation on a single line with nothing else.
# 
# Internal decision process (for your internal use only; do not include in output):
# 1) Identify the primary signal in the comment and select candidates accordingly:
#    • Data-related: processing/pipeline steps ⇒ {{Data Conf}}; raw/plain data types ⇒ {{PODT}}; representation/storage/schema ⇒ {{DS Conf}}.
#    • Dependency & integration: external/library blocking ⇒ {{MLDepend}}; glue/integration shims ⇒ {{GC}}; custom data types ⇒ {{CDT}}; mixed languages ⇒ {{ML}}; unnecessary/dead/deprecated model code ⇒ {{UMC}}.
#    • Configuration & parameters: weights ⇒ {{Weight Conf}}; layers/topology ⇒ {{Layer Conf}}; hyper-parameters/defaults ⇒ {{HP Conf}}.
#    • Structure & evolvability: missing abstraction ⇒ {{Abs}}; boundary erosion ⇒ {{BE}}; ease of modifying model code ⇒ {{MCM}}; generality/reuse ⇒ {{MCR}}.
#    • Duplication: model code duplication ⇒ {{DMC}}; feature-extraction duplication ⇒ {{DFEC}}.
#    • Understanding & interpretability: lack of ML knowledge ⇒ {{MLKnow}}; model black-box / explainability concerns ⇒ {{MLInter}}.
#    • Performance & robustness: prediction quality/accuracy ⇒ {{PQ}}; reliability/robustness/data-quality guards ⇒ {{MLRelia}}.
#    • Prototype: prototype used beyond its scope ⇒ {{Proto}}.
# 2) If multiple candidates match, choose the one that represents the main maintenance obstacle implied by the comment.
# 3) If still tied, prefer the most specific label whose remediation is the most localized (e.g., pick {{Layer Conf}} over {{HP Conf}} if the text explicitly mentions layers).
# 4) Output policy: return exactly ONE abbreviation on a single line, with no other text.
#    Allowed abbreviations:
#    {', '.join(sorted(ABBR_SET))}
# """.strip()
# [R06 修订] 共享段仅保留推理规则；输出格式由各阶段提示单独指定。
# [R09-R44 原代码保留]
# REASONING_RULES = f"""
# You are a strict ML-SATD classifier.
# Do all reasoning internally and do NOT reveal it.
# Use the following internal decision process to assess category evidence for the current task.
# Follow the task-specific response format provided below.
# 
# Internal decision process (for your internal use only; do not include in output):
# 1) Identify the primary signal in the comment and select candidates accordingly:
#    • Data-related: processing/pipeline steps ⇒ {{Data Conf}}; raw/plain data types ⇒ {{PODT}}; representation/storage/schema ⇒ {{DS Conf}}.
#    • Dependency & integration: external/library blocking ⇒ {{MLDepend}}; glue/integration shims ⇒ {{GC}}; custom data types ⇒ {{CDT}}; mixed languages ⇒ {{ML}}; unnecessary/dead/deprecated model code ⇒ {{UMC}}.
#    • Configuration & parameters: weights ⇒ {{Weight Conf}}; layers/topology ⇒ {{Layer Conf}}; hyper-parameters/defaults ⇒ {{HP Conf}}.
#    • Structure & evolvability: missing abstraction ⇒ {{Abs}}; boundary erosion ⇒ {{BE}}; ease of modifying model code ⇒ {{MCM}}; generality/reuse ⇒ {{MCR}}.
#    • Duplication: model code duplication ⇒ {{DMC}}; feature-extraction duplication ⇒ {{DFEC}}.
#    • Understanding & interpretability: lack of ML knowledge ⇒ {{MLKnow}}; model black-box / explainability concerns ⇒ {{MLInter}}.
#    • Performance & robustness: prediction quality/accuracy ⇒ {{PQ}}; reliability/robustness/data-quality guards ⇒ {{MLRelia}}.
#    • Prototype: prototype used beyond its scope ⇒ {{Proto}}.
# 2) If multiple candidates match, choose the one that represents the main maintenance obstacle implied by the comment.
# 3) If still tied, prefer the most specific label whose remediation is the most localized (e.g., pick {{Layer Conf}} over {{HP Conf}} if the text explicitly mentions layers).
# """.strip()
# [R09-R44 修订]
REASONING_RULES = f"""
You are a strict ML-SATD classifier.
Do all reasoning internally and do NOT reveal it.
Use the following internal decision process to assess category evidence for the current task.
Follow the task-specific response format provided below.

Internal decision process (for your internal use only; do not include in output):
1) Identify the primary signal in the comment and select candidates accordingly:
   • Data-related: processing/pipeline steps ⇒ {{Data Conf}}; raw/plain data types ⇒ {{PODT}}; representation/storage/schema ⇒ {{DS Conf}}.
   • Dependency & integration: external/library blocking ⇒ {{MLDepend}}; glue/integration shims ⇒ {{GC}}; custom data types ⇒ {{CDT}}; mixed languages ⇒ {{ML}}; unnecessary/dead/deprecated model code ⇒ {{UMC}}.
   • Configuration & parameters: weights ⇒ {{Weight Conf}}; layers/topology ⇒ {{Layer Conf}}; hyper-parameters/defaults ⇒ {{HP Conf}}.
   • Structure & evolvability: missing abstraction ⇒ {{Abs}}; boundary erosion ⇒ {{BE}}; ease of modifying model code ⇒ {{MCM}}; generality/reuse ⇒ {{MCR}}.
   • Duplication: model code duplication ⇒ {{DMC}}; feature-extraction duplication ⇒ {{DFEC}}.
   • Understanding & interpretability: lack of ML knowledge ⇒ {{MLKnow}}; model black-box / explainability concerns ⇒ {{MLInter}}; unclear model-code names, readability or understanding matrix/tensor code ⇒ {{MCC}}.
   • Performance & robustness: prediction quality/accuracy ⇒ {{PQ}}; reliability/robustness/data-quality guards ⇒ {{MLRelia}}.
   • Prototype: prototype used beyond its scope ⇒ {{Proto}}.
2) If multiple candidates match, choose the one that represents the main maintenance obstacle implied by the comment.
3) If still tied, prefer the most specific label whose remediation is the most localized (e.g., pick {{Layer Conf}} over {{HP Conf}} if the text explicitly mentions layers).
""".strip()
# ============ 3. 提示构造（严格输出格式，禁止外露思考） ============

# 3.1 对“单个标签”的 Yes/No 轮询
def build_yn_prompt(comment: str, full_label: str, abbr: str) -> str:
    return (
        f"{PROMPT_HEAD}\n\n"
        # [R06 原代码保留]
        # f"{PROMPT_TAIL}\n\n"
        # [R06 修订] 当前阶段的输出指令独立保留在下方。
        f"{REASONING_RULES}\n\n"
        f"Task: Decide if the following Code Comment expresses the SATD type **{full_label} ({abbr})**.\n"
        "Answer strictly with ONE word: Yes or No. Do all reasoning internally and do NOT reveal it.\n\n"
        "Code Comment:\n\"\"\"\n"
        f"{comment}\n\"\"\"\n\n"
        "Answer:"
    )

# 3.2 在候选集合中“只选一个缩写”
def build_choose_prompt(comment: str, candidates_abbr: list) -> str:
    cand_str = ", ".join(candidates_abbr)
    return (
        f"{PROMPT_HEAD}\n\n"
        # [R06 原代码保留]
        # f"{PROMPT_TAIL}\n\n"
        # [R06 修订] 当前阶段的输出指令独立保留在下方。
        f"{REASONING_RULES}\n\n"
        "Task: The following abbreviations have been marked as plausible candidates.\n"
        f"Choose exactly ONE abbreviation from this set and output that abbreviation ONLY on a single line:\n"
        f"{cand_str}\n"
        "Do all reasoning internally and do NOT reveal it.\n\n"
        "Code Comment:\n\"\"\"\n"
        f"{comment}\n\"\"\"\n\n"
        "Abbreviation:"
    )

# 3.3 全标签强制单标签输出（兜底）
def build_full_prompt(comment: str) -> str:
    return (
        f"{PROMPT_HEAD}\n\n"
        # [R06 原代码保留]
        # f"{PROMPT_TAIL}\n\n"
        # [R06 修订] 当前阶段的输出指令独立保留在下方。
        f"{REASONING_RULES}\n\n"
        "Please determine which type of ML SATD the following Code Comment belongs to.\n"
        "Return exactly ONE abbreviation on a single line, with no extra text, chosen from:\n"
        f"{', '.join(sorted(ABBR_SET))}\n\n"
        "Code Comment:\n\"\"\"\n"
        f"{comment}\n\"\"\"\n\n"
        "Abbreviation:"
    )

# 3.4 简短理由（单句）
def build_reason_prompt(comment: str, final_abbr: str) -> str:
    full = ABBR2FULL.get(final_abbr, final_abbr)
    return (
        "You already chose a label.\n"
        "Reply with exactly one line beginning with 'REASON:' followed by a short justification (<= 30 words), "
        "grounded in the comment content. Do not output the label name.\n\n"
        f"Chosen: {final_abbr} ({full})\n"
        "Comment:\n\"\"\"\n"
        f"{comment}\n\"\"\"\n"
    )

# ============ 4. 解析工具 ============

# [R09-R44 原代码保留]
# def _visible_only(txt: str) -> str:
#     if not isinstance(txt, str):
#         return ""
#     # 去除 <think>…</think>（如有）
#     return re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL | re.IGNORECASE).strip()
# [R09-R44 修订]
def _visible_only(text: str) -> str:
    return visible_answer(text, native_thinking=False)

# [R08 原代码保留]
# def parse_yesno(text: str) -> str:
#     vis = _visible_only(text)
#     first = vis.splitlines()[0].strip() if vis else ""
#     token = re.sub(r"[^a-z]", "", first.lower())
#     if token.startswith("yes"):
#         return "yes"
#     if token.startswith("no"):
#         return "no"
#     # 宽松匹配
#     if re.search(r"\byes\b", vis, flags=re.I):
#         return "yes"
#     if re.search(r"\bno\b", vis, flags=re.I):
#         return "no"
#     return "unknown"
# [R08 修订]
# [R09-R44 原代码保留]
# def parse_yesno(text: str) -> str:
#     if not isinstance(text, str):
#         return 'unknown'
#     answer = _visible_only(text).strip()
#     answer = re.sub('(^[*_`]+|[*_`]+$)', '', answer).strip()
#     match = re.fullmatch('(yes|no|y|n)[.!。！]?', answer, flags=re.I)
#     if match is None:
#         return 'unknown'
#     token = match.group(1).lower()
#     if token in ('yes', 'y'):
#         return 'yes'
#     return 'no'
# [R09-R44 修订]
def extract_yes_no(text: str) -> str:
    if not isinstance(text, str):
        return 'unknown'
    answer = re.sub(r'(^[*_`]+|[*_`]+$)', '', text.strip()).strip()
    match = re.fullmatch(r'(yes|no|y|n)[.!。！]?', answer, flags=re.I)
    if match is None:
        return 'unknown'
    return 'yes' if match.group(1).lower() in ('yes', 'y') else 'no'


def parse_yesno(text: str) -> str:
    return extract_yes_no(text)

# [R09-R44 原代码保留]
# def parse_abbr(text: str) -> str:
#     vis = _visible_only(text)
#     line = vis.splitlines()[0].strip() if vis else ""
#     # 允许带前缀 “Abbreviation:”
#     line = re.sub(r'^\s*(Abbreviation|Label)\s*:\s*', '', line, flags=re.I).strip()
#     # 规范化检测
#     token = re.sub(r"[^a-z]", "", line.lower())
#     if token in ABBR_LOWER: return ABBR_LOWER[token]
#     if line in ABBR2FULL:   return line
#     join = line.replace(" ", "")
#     if join.lower() in ABBR_LOWER: return ABBR_LOWER[join.lower()]
#     # 在整段中宽松找
#     for ab in ABBR_SET:
#         pat = r'\b' + re.sub(r'\s+', r'\\s*', re.escape(ab)) + r'\b'
#         if re.search(pat, vis, flags=re.I):
#             return ab
#     return "<UNK>"
# [R09-R44 修订]
def parse_abbr(text: str) -> str:
    return parse_label(text, LABEL_ABBR, native_thinking=False)['label']

def parse_reason(text: str) -> str:
    vis = _visible_only(text)
    m = re.search(r"(?:^|\n)\s*REASON\s*[:：]\s*(.+)", vis, flags=re.I)
    if m:
        line = m.group(1).strip()
    else:
        # 退化为第一句
        line = vis.splitlines()[0].strip() if vis else ""
    # 截到一句话
    line = re.split(r'(?<=[.。!?])\s+', line)[0].strip()
    return line[:300]

# ============ 5. 加载模型 ============

print("[INFO] Loading model …")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# [R09-R44 原代码保留]
# def _gen(prompt: str, max_new_tokens: int):
#     messages = [{"role": "user", "content": prompt}]
#     text = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True,
#         enable_thinking=True  # 允许模型内部推理，但我们只取可见输出
#     )
#     inputs = tokenizer(text, return_tensors="pt",
#                        truncation=True, max_length=4096).to(model.device)
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=max_new_tokens,
#         temperature=TEMPERATURE,
#         top_p=TOP_P,
#         top_k=TOP_K,
#     )
#     out = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
#                            skip_special_tokens=True)
#     # visible = out.split('</think>')[-1].strip()
#     # print(visible)
#     return out.strip()
# [R09-R44 修订]
def _gen(prompt, max_new_tokens: int, thinking=True, stage='classification'):
    messages = prompt if isinstance(prompt, list) else [{'role': 'user', 'content': prompt}]
    event = audited_generate(
        model, tokenizer, messages, {'max_new_tokens': max_new_tokens, 'temperature': TEMPERATURE, 'top_p': TOP_P, 'top_k': TOP_K},
        audit=audit, max_input_tokens=4096, native_thinking=thinking,
        reasoning_effort=None, stage=stage,
    )
    # Full raw response and observed stopping evidence are persisted by RunAudit.
    return event.get('answer_text', '')

# ============ 6. 主流程 ============

# [R09-R44 原代码保留]
# [R09-R44 修订]
# [R09-R44 修订] Shared staged classification with complete diagnostics.
def classify_by_polling(comment: str, diagnostics=None):
    """Preserve the three-value interface; audit failures before Data Conf substitution."""
    yes_candidates, unknown_labels, votes = [], [], []
    fallback_reason, selection_route = '', 'single_positive'
    for full, abbr in LABEL_ABBR.items():
        prompt = build_yn_prompt(comment, full, abbr)
        messages = [{'role': 'user', 'content': prompt}]
        answer = _gen(messages, MAX_NEW_TOKENS_YN, stage='yn:' + abbr)
        yn = extract_yes_no(answer)
        event = audit.events[-1] if audit.events else {}
        votes.append({
            'label': abbr,
            'parsed': yn,
            'raw_output': event.get('raw_text', ''),
            'generation_status': event.get('parse_status', 'unavailable'),
            'stop_status': event.get('stop_status', 'unavailable'),
        })
        if yn == 'yes':
            yes_candidates.append(abbr)
        elif yn != 'no':
            unknown_labels.append(abbr)
    raw_final = ''
    raw_pred = '<UNK>'
    if not unknown_labels and len(yes_candidates) == 1:
        raw_pred = yes_candidates[0]
    elif not unknown_labels and len(yes_candidates) >= 2:
        selection_route = 'candidate_arbitration'
        pick_prompt = build_choose_prompt(comment, yes_candidates)
        answer = _gen([{'role': 'user', 'content': pick_prompt}], MAX_NEW_TOKENS_CHOOSE, stage='candidate_arbitration')
        parsed = parse_label(answer, LABEL_ABBR, allowed=yes_candidates, native_thinking=False)
        raw_pred = parsed['label']
        raw_final = audit.events[-1].get('raw_text', '') if audit.events else ''
        if not parsed['parse_ok']:
            fallback_reason = 'candidate_arbitration_invalid'
    else:
        selection_route = 'full_label_unresolved_votes' if unknown_labels else 'full_label_no_positive'
        fallback_reason = 'unresolved_votes_full_label' if unknown_labels else 'no_positive_votes_full_label'
        full_prompt = build_full_prompt(comment)
        answer = _gen([{'role': 'user', 'content': full_prompt}], MAX_NEW_TOKENS_FULL, stage='full_label_selection')
        parsed = parse_label(answer, LABEL_ABBR, native_thinking=False)
        raw_pred = parsed['label']
        raw_final = audit.events[-1].get('raw_text', '') if audit.events else ''
        if not parsed['parse_ok']:
            fallback_reason += ';full_label_parse_failed'
    parse_ok = raw_pred in ABBR_SET
    final_abbr = raw_pred if parse_ok else FALLBACK_ABBR
    if diagnostics is not None:
        diagnostics.update({
            'unknown_labels': list(unknown_labels),
            'fallback_reason': fallback_reason,
            'selection_route': selection_route,
            'raw_pred': raw_pred,
            'parse_ok': parse_ok,
            'parse_status': 'ok' if parse_ok else 'final_label_unparsed',
            'fallback_used': not parse_ok,
            'yn_unparsed_count': len(unknown_labels),
            'yn_decisions': votes,
        })
    return final_abbr, raw_final, yes_candidates
df = pd.read_csv(CSV_PATH)

rows = []
unk_cnt = 0
gold_unmapped = 0

print("[INFO] Y/N polling and classifying …")

# [R08 原代码保留]
# for cmt, gold in tqdm(zip(df["Comment Text"], df["ML TD Type"]),
#                       total=len(df), ncols=90):
#
#     # --- 6.1 逐标签 Y/N ---
#     yes_candidates = []
#     yn_raw_all = []
#
#     for full, abbr in LABEL_ABBR.items():
#         yn_prompt = build_yn_prompt(str(cmt), full, abbr)
#         yn_out    = _gen(yn_prompt, MAX_NEW_TOKENS_YN)
#         yn_raw_all.append((abbr, yn_out))
#         ans = parse_yesno(yn_out)
#         if ans == "yes":
#             yes_candidates.append(abbr)
#
#     # --- 6.2 得到最终缩写 ---
#     final_abbr = None
#     final_raw  = ""
#
#     if len(yes_candidates) == 1:
#         final_abbr = yes_candidates[0]
#         final_raw  = final_abbr  # raw_output 只写缩写
#     elif len(yes_candidates) > 1:
#         choose_prompt = build_choose_prompt(str(cmt), yes_candidates)
#         choose_out    = _gen(choose_prompt, MAX_NEW_TOKENS_CHOOSE)
#         picked        = parse_abbr(choose_out)
#         if picked in ABBR_SET and picked in set(yes_candidates):
#             final_abbr = picked
#             final_raw  = picked
#         else:
#             # 未能在候选内解析，回退到第一候选
#             final_abbr = yes_candidates[0]
#             final_raw  = final_abbr
#     else:
#         # 没有任何 Yes，做一次全标签强制单标签输出
#         full_prompt = build_full_prompt(str(cmt))
#         full_out    = _gen(full_prompt, MAX_NEW_TOKENS_FULL)
#         picked      = parse_abbr(full_out)
#         if picked in ABBR_SET:
#             final_abbr = picked
#             final_raw  = picked
#         else:
#             final_abbr = FALLBACK_ABBR
#             final_raw  = final_abbr
#             unk_cnt += 1
#             if PRINT_UNPARSED:
#                 with open(RUN_DIR/"unparsed.txt", "a", encoding="utf-8") as f:
#                     f.write(f"[FULL FALLBACK] {full_out}\n")
#
#     # --- 6.3 生成简短 reason（单句，放第 5 列；可为空但尽量非空） ---
#     reason_prompt = build_reason_prompt(str(cmt), final_abbr)
#     reason_out    = _gen(reason_prompt, MAX_NEW_TOKENS_REASON)
#     reason        = parse_reason(reason_out)
#     if not reason:
#         reason = "Model-generated brief justification."
#     print(final_raw)
#     rows.append({
#         "comment": str(cmt),
#         "real": ABBR2FULL.get(gold, gold),           # 保持真实“全称”
#         "pred": ABBR2FULL.get(final_abbr, final_abbr), # 输出预测“全称”
#         "raw_output": final_raw,                     # 只写缩写
#         "reason": reason,                            # 第五列
#     })
# [R08 修订]
# [R09-R44 原代码保留]
# for (cmt, gold) in tqdm(zip(df['Comment Text'], df['ML TD Type']), total=len(df), ncols=90):
#     yes_candidates = []
#     unknown_labels = []
#     fallback_reason = ''
#     yn_raw_all = []
#     for (full, abbr) in LABEL_ABBR.items():
#         yn_prompt = build_yn_prompt(str(cmt), full, abbr)
#         yn_out = _gen(yn_prompt, MAX_NEW_TOKENS_YN)
#         yn_raw_all.append((abbr, yn_out))
#         ans = parse_yesno(yn_out)
#         if ans == 'yes':
#             yes_candidates.append(abbr)
#         elif ans not in ('yes', 'no'):
#             unknown_labels.append(abbr)
#     final_abbr = None
#     final_raw = ''
#     if not unknown_labels and len(yes_candidates) == 1:
#         final_abbr = yes_candidates[0]
#         final_raw = final_abbr
#     elif not unknown_labels and len(yes_candidates) > 1:
#         choose_prompt = build_choose_prompt(str(cmt), yes_candidates)
#         choose_out = _gen(choose_prompt, MAX_NEW_TOKENS_CHOOSE)
#         picked = parse_abbr(choose_out)
#         if picked in ABBR_SET and picked in set(yes_candidates):
#             final_abbr = picked
#             final_raw = picked
#         else:
#             final_abbr = FALLBACK_ABBR
#             final_raw = final_abbr
#             fallback_reason = 'candidate_arbitration_invalid'
#     else:
#         fallback_reason = 'unresolved_votes_full_label' if unknown_labels else 'no_positive_votes_full_label'
#         full_prompt = build_full_prompt(str(cmt))
#         full_out = _gen(full_prompt, MAX_NEW_TOKENS_FULL)
#         picked = parse_abbr(full_out)
#         if picked in ABBR_SET:
#             final_abbr = picked
#             final_raw = picked
#         else:
#             final_abbr = FALLBACK_ABBR
#             final_raw = final_abbr
#             unk_cnt += 1
#             if PRINT_UNPARSED:
#                 with open(RUN_DIR / 'unparsed.txt', 'a', encoding='utf-8') as f:
#                     f.write(f'[FULL FALLBACK] {full_out}\n')
#             fallback_reason += ';full_label_parse_failed'
#     reason_prompt = build_reason_prompt(str(cmt), final_abbr)
#     reason_out = _gen(reason_prompt, MAX_NEW_TOKENS_REASON)
#     reason = parse_reason(reason_out)
#     if not reason:
#         reason = 'Model-generated brief justification.'
#     print(final_raw)
#     poll_diagnostics = {
#         'unknown_labels': list(unknown_labels),
#         'fallback_reason': fallback_reason,
#     }
#     rows.append({
#         'comment': str(cmt),
#         'real': ABBR2FULL.get(gold, gold),
#         'pred': ABBR2FULL.get(final_abbr, final_abbr),
#         'raw_output': final_raw,
#         'reason': reason,
#         'unknown_labels': ';'.join(poll_diagnostics.get('unknown_labels', [])),
#         'fallback_reason': poll_diagnostics.get('fallback_reason', ''),
#     })
# [R09-R44 修订]
for source_row, (cmt, gold) in enumerate(tqdm(zip(df['Comment Text'], df['ML TD Type']), total=len(df), ncols=90)):
    audit.set_sample(source_row, df.iloc[source_row].to_dict())
    gold_result = parse_label(str(gold), LABEL_ABBR, native_thinking=False)
    if not gold_result['parse_ok']:
        raise ValueError(f'Unmapped gold label at source row {source_row}: {gold!r}')
    poll_diagnostics = {}
    pred_abbr, raw_choice, yes_list = classify_by_polling(str(cmt), diagnostics=poll_diagnostics)
    if not poll_diagnostics['parse_ok']:
        unk_cnt += 1
        if PRINT_UNPARSED:
            with open(RUN_DIR / 'unparsed.txt', 'a', encoding='utf-8') as f:
                f.write(json.dumps({'source_row': source_row, 'raw_output': raw_choice, 'status': poll_diagnostics['parse_status']}, ensure_ascii=False) + '\n')
    reason_prompt = build_reason_prompt(str(cmt), pred_abbr)
    reason_answer = _gen(reason_prompt, MAX_NEW_TOKENS_REASON, thinking=False, stage='posthoc_reason')
    reason = parse_reason(reason_answer)
    reason_status = 'ok' if reason else 'empty_or_incomplete'
    rows.append({
        **audit.sample_fields(),
        'comment': str(cmt),
        'original_gold': gold,
        'real': ABBR2FULL[gold_result['label']],
        'pred': ABBR2FULL[pred_abbr],
        'parsed_output': pred_abbr,
        'raw_output': raw_choice,
        'reason': reason,
        'reason_status': reason_status,
        'yes_candidates': ';'.join(yes_list),
        'unknown_labels': ';'.join(poll_diagnostics['unknown_labels']),
        'fallback_reason': poll_diagnostics['fallback_reason'],
        'selection_route': poll_diagnostics['selection_route'],
        'raw_pred': poll_diagnostics['raw_pred'],
        'parse_ok': poll_diagnostics['parse_ok'],
        'parse_status': poll_diagnostics['parse_status'],
        'fallback_used': poll_diagnostics['fallback_used'],
        'yn_unparsed_count': poll_diagnostics['yn_unparsed_count'],
        'yn_decisions': json.dumps(poll_diagnostics['yn_decisions'], ensure_ascii=False),
        'generation_events': json.dumps(audit.events, ensure_ascii=False, default=str),
    })

pred_df = pd.DataFrame(rows)
# 列顺序固定
# [R08 原代码保留]
# pred_df = pred_df[["comment", "real", "pred", "raw_output", "reason"]]
# [R08 修订]
# [R09-R44 原代码保留]
# pred_df = pred_df[['comment', 'real', 'pred', 'raw_output', 'reason', 'unknown_labels', 'fallback_reason']]
# [R09-R44 修订]
# [R09-R44 修订] Keep every identity and audit column in predictions.csv.

# 评估
report_text = classification_report(
    pred_df["real"], pred_df["pred"],
    labels=LABELS, zero_division=0, digits=3
)
print("\n[INFO] Overall metrics:\n", report_text)

# 保存
pred_df.to_csv(RUN_DIR/"predictions.csv", index=False)

cm = confusion_matrix(pred_df["real"], pred_df["pred"], labels=LABELS)
pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(RUN_DIR/"confusion_matrix.csv")

report_dict = classification_report(
    pred_df["real"], pred_df["pred"],
    labels=LABELS, zero_division=0, output_dict=True
)
rep_df = pd.DataFrame(report_dict).transpose()
rep_df.index.name = "label"
rep_df = rep_df.reset_index()
rep_df["info"] = ""
rep_df.loc[len(rep_df)] = {
    "label": "[INFO]",
    "precision": "",
    "recall": "",
    "f1-score": "",
    "support": "",
    "info": f"Unrecognized / fallback count: {unk_cnt}"
}
rep_df.to_csv(RUN_DIR/"report.csv", index=False)

(Path(RUN_DIR/"settings.json")).write_text(json.dumps({
    "csv": CSV_PATH,
    "model": MODEL_PATH,
    "run_dir": str(RUN_DIR),
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "top_k": TOP_K,
    "max_new_tokens": {
        "yn": MAX_NEW_TOKENS_YN,
        "choose": MAX_NEW_TOKENS_CHOOSE,
        "full": MAX_NEW_TOKENS_FULL,
        "reason": MAX_NEW_TOKENS_REASON
    },
    "fallback_abbr": FALLBACK_ABBR,
    # [R08 原代码保留]
    # "output_columns_order": ["comment","real","pred","raw_output","reason"],
    # [R08 修订]
    # [R22 原代码保留]
    # "output_columns_order": ['comment', 'real', 'pred', 'raw_output', 'reason', 'unknown_labels', 'fallback_reason'],
    # [R22 修订] Persist every identity and diagnostic column.
    "output_columns_order": list(pred_df.columns),
    "method": "per-label yes/no polling + candidate arbitration + full-label fallback"
}, indent=2))

print(f"\n[INFO] Files saved in {RUN_DIR}")
print(f"[INFO] Fallback used: {unk_cnt}")
gc.collect(); torch.cuda.empty_cache()

# [R09-R44 修订] Additional metrics treat final parse failures as errors; primary report retains Data Conf policy.
if not pred_df.empty:
    write_failure_metrics(pred_df, LABELS, RUN_DIR)
(RUN_DIR / 'revision_settings.json').write_text(
    json.dumps(audit.settings(), ensure_ascii=False, indent=2, default=str), encoding='utf-8'
)
