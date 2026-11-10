#!/usr/bin/env python
# -*- coding: utf-8 -*-
# [R16/R23/R31 原代码保留]
# """
# Two-stage ML-SATD classifier (coarse -> fine), with robust abbreviation parsing.
#
# Fixes:
# - Stage-1 outputs like "Abbreviation: **AWR** (Awareness)" will be correctly parsed (no more <UNK>).
# - Stage-2 restricts candidates to the fine labels belonging to the predicted coarse class.
# - Adds coarse/fine metrics & confusion matrices; writes Stage-1 result into predictions.csv.
#
# Author: you
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
from transformers import AutoModelForCausalLM, AutoTokenizer
# [R16/R22/R24/R31/R33 修订]
import sys
import hashlib
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from revision_runtime import RunAudit, audited_generate, parse_label, write_failure_metrics

# ========== 0. 基础路径（仅此处改） ==========
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # GPU 号
CSV_PATH   = "/data/yangzhifei/pro/ML_SATD/dataset/sampled_label_dataset_normalized_rmvtodo.csv"  # ← 修改
MODEL_PATH = "/data/yangzhifei/pro/ML_SATD/models/CodeLlama-7b-Instruct-hf"                                       # ← 修改

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

# ========== 1. 其它参数 ==========
MAX_NEW_TOKENS = 20
PRINT_UNPARSED = True

# [R45 OFFICIAL PROFILE 原代码保留]
# DO_SAMPLE = True  # 是否采样（True: 采样；False: 贪婪）
# [R45 OFFICIAL PROFILE 修订]
# Sampling is resolved only by the shared official profile.
# [R45 OFFICIAL PROFILE 原代码保留]
# TEMPERATURE       = 0.2
# [R45 OFFICIAL PROFILE 修订]
# Sampling is resolved only by the shared official profile.
# [R45 OFFICIAL PROFILE 原代码保留]
# TOP_P             = 0.95
# [R45 OFFICIAL PROFILE 修订]
# Sampling is resolved only by the shared official profile.


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
COARSE_ABBR_LOWER = {a.lower():a for a in COARSE_LABELS}

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
    "Machine Learning Knowledge":"AWR",
    "Model Interpretability":"AWR",
    "Model Code Comprehension":"RDB",
    "Duplicate Model Code":"DCE",
    "Duplicate Feature Extraction Code":"DCE",
    "Weight Configuration":"CFO",
    "Layer Configuration":"CFO",
    "Hyper-parameter Configuration":"CFO",
    "Code Dependency":"CDD",
    "Glue Code":"CDD",
    "Custom Data Type":"CDD",
    "Multiple Languages":"CDD",
    "Unnecessary Model Code":"CDD",
    "Data Processing Configuration":"DTD",
    "Plain Old Data Type":"DTD",
    "Data Storage Configuration":"DTD",
    "Abstraction":"MDL",
    "Boundary Erosion":"MDL",
    "Model Code Modifiability":"MDL",
    "Model Code Reusability":"MDL",
    "Prediction Quality":"PRF",
    "Machine Learning Reliability":"PRF",
    "Prototype":"SCL",
}

# 各粗类下细类的回退（细类解析失败时使用）
FALLBACK_FINE_BY_COARSE = {
    "AWR":"MLKnow",
    "RDB":"MCC",
    "DCE":"DMC",
    "CFO":"HP Conf",
    "CDD":"MLDepend",
    "DTD":"Data Conf",
    "MDL":"MCM",
    "PRF":"PQ",
    "SCL":"Proto",
}

# ========== 4. Prompt（Coarse 9 类） ==========
COARSE_PROMPT_HEAD = r"""
There are 9 coarse-grained types of Machine Learning (ML) Self-Admitted Technical Debt (SATD):

Awareness (AWR),
Readability (RDB),
Duplicate Code Elimination (DCE),
Configurable Options (CFO),
Code Dependency (CDD),
Data Dependency (DTD),
Modularity (MDL),
Performance (PRF),
Scalability (SCL).
""".strip()

# COARSE_PROMPT_HEAD = r"""
# There are 9 coarse-grained types of ML SATD:

# Awareness (AWR): Uninformed ML design or insufficient model explainability leads to uncertain behavior; naive solutions must be revisited and interpretability evidence should be provided.

# Readability (RDB): Model code suffers from poor readability (e.g., unclear naming, long or tangled logic), impeding comprehension and safe evolution.

# Duplicate Code Elimination (DCE): Duplicate logic in model or feature pipelines increases maintenance cost and inconsistency risk; it should be consolidated.

# Configurable Options (CFO): Hard-coded or insufficiently exposed options (weights, layers, hyperparameters) force code edits for experimentation; they should be parameterized.

# Code Dependency (CDD): Integration- and dependency-driven constraints—external libraries, glue code, custom types, mixed languages, or legacy code—impede evolution and require interim shims or removals.

# Data Dependency (DTD): Debt rooted in data handling—processing steps, raw types, and storage/representation—causes brittleness and confusion; pipelines and schemas must be explicit and robust.

# Modularity (MDL): Weak modular structure—missing abstractions, eroded boundaries, or non-pluggable design—reduces modifiability and reusability; components should be decoupled and interface-driven.

# Performance (PRF): Workflow changes may degrade model quality or robustness; performance must be measured, guarded by tests, and strengthened with reliability controls.

# Scalability (SCL): Prototype-grade solutions promoted to production lack scalability and hardening; they must be engineered for scale and operational safety.
# """.strip()

# COARSE_PROMPT_HEAD = r"""
# There are 9 coarse-grained types of ML SATD:

# Awareness (AWR): Lack of developer understanding or certainty about ML algorithms, APIs, data, or model behavior. Often arises due to “black box” nature of ML models.

# Readability (RDB):Model code has poor legibility or unclear logic, especially in ML contexts (e.g., poorly named variables, dense math code).

# Duplicate Code Elimination (DCE): Identical or near-identical ML model or preprocessing logic appears in multiple places.

# Configurable Options (CFO): Missing or poorly implemented configuration points for weights, network layers, or model parameters.

# Code Dependency (CDD): Code depends on other code or libraries in a way that blocks maintenance, requires workarounds, or is outdated.

# Data Dependency (DTD): Issues in how data is processed, represented, or stored that hinder maintainability or correctness.

# Modularity (MDL): Poor modular boundaries or design make the system harder to evolve, reuse, or maintain.

# Performance (PRF): Performance or robustness is insufficient, requiring evaluation, optimization, or reliability improvements.

# Scalability (SCL): Prototype-level code is used in production without scaling optimizations.
# """.strip()

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
    tail = "Please classify the following Code Comment into exactly ONE of these coarse abbreviations:\n" \
           + ", ".join(COARSE_LABELS) + \
           "\nReturn the abbreviation only (e.g., AWR)."
    return f"{COARSE_PROMPT_HEAD}\n\n{tail}\n\nCode Comment:\n\"\"\"\n{comment}\n\"\"\"\n\nAnswer:"

# ========== 5. Prompt（Fine 23 类；按粗类限制候选） ==========
FINE_PROMPT_HEAD = r"""
There are 23 types of Machine Learning (ML) Self-Admitted Technical Debt (SATD):

MACHINE LEARNING KNOWLEDGE (MLKnow): Machine-learning software carries plenty of unique challenges. Uneducated solutions by unaware developers may have to be revisited.

MODEL INTERPRETABILITY (MLInter): Machine-learning models are a black box, causing poor understanding of model functionality; this can lead to unknown behaviour.

MODEL CODE COMPREHENSION (MCC): Model code carries extra legibility concerns not present in traditional software (e.g., poorly-named temporary matrix variables).

DUPLICATE MODEL CODE (DMC): Code duplication frequently occurs in model code.

DUPLICATE FEATURE EXTRACTION CODE (DFEC): Code duplication frequently occurs in feature-extraction code.

WEIGHT CONFIGURATION (Weight Conf): Editing code that involves the weights of a model, or configuring a model's weights directly.

LAYER CONFIGURATION (Layer Conf): Editing code that deals with model layers, or configuring layers themselves.

HYPER-PARAMETER CONFIGURATION (HP Conf): Configuring hyper-parameters of a model, or editing default values of off-the-shelf models.

CODE DEPENDENCY (MLDepend): A needed change is blocked by an external dependency (library / subsystem).

GLUE CODE (GC): Supporting code written to interface with other code, inhibiting improvements due to dependent peculiarities.

CUSTOM DATA TYPE (CDT): Using data types provided by general-purpose packages can cause extensive inter-operability burden.

MULTIPLE LANGUAGES (ML): Components written in other languages may introduce difficulties in ML development.

UNNECESSARY MODEL CODE (UMC): Model code that bottlenecks execution, is unreachable / deprecated, or is unnecessary and should be removed.

DATA PROCESSING CONFIGURATION (Data Conf): Configuring how data is processed, by editing the data directly or adding new processing steps.

PLAIN OLD DATA TYPE (PODT): Using raw data types in ML causes confusion when interpreting processes.

DATA STORAGE CONFIGURATION (DS Conf): Configuring how data is represented internally or stored externally.

ABSTRACTION (Abs): Lack of abstractions in ML systems causes cascading changes when a component is modified.

BOUNDARY EROSION (BE): Lack of boundaries between subsystems complicates maintenance and isolates changes poorly.

MODEL CODE MODIFIABILITY (MCM): Model code should enable easy maintenance and future modifications.

MODEL CODE REUSABILITY (MCR): Model code should be generalised for reuse in varying situations.

PREDICTION QUALITY (PQ): Workflow changes may affect performance.

MACHINE LEARNING RELIABILITY (MLRelia): Model functionality depends on data quality; robustness measures should be in place.

PROTOTYPE (Proto): Deploying small-scale prototypes into full systems can be dangerous.
""".strip()

# [R28 原代码保留]
# def build_prompt_fine(comment: str, coarse_abbr: str) -> str:
#     allowed = COARSE2FINE_ABBR.get(coarse_abbr, sorted(FINE_ABBR_SET))
#     tail = "Please determine the fine-grained ML SATD type. You MUST answer with only ONE abbreviation from:\n" \
#            + ", ".join(allowed) + "\nReturn the abbreviation only (e.g., HP Conf)."
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

# ========== 6. 解析函数（健壮版） ==========
COARSE_REGEX = re.compile(r'\b(AWR|RDB|DCE|CFO|CDD|DTD|MDL|PRF|SCL)\b', re.I)

# [R16 原代码保留]
# def extract_coarse(raw: str) -> str:
#     """从任意首行文本里提取粗类缩写（容忍 'Abbreviation: **AWR** (Awareness)' 等格式）"""
#     # [R04 原代码保留]
#     # if not isinstance(raw, str):
#     #     return "<UNK>"
#     # [R04 修订]
#     if not isinstance(raw, str) or not raw.strip():
#         return "<UNK>"
#     s = raw.splitlines()[0].strip()
#
#     # 1) 直接检索允许缩写
#     m = COARSE_REGEX.search(s)
#     if m:
#         return m.group(1).upper()
#
#     # 2) markdown **AWR**
#     m = re.search(r'\*\*\s*(AWR|RDB|DCE|CFO|CDD|DTD|MDL|PRF|SCL)\s*\*\*', s, flags=re.I)
#     if m:
#         return m.group(1).upper()
#
#     # 3) 兜底（去非字母后包含匹配，如 abbreviationawrawareness）
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
#     # 1) 动态构建模式：支持 'HP Conf' 中的空格写法
#     pat_choices = [re.escape(x).replace(r'\ ', r'\s*') for x in allowed_abbr_list]
#     pat = r'\b(' + '|'.join(pat_choices) + r')\b'
#     m = re.search(pat, s, flags=re.I)
#     if m:
#         hit = re.sub(r'\s+', ' ', m.group(1)).strip().lower()
#         for a in allowed_abbr_list:
#             if re.sub(r'\s+', ' ', a).strip().lower() == hit:
#                 return a
#
#     # 2) markdown **HP Conf**
#     m = re.search(r'\*\*\s*(' + '|'.join(pat_choices) + r')\s*\*\*', s, flags=re.I)
#     if m:
#         hit = re.sub(r'\s+', ' ', m.group(1)).strip().lower()
#         for a in allowed_abbr_list:
#             if re.sub(r'\s+', ' ', a).strip().lower() == hit:
#                 return a
#
#     # 3) 兜底：letters-only 包含匹配
#     letters = re.sub(r'[^A-Za-z]', '', s).lower()
#     for a in allowed_abbr_list:
#         if re.sub(r'[^A-Za-z]', '', a).lower() in letters:
#             return a
#     return "<UNK>"
# [R16 修订]
def extract_fine(raw: str, allowed_abbr_list) -> str:
    """Normalize full names/abbreviations consistently and enforce the candidate set."""
    return parse_label(raw, FINE_LABEL_ABBR, allowed=allowed_abbr_list)["label"]

# ========== 7. 模型加载 ==========
print("[INFO] Loading model …")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.truncation_side = "left"  # 尽量保留结尾的 "Answer:" 和注释

def allowed_ctx_limit(model, tokenizer, reserve=MAX_NEW_TOKENS+64, default_cap=8192):
    limit = getattr(model.config, "max_position_embeddings", None) \
         or getattr(model.config, "max_sequence_length", None) \
         or tokenizer.model_max_length
    if not isinstance(limit, int) or limit > 1_000_000:
        limit = default_cap
    return max(512, limit - reserve)

COARSE_MAX_LEN = min(2048, allowed_ctx_limit(model, tokenizer))
FINE_MAX_LEN   = min(8192, allowed_ctx_limit(model, tokenizer))

def apply_template(messages):
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        # 兼容不支持 enable_thinking 的 tokenizer
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

# [R45 OFFICIAL PROFILE 原代码保留]
# @torch.inference_mode()
# # [R17/R24/R31 原代码保留]
# # def generate_once(prompt: str, max_len: int):
# #     messages = [{"role": "user", "content": prompt}]
# #     text = apply_template(messages)
# #     inputs = tokenizer(text, return_tensors="pt",
# #                        truncation=True, max_length=max_len).to(model.device)
# #
# #     outputs = model.generate(
# #         **inputs,
# #         max_new_tokens=MAX_NEW_TOKENS,
# #         do_sample=DO_SAMPLE,
# #         temperature=TEMPERATURE,
# #         top_p=TOP_P,
# #         pad_token_id=tokenizer.eos_token_id,
# #     )
# #     raw = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
# #                            skip_special_tokens=True).strip()
# #     return re.split(r"[\r\n]", raw, maxsplit=1)[0]
# # [R17/R24/R31 修订]
# def generate_once(prompt: str, max_len: int, stage: str = "classification"):
#     """Encode the full rendered chat once; retain raw output and audit budget/seed/config."""
#     messages = [{"role": "user", "content": prompt}]
#     return audited_generate(
#         model, tokenizer, messages,
#         {'max_new_tokens': MAX_NEW_TOKENS, 'do_sample': DO_SAMPLE, 'temperature': TEMPERATURE, 'top_p': TOP_P, 'pad_token_id': tokenizer.eos_token_id},
#         audit=AUDIT, max_input_tokens=max_len, native_thinking=False, stage=stage,
#     )
# [R45 OFFICIAL PROFILE 修订]
@torch.inference_mode()
def generate_once(prompt: str, max_len: int, stage: str = "classification"):
    """Retain the original task output cap while the shared official profile owns sampling."""
    messages = [{"role": "user", "content": prompt}]
    return audited_generate(
        model, tokenizer, messages,
        {"max_new_tokens": MAX_NEW_TOKENS},
        audit=AUDIT, max_input_tokens=max_len, native_thinking=False,
        official_profile=True, stage=stage,
    )

# ========== 8. 两阶段推理 ==========
@torch.inference_mode()
# [R10/R16/R31/R33 原代码保留]
# def classify_two_stage(comment: str):
#     # Stage-1: coarse
#     coarse_prompt = build_prompt_coarse(comment)
#     coarse_raw = generate_once(coarse_prompt, max_len=COARSE_MAX_LEN)
#     # print(coarse_raw)
#     coarse_abbr = extract_coarse(coarse_raw)
#
#     # Stage-2: fine
#     if coarse_abbr == "<UNK>":
#         allowed = sorted(FINE_ABBR_SET)  # 粗类不明 -> 全局允许
#     else:
#         allowed = COARSE2FINE_ABBR.get(coarse_abbr, sorted(FINE_ABBR_SET))
#
#     fine_prompt = build_prompt_fine(comment, coarse_abbr)
#     fine_raw = generate_once(fine_prompt, max_len=FINE_MAX_LEN)
#     fine_abbr = extract_fine(fine_raw, allowed)
#
#     if fine_abbr == "<UNK>":
#         # 若细类不在允许集合，回退到该粗类默认
#         fine_abbr = FALLBACK_FINE_BY_COARSE.get(coarse_abbr, "Data Conf")
#
#     return coarse_abbr, coarse_raw, fine_abbr, fine_raw
# [R10/R16/R31/R33 修订]
def classify_two_stage(comment: str):
    coarse_prompt = build_prompt_coarse(comment)
    coarse_result = generate_once(coarse_prompt, max_len=COARSE_MAX_LEN, stage="coarse")
    coarse_parse = parse_label(
        coarse_result["answer_text"], {abbr: abbr for abbr in COARSE_LABELS}, allowed=COARSE_LABELS
    )
    coarse_abbr = coarse_parse["label"]
    # Preserve the original policy: an invalid coarse answer opens all fine candidates.
    allowed = COARSE2FINE_ABBR.get(coarse_abbr, sorted(FINE_ABBR_SET))
    fine_prompt = build_prompt_fine(comment, coarse_abbr)
    fine_result = generate_once(fine_prompt, max_len=FINE_MAX_LEN, stage="fine")
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

# ========== 9. 主流程 ==========
# [R10/R22/R31/R32/R33 原代码保留]
# df = pd.read_csv(CSV_PATH)
# rows = []
# unk_coarse_cnt = 0
# unk_fine_cnt   = 0
#
# print("[INFO] Classifying (two-stage, fixed parser) …")
# for cmt, gold in tqdm(zip(df["Comment Text"], df["ML TD Type"]),
#                       total=len(df), ncols=80):
#     coarse_abbr, coarse_raw, fine_abbr, fine_raw = classify_two_stage(cmt)
#
#     if coarse_abbr == "<UNK>":
#         unk_coarse_cnt += 1
#     if fine_abbr not in FINE_ABBR_SET:
#         unk_fine_cnt += 1
#
#     rows.append({
#         "comment": cmt,
#         "real": FINE_ABBR2FULL.get(gold, gold),     # 细类全称（真实）
#         "pred": FINE_ABBR2FULL.get(fine_abbr, fine_abbr),  # 细类全称（预测）
#         # —— Stage-1 —— #
#         "coarse_pred": coarse_abbr,
#         "coarse_raw_output": coarse_raw,
#         # —— Stage-2 —— #
#         "fine_abbr": fine_abbr,
#         "raw_output": fine_raw,
#     })
#
# pred_df = pd.DataFrame(rows)
#
# # ========== 10. 评估 ==========
# # 细（23）
# fine_report_text = classification_report(pred_df["real"], pred_df["pred"],
#                                          labels=FINE_LABELS, zero_division=0, digits=3)
# print("\n[INFO] Fine-grained metrics (23):\n", fine_report_text)
#
# # 粗（9）
# pred_df["real_coarse"] = pred_df["real"].map(FINE2COARSE_FULL)
# pred_df["pred_coarse"] = pred_df["pred"].map(FINE2COARSE_FULL)
# coarse_report_text = classification_report(pred_df["real_coarse"], pred_df["pred_coarse"],
#                                            labels=COARSE_LABELS, zero_division=0, digits=3)
# print("\n[INFO] Coarse-grained metrics (9):\n", coarse_report_text)
#
# # ========== 11. 保存 ==========
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
#     "run_dir": str(RUN_DIR),
#     "max_new_tokens": MAX_NEW_TOKENS,
#     "do_sample": DO_SAMPLE,
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
# [R45 OFFICIAL PROFILE 原代码保留]
# settings = {'csv': CSV_PATH, 'model': MODEL_PATH, 'run_dir': str(RUN_DIR), 'max_new_tokens': MAX_NEW_TOKENS, 'do_sample': DO_SAMPLE, 'two_stage': True, 'unknown_coarse': unk_coarse_cnt, 'unknown_fine': unk_fine_cnt}
# [R45 OFFICIAL PROFILE 修订]
settings = {'csv': CSV_PATH, 'model': MODEL_PATH, 'run_dir': str(RUN_DIR), 'max_new_tokens': MAX_NEW_TOKENS, 'two_stage': True, 'unknown_coarse': unk_coarse_cnt, 'unknown_fine': unk_fine_cnt, 'generation_profile_policy': 'Shared revision_runtime official_profile=True: verified author recommendations where available, otherwise checkpoint defaults; no per-script sampler overrides.', 'generation_configuration_record': 'Actual resolved profile, source identity and effective per-call generation config are recorded by AUDIT.settings() and generation_events.jsonl.'}
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
    "legacy_stage_input_caps": {"coarse": COARSE_MAX_LEN, "fine": FINE_MAX_LEN},
    "primary_metric_policy": "original fallback prediction retained in pred/report_fine/report_coarse; failure-as-error is additional",
    "coarse_report_policy": "report_coarse is final_fine_aggregated_coarse; report_coarse_stage1 independently evaluates stage1_coarse_pred",
    "fallback_count": int(pred_df["fallback_used"].sum()),
    "historical_results": "unchanged; no re-evaluation executed by this revision",
})
(RUN_DIR / "settings.json").write_text(json.dumps(settings, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
print(f"\n[INFO] Files saved in {RUN_DIR}")
print(f"[INFO] Unknown coarse predictions before fallback: {unk_coarse_cnt}")
print(f"[INFO] Unknown fine predictions before fallback: {unk_fine_cnt}")
gc.collect()
torch.cuda.empty_cache()
