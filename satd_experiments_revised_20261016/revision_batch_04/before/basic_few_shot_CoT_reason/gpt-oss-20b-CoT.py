#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# [R13/R16/R17/R22/R23/R24/R29/R31/R33 修订] Shared explicit audit runtime.
import sys as _revision_sys
from pathlib import Path as _RevisionPath
_revision_sys.path.insert(0, str(_RevisionPath(__file__).resolve().parents[1]))
from revision_runtime import RunAudit, audited_generate, parse_label, parse_classification, visible_answer, write_failure_metrics

import re, os, gc, json
from pathlib import Path
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import torch
from transformers import pipeline

# -------- 0. 基础路径（仅此处改） --------
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
CSV_PATH   = "/data/yangzhifei/pro/ML_SATD/dataset/sampled_label_dataset_normalized_rmvtodo.csv"
MODEL_PATH = "/data/yangzhifei/pro/ML_SATD/models/gpt-oss-20b"   # 或 "openai/gpt-oss-20b"

# 目录名包含时间戳 + 模型名
MODEL_NAME = Path(MODEL_PATH).name
# [R31 原代码保留]
# RUN_DIR    = Path("results") / f"run-{datetime.now():%Y%m%d-%H%M%S}-{MODEL_NAME}"
# [R31 修订] 修订实验使用独立输出根目录和微秒时间戳。
# [R31 修订] 将新产物固定在修订目录，避免依赖启动工作目录。
# RUN_DIR = Path("revised_outputs") / f"run-{datetime.now():%Y%m%d-%H%M%S-%f}-{MODEL_NAME}"
RUN_DIR = Path(__file__).resolve().parents[1] / "revised_outputs" / f'run-{datetime.now():%Y%m%d-%H%M%S-%f}-{MODEL_NAME}'
# [R31 原代码保留]
# RUN_DIR.mkdir(parents=True, exist_ok=True)
# [R31 修订] 拒绝覆盖同名运行。
RUN_DIR.mkdir(parents=True, exist_ok=False)

# -------- 1. 其它参数 --------
PRINT_UNPARSED    = True
FALLBACK_ABBR     = "Data Conf"
MAX_NEW_TOKENS    = 32768     # 分类阶段：保持不变
MAX_INPUT_TOKENS  = 16384
REASON_MAX_NEW_TOKENS = 512   # 仅用于 reason，可适度加长以避免截断

# ----------------- 标签映射（全称 -> 缩写） -----------------
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
LABELS      = list(LABEL_ABBR.keys())
ABBR_SET    = set(LABEL_ABBR.values())
ABBR2FULL   = {abbr: full for full, abbr in LABEL_ABBR.items()}
FULL2ABBR   = LABEL_ABBR

# 规范化工具：去掉所有非字母，转小写
def _canon(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())

# 规范化字典：任意“全称/缩写/大小写/空格变体” → 全称
FULL_BY_CANON = {_canon(full): full for full in LABELS}
FULL_BY_ABBR_CANON = {_canon(abbr): full for full, abbr in LABEL_ABBR.items()}

def normalize_to_full(label: str):
    if not isinstance(label, str):
        return None
    s = label.strip()
    if s in LABELS:
        return s
    if s in ABBR2FULL:
        return ABBR2FULL[s]
    cz = _canon(s)
    if cz in FULL_BY_CANON:
        return FULL_BY_CANON[cz]
    if cz in FULL_BY_ABBR_CANON:
        return FULL_BY_ABBR_CANON[cz]
    return None

# ============ 2. Prompt（简洁 + 强约束） ============
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

PROMPT_TAIL = f"""Please determine which type of ML Self-Admitted Technical Debt (SATD) the following Code Comment belongs to. 
You MUST answer with **only ONE** abbreviation from:
{', '.join(sorted(ABBR_SET))}
""".strip()

def build_prompt(comment: str) -> str:
    return (
        f"{PROMPT_HEAD}\n\n{PROMPT_TAIL}\n\n"
        "Code Comment:\n"
        f"```text\n{comment}\n```\n\n"
        "#Abbreviation:"
    )

# [R13/R16 原代码保留]
# def extract_abbr_from_text(text: str) -> str:
#     if not text:
#         return "<UNK>"
#     cz = _canon(text)
#     canon_map = {_canon(a): a for a in ABBR_SET}
#     if cz in canon_map:
#         return canon_map[cz]
#     for abbr in ABBR_SET:
#         pat = r'\b' + re.sub(r'\s+', r'\\s*', re.escape(abbr)) + r'\b'
#         if re.search(pat, text, flags=re.IGNORECASE):
#             return abbr
#     for full, abbr in FULL2ABBR.items():
#         pat = r'\b' + re.sub(r'\s+', r'\\s*', re.escape(full)) + r'\b'
#         if re.search(pat, text, flags=re.IGNORECASE):
#             return abbr
#     return "<UNK>"
# [R13/R16 修订]
def extract_abbr_from_text(raw: str) -> str:
    return parse_label(raw, LABEL_ABBR)["label"]

# -------- 3. 加载模型（保持不变） --------
print("[INFO] Loading model …")
pipe = pipeline(
    "text-generation",
    model=MODEL_PATH,
    torch_dtype="auto",
    device_map="auto",
)

# ============ 4. reason 的提示与解析（强制成功；不使用本地默认兜底） ============
REASON_SYS_1 = (
    "You are a concise classifier. Think silently. "
    "Reply with ONE paragraph that starts with 'REASON:' followed by a short justification (<= 512 words)."
)
REASON_SYS_2 = (
    "Return exactly ONE paragraph. It must start with 'REASON:' then a short justification (<= 512 words). "
    "No preface, no label, no markdown."
)
REASON_SYS_FORCE = (
    "Output exactly ONE paragraph starting with 'REASON:' followed by a brief justification (<= 512 words). "
    "Ground it in the comment text. No label names, no category names, no markdown, no quotes."
)

def build_reason_prompt(comment: str, abbr: str) -> str:
    # 已确定的类别仅作为上下文，不要求模型复述标签
    return (
        "You have already chosen the ML-SATD abbreviation: " + abbr + ".\n"
        "Provide a brief human-readable justification grounded in the comment content.\n"
        "Format strictly:\n"
        "REASON: <short sentence>\n\n"
        "Code Comment:\n"
        f"```text\n{comment}\n```"
    )

# [R13/R23 原代码保留]
# def _extract_visible(content: str) -> str:
#     if not isinstance(content, str):
#         return ""
#     # 去除可能的 <think>…</think>
#     content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE)
#     # 若含 assistantfinal，取其后内容
#     m = re.search(r"assistantfinal\s*(.*)$", content, flags=re.DOTALL | re.IGNORECASE)
#     return m.group(1).strip() if m else content.strip()
# [R13/R23 修订]
def _extract_visible(raw: str) -> str:
    return visible_answer(raw)

# [R13/R33 原代码保留]
# def parse_reason(text: str) -> str:
#     body = _extract_visible(text)
#     # 1) 优先抓 "REASON:"
#     m = re.search(r"(?:^|\n)\s*REASON\s*[:：]\s*(.+)", body, flags=re.IGNORECASE)
#     cand = None
#     if m:
#         cand = m.group(1).strip()
#     else:
#         # 2) 退化为“第一句非空文本”（仍是模型生成），避免空值
#         lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
#         if lines:
#             cand = lines[0]
#     if not cand:
#         return ""
#     cand = re.sub(r'^[\'"`“”]+|[\'"`“”]+$', '', cand).strip()
#     # 控长度：太长则截断为一句
#     cand = re.split(r'(?<=[.。!?])\s+', cand)[0].strip()
#     return cand[:400]
# [R13/R33 修订]
def parse_reason(raw: str) -> str:
    if not isinstance(raw, str):
        return ""
    view = visible_answer(raw)
    match = re.search(r"(?mis)^\s*(?:#Reason|Reason)\s*[:：]\s*(.+)", view)
    return match.group(1).strip() if match else ""

# [R13/R21/R23/R24/R29/R31 原代码保留]
# @torch.inference_mode()
# def generate_reason(comment: str, abbr: str) -> str:
#     # 1) 首次尝试
#     msgs1 = [
#         {"role": "system", "content": REASON_SYS_1},
#         {"role": "user", "content": build_reason_prompt(comment, abbr)},
#     ]
#     out1 = pipe(msgs1, max_new_tokens=REASON_MAX_NEW_TOKENS)
#     content1 = out1[0]["generated_text"][-1]["content"]
#     reason1 = parse_reason(content1)
#     if len(reason1) >= 3:
#         return reason1
#
#     # 2) 第二次加强约束
#     msgs2 = [
#         {"role": "system", "content": REASON_SYS_2},
#         {"role": "user", "content": build_reason_prompt(comment, abbr)},
#     ]
#     out2 = pipe(msgs2, max_new_tokens=REASON_MAX_NEW_TOKENS)
#     content2 = out2[0]["generated_text"][-1]["content"]
#     reason2 = parse_reason(content2)
#     if len(reason2) >= 3:
#         return reason2
#
#     # 3) 强制格式：仍由模型生成，但更刚性
#     msgs3 = [
#         {"role": "system", "content": REASON_SYS_FORCE},
#         {"role": "user", "content": build_reason_prompt(comment, abbr)},
#     ]
#     out3 = pipe(msgs3, max_new_tokens=REASON_MAX_NEW_TOKENS)
#     content3 = out3[0]["generated_text"][-1]["content"]
#     reason3 = parse_reason(content3)
#     # 最终保证非空（若仍空，则取模型完整可见体的第一行作为理由——依然是模型生成的文本）
#     if len(reason3) < 1:
#         reason3 = _extract_visible(content3).splitlines()[0].strip() if _extract_visible(content3) else "Model-generated reason."
#     return reason3[:400]
# [R13/R21/R23/R24/R29/R31 修订]
def generate_reason(comment: str, final_abbr: str) -> str:
    event = audited_generate(pipe.model, pipe.tokenizer,
                             [{"role": "user", "content": build_reason_prompt(comment, final_abbr)}],
                             {"generation_config": getattr(pipe, "generation_config", pipe.model.generation_config), 'max_new_tokens': REASON_MAX_NEW_TOKENS}, audit=_REVISION_AUDIT,
                             native_thinking=False,
                             reasoning_effort=os.environ.get('SATD_REASONING_EFFORT', 'low'),
                             stage="posthoc_reason")
    if event["parse_status"] != "ok":
        return ""
    visible = event["answer_text"].strip()
    return re.sub(r"(?i)^\s*(?:#Reason|Reason)\s*[:：]\s*", "", visible).strip()

# -------- 5. 分类函数（保持你的做法与参数） --------
# [R13/R16/R23/R24/R29/R31/R33 原代码保留]
# @torch.inference_mode()
# def classify(comment: str):
#     messages = [
#         {"role": "system", "content":
#          "You are a strict classifier. Reasoning: high. "
#          "Output ONLY the abbreviation, one line."},
#         {"role": "user", "content": build_prompt(comment)},
#     ]
#     outputs = pipe(messages, max_new_tokens=MAX_NEW_TOKENS)
#     answer = outputs[0]["generated_text"][-1]['content']
#     raw_answer = answer.split('assistantfinal')[-1] if 'assistantfinal' in answer else answer
#     abbr = extract_abbr_from_text(raw_answer)
#     return abbr, raw_answer
# [R13/R16/R23/R24/R29/R31/R33 修订]
def classify(comment: str):
    messages = [{'role': 'system', 'content': 'You are a strict classifier. Output ONLY the abbreviation, one line.'}, {'role': 'user', 'content': build_prompt(comment)}]
    event = audited_generate(pipe.model, pipe.tokenizer, messages,
                             {"generation_config": getattr(pipe, "generation_config", pipe.model.generation_config), 'max_new_tokens': MAX_NEW_TOKENS}, audit=_REVISION_AUDIT,
                             max_input_tokens=None, native_thinking=False,
                             reasoning_effort=os.environ.get('SATD_REASONING_EFFORT', 'high'), stage="classification")
    parsed = parse_classification(event["answer_text"], LABEL_ABBR, with_reason=False)
    if event["parse_status"] != "ok":
        parsed.update(label="<UNK>", parse_ok=False, status=event["parse_status"])
    return {**parsed, "raw_text": event["raw_text"], "generation_parse_status": event["parse_status"]}

# -------- 6. 主流程 --------
# [R22/R31/R33 原代码保留]
# df = pd.read_csv(CSV_PATH)
#
# rows = []
# unk_cnt = 0
# gold_unmapped = 0
#
# print("[INFO] Classifying …")
# for cmt, gold in tqdm(zip(df["Comment Text"], df["ML TD Type"]),
#                       total=len(df), ncols=80):
#     real_full = normalize_to_full(str(gold))
#     if real_full is None:
#         gold_unmapped += 1
#         continue
#
#     abbr, raw = classify(str(cmt))
#     pred_full = ABBR2FULL.get(abbr, None)
#     if pred_full is None:
#         unk_cnt += 1
#         if PRINT_UNPARSED:
#             with open(RUN_DIR/"unparsed.txt", "a", encoding="utf-8") as f:
#                 f.write(raw + "\n")
#         pred_full = ABBR2FULL[FALLBACK_ABBR]
#         abbr = FALLBACK_ABBR
#
#     # 强制让模型产出 reason（不采用本地默认文本）
#     reason = generate_reason(str(cmt), abbr)
#
#     rows.append({
#         "comment": cmt,
#         "real": real_full,
#         "pred": pred_full,
#         "raw_output": raw,
#         "reason": reason,             # 新增列：始终非空，由模型生成
#     })
#
# pred_df = pd.DataFrame(rows)
#
# if pred_df.empty:
#     print("\n[WARN] No valid samples after label normalization. "
#           "Please check the dataset's label names (column 'ML TD Type').")
# else:
#     report_text = classification_report(
#         pred_df["real"], pred_df["pred"],
#         labels=LABELS, zero_division=0, digits=3
#     )
#     print("\n[INFO] Overall metrics:\n", report_text)
#
#     pred_df.to_csv(RUN_DIR/"predictions.csv", index=False)
#
#     cm = confusion_matrix(pred_df["real"], pred_df["pred"], labels=LABELS)
#     pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(RUN_DIR/"confusion_matrix.csv")
#
#     report_dict = classification_report(
#         pred_df["real"], pred_df["pred"],
#         labels=LABELS, zero_division=0, output_dict=True
#     )
#     rep_df = pd.DataFrame(report_dict).transpose()
#     rep_df.index.name = "label"
#     rep_df = rep_df.reset_index()
#     rep_df["info"] = ""
#
#     info_lines = [
#         f"Unrecognized outputs mapped to '{FALLBACK_ABBR}': {unk_cnt}",
#         f"Gold labels unmapped (skipped): {gold_unmapped}",
#         f"Total evaluated samples: {len(pred_df)} / original {len(df)}"
#     ]
#     rep_df.loc[len(rep_df)] = {
#         "label": "[INFO]",
#         "precision": "",
#         "recall": "",
#         "f1-score": "",
#         "support": "",
#         "info": " | ".join(info_lines)
#     }
#     rep_df.to_csv(RUN_DIR/"report.csv", index=False)
#
# print(f"\n[INFO] Files saved in {RUN_DIR}")
# print(f"[INFO] Unrecognized mapped to '{FALLBACK_ABBR}': {unk_cnt}")
# print(f"[INFO] Gold labels unmapped (skipped): {gold_unmapped}")
#
# gc.collect(); torch.cuda.empty_cache()
# [R22/R31/R33 修订]
# [R22/R31/R33 修订] Per-row identity and pre-fallback diagnostics; old policy report stays primary.
_REVISION_AUDIT = RunAudit(RUN_DIR, MODEL_PATH, CSV_PATH)
df = pd.read_csv(CSV_PATH)
rows = []
unk_cnt = 0
for source_row, source in tqdm(df.iterrows(), total=len(df), ncols=80):
    _REVISION_AUDIT.set_sample(source_row, source.to_dict())
    gold_result = parse_label(str(source["ML TD Type"]), LABEL_ABBR)
    if not gold_result["parse_ok"]:
        raise ValueError(f"Invalid gold label at source row {source_row}; no rows are silently skipped")
    cmt = str(source["Comment Text"])
    result = classify(cmt)
    raw_pred = result["label"]
    parse_ok = bool(result["parse_ok"])
    final_abbr = raw_pred if parse_ok else FALLBACK_ABBR
    unk_cnt += int(not parse_ok)
    reason = generate_reason(cmt, final_abbr)
    rows.append({
        **_REVISION_AUDIT.sample_fields(),
        "comment": cmt,
        "real": ABBR2FULL[gold_result["label"]],
        "pred": ABBR2FULL[final_abbr],
        "pred_before_fallback": raw_pred,
        "parsed_output": raw_pred,
        "parse_ok": parse_ok,
        "parse_status": result["status"],
        "fallback_used": not parse_ok,
        "fallback_reason": "" if parse_ok else result["status"],
        "raw_output": result["raw_text"],
        "reason": reason,
        "reason_parse_ok": bool(reason),
        "generation_audit": json.dumps(_REVISION_AUDIT.events, ensure_ascii=False),
    })
pred_df = pd.DataFrame(rows)
pred_df.to_csv(RUN_DIR/"predictions.csv", index=False)
report_text = classification_report(pred_df["real"], pred_df["pred"], labels=LABELS, zero_division=0, digits=3)
print(report_text)
(RUN_DIR/"report.txt").write_text(report_text, encoding="utf-8")
report_dict = classification_report(pred_df["real"], pred_df["pred"], labels=LABELS, zero_division=0, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose().rename_axis("label").reset_index()
report_df["info"] = ""
report_df.loc[len(report_df)] = {"label": "[INFO]", "info": f"Parsing failures before fallback: {unk_cnt} / {len(df)}"}
report_df.to_csv(RUN_DIR/"report.csv", index=False)
pd.DataFrame(confusion_matrix(pred_df["real"], pred_df["pred"], labels=LABELS), index=LABELS, columns=LABELS).to_csv(RUN_DIR/"confusion_matrix.csv")
write_failure_metrics(pred_df, LABELS, RUN_DIR)
(RUN_DIR/"settings.json").write_text(json.dumps({
    **_REVISION_AUDIT.settings(),
    "model": MODEL_PATH, "csv": CSV_PATH, "run_dir": str(RUN_DIR),
    "max_new_tokens": MAX_NEW_TOKENS,
    "reasoning_mode": "gpt_oss_native_reasoning",
    "fallback_abbr": FALLBACK_ABBR,
    "metrics_policy": "Primary: existing final fallback predictions. Additional failure_as_error metrics mark invalid final parses wrong.",
    "input_policy": "Full templated input, including demonstrations/comment/code context; actual context with 64 safety tokens; requested generation cap is reduced to remaining context and the effective cap is recorded; no input truncation.",
    "parser_policy": "Unique complete canonical abbreviation/full-name or the explicitly requested answer field; never search reasoning.",
    "unknown_count_before_fallback": unk_cnt,
}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[INFO] Revised results saved to {RUN_DIR}")
