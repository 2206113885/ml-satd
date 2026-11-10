#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re, os, gc, json
from pathlib import Path
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import torch
from transformers import pipeline

# -------- 0. 基础路径（仅此处改） --------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
CSV_PATH   = "/data/yangzhifei/pro/ML_SATD/dataset/sampled_label_dataset_normalized_rmvtodo.csv"
MODEL_PATH = "/data/yangzhifei/pro/ML_SATD/models/gpt-oss-20b"   # 或 "openai/gpt-oss-20b"

# 目录名包含时间戳 + 模型名
MODEL_NAME = Path(MODEL_PATH).name
RUN_DIR    = Path("results") / f"run-{datetime.now():%Y%m%d-%H%M%S}-{MODEL_NAME}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

# -------- 1. 其它参数 --------
PRINT_UNPARSED    = True
FALLBACK_ABBR     = "Data Conf"
MAX_NEW_TOKENS    = 32768
MAX_INPUT_TOKENS  = 16384

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
# 常用集合/映射
LABELS      = list(LABEL_ABBR.keys())                 # 23个“全称”
ABBR_SET    = set(LABEL_ABBR.values())                # 23个“缩写”
ABBR2FULL   = {abbr: full for full, abbr in LABEL_ABBR.items()}  # abbr->full
FULL2ABBR   = LABEL_ABBR                                            # full->abbr

# 规范化工具：去掉所有非字母，转小写
def _canon(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())

# 规范化字典：任意“全称/缩写/大小写/空格变体” → 全称
FULL_BY_CANON = {_canon(full): full for full in LABELS}
FULL_BY_ABBR_CANON = {_canon(abbr): full for full, abbr in LABEL_ABBR.items()}

def normalize_to_full(label: str):
    """把数据集里的金标规范到“全称”。支持全称/缩写/大小写差异/空格差异。"""
    if not isinstance(label, str):
        return None
    s = label.strip()
    if s in LABELS:        # 已经是全称（完全匹配）
        return s
    if s in ABBR2FULL:     # 是缩写（完全匹配）
        return ABBR2FULL[s]
    cz = _canon(s)
    if cz in FULL_BY_CANON:
        return FULL_BY_CANON[cz]
    if cz in FULL_BY_ABBR_CANON:
        return FULL_BY_ABBR_CANON[cz]
    return None  # 无法识别，返回 None，后续会跳过该样本

# ============ 2. Prompt（简洁 + 强约束） ============
# PROMPT_HEAD = r"""
# There are 23 types of Machine Learning (ML) SATD (Self-Admitted Technical Debt):

# MACHINE LEARNING KNOWLEDGE (MLKnow): Machine-learning software carries plenty of unique challenges. Uneducated solutions by unaware developers may have to be revisited.
# examples:
# 1. `TODO: Confirm whether layer-norm before attention actually boosts convergence for small-batch training on this dataset`
# 2. `TODO: Figure out if label smoothing distorts precision-recall when classes are highly imbalanced`
# 3. `XXX: Revisit our choice of cosine annealing—does it still make sense once we switch to AdamW?`

# MODEL INTERPRETABILITY (MLInter): Machine-learning models are a black box, causing poor understanding of model functionality; this can lead to unknown behaviour.
# examples:
# 1. `TODO: Generate SHAP summary plot so users can see feature impact`
# 2. `TODO: Log attention weights for each prediction to aid debugging`
# 3. `XXX: Provide saliency-map overlay in evaluation notebook for misclassified images`

# MODEL CODE COMPREHENSION (MCC): Model code carries extra legibility concerns not present in traditional software (e.g., poorly-named temporary matrix variables).
# examples:
# 1. `TODO: break up monolithic function train_eval_predict() into clearer, single-purpose helpers`
# 2. `FIXME: variable “h” is reused for both hidden state and Hessian—pick distinct, descriptive names`
# 3. `XXX: replace magic constant 512 (actually EMBED_SIZE) with a config parameter or enum`

# DUPLICATE MODEL CODE (DMC): Code duplication frequently occurs in model code.
# examples:
# 1. `TODO: remove copy-pasted ConvBlock definition—reuse common_layers.ConvBlock instead`
# 2. `FIXME: forward() in MyResNet duplicates logic from BaseResNet; refactor to call super()`
# 3. `XXX: data_augmentation_v2() is 95 % identical to data_augmentation(); consolidate into one helper`

# DUPLICATE FEATURE EXTRACTION CODE (DFEC): Code duplication frequently occurs in feature-extraction code.
# examples:
# 1. `TODO: merge extract_mfcc() and extract_mfcc_fast(); only differ in hop_length`
# 2. `FIXME: pixel_stats_v2() repeats almost all logic from pixel_stats(); refactor into shared util`
# 3. `XXX: text_features_v1 and text_features_v2 are copy-pasted—create one configurable pipeline`

# WEIGHT CONFIGURATION (Weight Conf): Editing code that involves the weights of a model, or configuring a model's weights directly.
# examples:
# 1. `TODO: expose weight_decay as configurable argument instead of hard-coding 0.01`
# 2. `FIXME: pretrained checkpoint loads fp16 weights—add auto-cast to ensure dtype consistency`
# 3. `XXX: add toggle to freeze first two Transformer layers during fine-tune via gradient mask`

# LAYER CONFIGURATION (Layer Conf): Editing code that deals with model layers, or configuring layers themselves.
# examples:
# 1. `TODO: swap BatchNorm for LayerNorm when depth > 24`
# 2. `FIXME: insert positional-embedding layer before first Transformer block`
# 3. `XXX: make dropout rate per-layer configurable instead of using global 0.1`

# HYPER-PARAMETER CONFIGURATION (HP Conf): Configuring hyper-parameters of a model, or editing default values of off-the-shelf models.
# examples:
# 1. `TODO: make learning_rate schedule (cosine vs. linear) selectable via CLI flag`
# 2. `FIXME: batch_size is hard-coded to 32; derive from GPU memory or user input`
# 3. `XXX: expose early_stopping_patience in config.yaml instead of magic value 10`

# CODE DEPENDENCY (MLDepend): A needed change is blocked by an external dependency (library / subsystem).
# examples:
# 1. `TODO: bump scikit-learn to ≥1.6 once conda forge wheels for arm64 become available`
# 2. `FIXME: delete custom tokenizer patch after huggingface/transformers merges PR #42812`
# 3. `XXX: keep legacy CUDA 11 build until upstream faiss releases official 12.x binaries`

# GLUE CODE (GC): Supporting code written to interface with other code, inhibiting improvements due to dependent peculiarities.
# examples:
# 1. `TODO: remove pandas-to-torch bridge once DataLoader accepts DataFrame directly`
# 2. `FIXME: adapter.py only exists to map old Predictor API to new one—rewrite consumers instead`
# 3. `XXX: consolidate fetch_from_s3_and_cache(); purely glue between boto3 and our local cache manager`

# CUSTOM DATA TYPE (CDT): Using data types provided by general-purpose packages can cause extensive inter-operability burden.
# examples:
# 1. `TODO: replace custom SparseTensorDict with standard torch.sparse_coo_tensor before exporting`
# 2. `FIXME: graph_utils.Node stores metadata in AttrDict; breaks ONNX serialization—use plain dict`
# 3. `XXX: pipeline returns dask DataFrame, but Keras expects NumPy array—add type-conversion layer`

# MULTIPLE LANGUAGES (ML): Components written in other languages may introduce difficulties in ML development.
# examples:
# 1. `TODO: port custom attention kernel from CUDA C++ to Metal for macOS deployment`
# 2. `FIXME: keep tokenizer logic in sync between Rust CLI and Python training script—generate bindings instead`
# 3. `XXX: consolidate duplicate data_cleaning.{scala,py}; two implementations drift out of spec easily`

# UNNECESSARY MODEL CODE (UMC): Model code that bottlenecks execution, is unreachable / deprecated, or is unnecessary and should be removed.
# examples:
# 1. `TODO: delete legacy compute_grad_stats(); superseded by torch.autograd.profiler`
# 2. `FIXME: remove flag use_old_preprocess once new pipeline is stable`
# 3. `XXX: drop debug print statements inside training loop before release`

# DATA PROCESSING CONFIGURATION (Data Conf): Configuring how data is processed, by editing the data directly or adding new processing steps.
# examples:
# 1. `TODO: add configurable pipeline step to down-sample majority class before batching`
# 2. `FIXME: current JSON parser drops NaNs—update to preserve missing values for later imputation`
# 3. `XXX: expose window_size and stride in config.yaml so sliding-window generator isn’t hard-coded`

# PLAIN OLD DATA TYPE (PODT): Using raw data types in ML causes confusion when interpreting processes.
# examples:
# 1. `TODO: replace raw list of logits with a NamedTuple so callers know the order`
# 2. `FIXME: function returns plain dict; define a TypedConfig dataclass instead for clarity`
# 3. `XXX: passing labels as bare NumPy array—wrap in LabelsTensor to preserve shape metadata`

# DATA STORAGE CONFIGURATION (DS Conf): Configuring how data is represented internally or stored externally.
# examples:
# 1. `TODO: migrate checkpoint format from .pth to safetensors for safer serialization`
# 2. `FIXME: results_writer currently appends CSV rows one by one—buffer and batch-flush to reduce I/O`
# 3. `XXX: store intermediate feature maps in mmap file to avoid exceeding GPU memory`

# ABSTRACTION (Abs): Lack of abstractions in ML systems causes cascading changes when a component is modified.
# examples:
# 1. `TODO: extract common scoring logic into BaseMetric so each model wrapper isn’t re-implementing it`
# 2. `FIXME: decouple data_loader from Trainer—should accept any iterable, not hard-coded DataLoader`
# 3. `XXX: replace direct calls to torch.cuda.* with a HardwareBackend interface to support TPU`

# BOUNDARY EROSION (BE): Lack of boundaries between subsystems complicates maintenance and isolates changes poorly.
# examples:
# 1. `TODO: migrate training-time logging out of Trainer class; should live in metrics.logger`
# 2. `FIXME: data_loader currently triggers augmentations inline—move to separate preprocess pipeline`
# 3. `XXX: remove direct DB writes from evaluate.py; expose results through results_api instead`

# MODEL CODE MODIFIABILITY (MCM): Model code should enable easy maintenance and future modifications.
# examples:
# 1. `TODO: move hyper-parameter defaults into config file so changing them doesn’t require code edits`
# 2. `FIXME: hard-coded embedding_dim (256) sprinkled across modules—define once in constants.py`
# 3. `XXX: split run_experiment.py into cli entrypoint and reusable Experiment class for easier reuse`

# MODEL CODE REUSABILITY (MCR): Model code should be generalised for reuse in varying situations.
# examples:
# 1. `TODO: abstract make_encoder() so VisionTransformer and ResNet builders share the same backbone factory`
# 2. `FIXME: refactor evaluate_batch() into mix-in class so both Trainer and HyperTune reuse identical logic`
# 3. `XXX: remove hard-coded Softmax loss; accept any callable loss_fn to support regression and ranking tasks`

# PREDICTION QUALITY (PQ): Workflow changes may affect performance.
# examples:
# 1. `TODO: rerun validation after enabling mixed-precision; confirm accuracy isn’t degraded`
# 2. `FIXME: measure F1 change when we swap tokenizer to WordPiece—may hurt rare-word recall`
# 3. `XXX: compare BLEU before/after removing dropout layer to ensure translation quality holds`

# MACHINE LEARNING RELIABILITY (MLRelia): Model functionality depends on data quality; robustness measures should be in place.
# examples:
# 1. `TODO: add sanity-check to reject examples with >20 % missing features before prediction`
# 2. `FIXME: wrap model call in try/except and fall back to last-known good output when decoder crashes`
# 3. `XXX: create robustness test suite with Gaussian noise and FGSM perturbations to monitor accuracy drift`

# PROTOTYPE (Proto): Deploying small-scale prototypes into full systems can be dangerous.
# examples:
# 1. `TODO: replace the single-threaded demo server with a production-ready Gunicorn setup`
# 2. `FIXME: temp JSON files used for state—migrate to proper Redis cache before rollout`
# 3. `XXX: remove hard-coded API key left from prototype; load securely from env vars instead`
# """.strip()

PROMPT_HEAD = r"""
There are 23 types of Machine Learning (ML) SATD (Self-Admitted Technical Debt):

MACHINE LEARNING KNOWLEDGE (MLKnow): Machine-learning software carries plenty of unique challenges. Uneducated solutions by unaware developers may have to be revisited.
example: `TODO: Confirm whether layer-norm before attention actually boosts convergence for small-batch training on this dataset`

MODEL INTERPRETABILITY (MLInter): Machine-learning models are a black box, causing poor understanding of model functionality; this can lead to unknown behaviour.
example: `TODO: Generate SHAP summary plot so users can see feature impact`

MODEL CODE COMPREHENSION (MCC): Model code carries extra legibility concerns not present in traditional software (e.g., poorly-named temporary matrix variables).
example: `TODO: break up monolithic function train_eval_predict() into clearer, single-purpose helpers`

DUPLICATE MODEL CODE (DMC): Code duplication frequently occurs in model code.
example: `TODO: remove copy-pasted ConvBlock definition—reuse common_layers.ConvBlock instead`

DUPLICATE FEATURE EXTRACTION CODE (DFEC): Code duplication frequently occurs in feature-extraction code.
example: `TODO: merge extract_mfcc() and extract_mfcc_fast(); only differ in hop_length`

WEIGHT CONFIGURATION (Weight Conf): Editing code that involves the weights of a model, or configuring a model's weights directly.
example: `TODO: expose weight_decay as configurable argument instead of hard-coding 0.01`

LAYER CONFIGURATION (Layer Conf): Editing code that deals with model layers, or configuring layers themselves.
example: `TODO: swap BatchNorm for LayerNorm when depth > 24`

HYPER-PARAMETER CONFIGURATION (HP Conf): Configuring hyper-parameters of a model, or editing default values of off-the-shelf models.
example: `TODO: make learning_rate schedule (cosine vs. linear) selectable via CLI flag`

CODE DEPENDENCY (MLDepend): A needed change is blocked by an external dependency (library / subsystem).
example: `TODO: bump scikit-learn to ≥1.6 once conda forge wheels for arm64 become available`

GLUE CODE (GC): Supporting code written to interface with other code, inhibiting improvements due to dependent peculiarities.
example: `TODO: remove pandas-to-torch bridge once DataLoader accepts DataFrame directly`

CUSTOM DATA TYPE (CDT): Using data types provided by general-purpose packages can cause extensive inter-operability burden.
example: `TODO: replace custom SparseTensorDict with standard torch.sparse_coo_tensor before exporting`

MULTIPLE LANGUAGES (ML): Components written in other languages may introduce difficulties in ML development.
example: `TODO: port custom attention kernel from CUDA C++ to Metal for macOS deployment`

UNNECESSARY MODEL CODE (UMC): Model code that bottlenecks execution, is unreachable / deprecated, or is unnecessary and should be removed.
example: `TODO: delete legacy compute_grad_stats(); superseded by torch.autograd.profiler`

DATA PROCESSING CONFIGURATION (Data Conf): Configuring how data is processed, by editing the data directly or adding new processing steps.
example: `TODO: add configurable pipeline step to down-sample majority class before batching`

PLAIN OLD DATA TYPE (PODT): Using raw data types in ML causes confusion when interpreting processes.
example: `TODO: replace raw list of logits with a NamedTuple so callers know the order`

DATA STORAGE CONFIGURATION (DS Conf): Configuring how data is represented internally or stored externally.
example: `TODO: migrate checkpoint format from .pth to safetensors for safer serialization`

ABSTRACTION (Abs): Lack of abstractions in ML systems causes cascading changes when a component is modified.
example: `TODO: extract common scoring logic into BaseMetric so each model wrapper isn’t re-implementing it`

BOUNDARY EROSION (BE): Lack of boundaries between subsystems complicates maintenance and isolates changes poorly.
example: `TODO: migrate training-time logging out of Trainer class; should live in metrics.logger`

MODEL CODE MODIFIABILITY (MCM): Model code should enable easy maintenance and future modifications.
example: `TODO: move hyper-parameter defaults into config file so changing them doesn’t require code edits`

MODEL CODE REUSABILITY (MCR): Model code should be generalised for reuse in varying situations.
example: `TODO: abstract make_encoder() so VisionTransformer and ResNet builders share the same backbone factory`

PREDICTION QUALITY (PQ): Workflow changes may affect performance.
example: `TODO: rerun validation after enabling mixed-precision; confirm accuracy isn’t degraded`

MACHINE LEARNING RELIABILITY (MLRelia): Model functionality depends on data quality; robustness measures should be in place.
example: `TODO: add sanity-check to reject examples with >20 % missing features before prediction`

PROTOTYPE (Proto): Deploying small-scale prototypes into full systems can be dangerous.
example: `TODO: replace the single-threaded demo server with a production-ready Gunicorn setup`
""".strip()

PROMPT_TAIL = f"""
You are a strict ML-SATD classifier.
Do all reasoning internally and do NOT reveal it.
Decide the single best label by following this internal decision process, then output exactly ONE abbreviation on a single line with nothing else.

Internal decision process (for your internal use only; do not include in output):
1) Identify the primary signal in the comment and select candidates accordingly:
   • Data-related: processing/pipeline steps ⇒ {{Data Conf}}; raw/plain data types ⇒ {{PODT}}; representation/storage/schema ⇒ {{DS Conf}}.
   • Dependency & integration: external/library blocking ⇒ {{MLDepend}}; glue/integration shims ⇒ {{GC}}; custom data types ⇒ {{CDT}}; mixed languages ⇒ {{ML}}; unnecessary/dead/deprecated model code ⇒ {{UMC}}.
   • Configuration & parameters: weights ⇒ {{Weight Conf}}; layers/topology ⇒ {{Layer Conf}}; hyper-parameters/defaults ⇒ {{HP Conf}}.
   • Structure & evolvability: missing abstraction ⇒ {{Abs}}; boundary erosion ⇒ {{BE}}; ease of modifying model code ⇒ {{MCM}}; generality/reuse ⇒ {{MCR}}.
   • Duplication: model code duplication ⇒ {{DMC}}; feature-extraction duplication ⇒ {{DFEC}}.
   • Understanding & interpretability: lack of ML knowledge ⇒ {{MLKnow}}; model black-box / explainability concerns ⇒ {{MLInter}}.
   • Performance & robustness: prediction quality/accuracy ⇒ {{PQ}}; reliability/robustness/data-quality guards ⇒ {{MLRelia}}.
   • Prototype: prototype used beyond its scope ⇒ {{Proto}}.
2) If multiple candidates match, choose the one that represents the main maintenance obstacle implied by the comment.
3) If still tied, prefer the most specific label whose remediation is the most localized (e.g., pick {{Layer Conf}} over {{HP Conf}} if the text explicitly mentions layers).
4) Output policy: return exactly ONE abbreviation on a single line, with no other text.
   Allowed abbreviations:
   {', '.join(sorted(ABBR_SET))}
""".strip()

def build_prompt(comment: str) -> str:
    return (
        f"{PROMPT_HEAD}\n\n{PROMPT_TAIL}\n\n"
        "Code Comment:\n"
        f"```text\n{comment}\n```\n\n"
        "#Abbreviation:"
    )

def extract_abbr_from_text(text: str) -> str:
    if not text:
        return "<UNK>"
    # 规范化直接命中
    cz = _canon(text)
    if cz in {_canon(a): a for a in ABBR_SET}:
        # 把 cz 映回真实缩写
        for a in ABBR_SET:
            if _canon(a) == cz:
                return a
    # 正则宽松匹配（允许空白差异）
    for abbr in ABBR_SET:
        pat = r'\b' + re.sub(r'\s+', r'\\s*', re.escape(abbr)) + r'\b'
        if re.search(pat, text, flags=re.IGNORECASE):
            return abbr
    # 全称也兜底匹配
    for full, abbr in FULL2ABBR.items():
        pat = r'\b' + re.sub(r'\s+', r'\\s*', re.escape(full)) + r'\b'
        if re.search(pat, text, flags=re.IGNORECASE):
            return abbr
    return "<UNK>"

# -------- 4. 加载模型（官方推荐：chat template） --------
print("[INFO] Loading model …")

pipe = pipeline(
    "text-generation",
    model=MODEL_PATH,
    torch_dtype="auto",
    device_map="auto",
)

# -------- 5. 分类函数（按官方做法解码+取 final） --------
@torch.inference_mode()
def classify(comment: str):
    messages = [
        {"role": "system", "content":
         "You are a strict classifier. Reasoning: high. "
         "Output ONLY the abbreviation, one line."},
        {"role": "user", "content": build_prompt(comment)},
    ]

    

    outputs =  pipe(
    messages,
    max_new_tokens=MAX_NEW_TOKENS,
    )

    # 关键：不要跳过特殊标记，便于找到 final 段
    answer = outputs[0]["generated_text"][-1]['content']
    raw_answer = answer.split('assistantfinal')[-1]
    # print(raw_answer)
    abbr = extract_abbr_from_text(raw_answer)

    return abbr,raw_answer

# -------- 6. 主流程 --------
df = pd.read_csv(CSV_PATH)

rows = []
unk_cnt = 0
gold_unmapped = 0

print("[INFO] Classifying …")
for cmt, gold in tqdm(zip(df["Comment Text"], df["ML TD Type"]),
                      total=len(df), ncols=80):
    # 1) 统一金标到“全称”
    real_full = normalize_to_full(str(gold))
    if real_full is None:
        gold_unmapped += 1
        # 跳过无法映射的样本，避免评估崩掉
        continue

    # 2) 推理+解析缩写；再映回全称
    abbr, raw = classify(str(cmt))
    pred_full = ABBR2FULL.get(abbr, None)
    if pred_full is None:
        unk_cnt += 1
        if PRINT_UNPARSED:
            with open(RUN_DIR/"unparsed.txt", "a", encoding="utf-8") as f:
                f.write(raw + "\n")
        pred_full = ABBR2FULL[FALLBACK_ABBR]  # 回退：用 Data Conf 的全称

    rows.append({
        "comment": cmt,
        "real": real_full,
        "pred": pred_full,
        "raw_output": raw
    })

pred_df = pd.DataFrame(rows)

# 若全部样本都被跳过，给出友好提示并退出
if pred_df.empty:
    print("\n[WARN] No valid samples after label normalization. "
          "Please check the dataset's label names (column 'ML TD Type').")
else:
    # 终端报告
    report_text = classification_report(
        pred_df["real"], pred_df["pred"],
        labels=LABELS, zero_division=0, digits=3
    )
    print("\n[INFO] Overall metrics:\n", report_text)

    # -------- 7. 保存结果 --------
    pred_df.to_csv(RUN_DIR/"predictions.csv", index=False)

    # 混淆矩阵（现在 y_true 一定包含 LABELS 的子集，不会再抛错）
    cm = confusion_matrix(pred_df["real"], pred_df["pred"], labels=LABELS)
    pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(RUN_DIR/"confusion_matrix.csv")

    # classification_report → CSV，并附加统计信息
    report_dict = classification_report(
        pred_df["real"], pred_df["pred"],
        labels=LABELS, zero_division=0, output_dict=True
    )
    rep_df = pd.DataFrame(report_dict).transpose()
    rep_df.index.name = "label"
    rep_df = rep_df.reset_index()
    rep_df["info"] = ""

    info_lines = [
        f"Unrecognized outputs mapped to '{FALLBACK_ABBR}': {unk_cnt}",
        f"Gold labels unmapped (skipped): {gold_unmapped}",
        f"Total evaluated samples: {len(pred_df)} / original {len(df)}"
    ]
    rep_df.loc[len(rep_df)] = {
        "label": "[INFO]",
        "precision": "",
        "recall": "",
        "f1-score": "",
        "support": "",
        "info": " | ".join(info_lines)
    }
    rep_df.to_csv(RUN_DIR/"report.csv", index=False)

print(f"\n[INFO] Files saved in {RUN_DIR}")
print(f"[INFO] Unrecognized mapped to '{FALLBACK_ABBR}': {unk_cnt}")
print(f"[INFO] Gold labels unmapped (skipped): {gold_unmapped}")

gc.collect(); torch.cuda.empty_cache()
