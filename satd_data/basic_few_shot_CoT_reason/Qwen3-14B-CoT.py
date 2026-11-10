#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re, os, gc, json
from pathlib import Path
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# -------- 0. 基础路径（仅此处改） --------
os.environ["CUDA_VISIBLE_DEVICES"] = "1"        # GPU 号
CSV_PATH   = "/data/yangzhifei/pro/ML_SATD/dataset/sampled_label_dataset_normalized_rmvtodo.csv"  # ← 修改
MODEL_PATH = "/data/yangzhifei/pro/ML_SATD/models/Qwen3-14B"                 # ← 修改

# 目录名包含时间戳 + 模型名
MODEL_NAME = Path(MODEL_PATH).name
RUN_DIR    = Path("results") / f"run-{datetime.now():%Y%m%d-%H%M%S}-{MODEL_NAME}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

# -------- 1. 其它参数 --------
PRINT_UNPARSED = True
FALLBACK_ABBR  = "Data Conf"
MAX_NEW_TOKENS = 32768   # 分类阶段：保持不变
REASON_MAX_TOKENS = 512  # 新增：理由阶段的生成上限（可适当增减，保证一句话即可）

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
LABELS     = list(LABEL_ABBR.keys())
ABBR_SET   = set(ABBR2FULL.keys())
ABBR_LOWER = {a.lower():a for a in ABBR_SET}

# ============ 2. Prompt 头（仅描述，无示例） ============
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

# —— 保持你给定的“thinking 模式 + 内部决策流程（不外露）” —— #
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

# -------- 3. build_prompt（分类阶段：保持只输出缩写） --------
def build_prompt(comment: str) -> str:
    return (
        f"{PROMPT_HEAD}\n\n{PROMPT_TAIL}\n\n"
        "Code Comment:"
        f"{comment}\n\"\"\"\n\nAbbreviation:"
    )

# -------- 4. 解析函数（宽容版） --------
def simple_extract(raw: str) -> str:
    first = re.split(r"[\r\n]", raw.strip())[0]
    token = re.sub(r"[^a-z]", "", first.lower())
    if token in ABBR_LOWER:
        return ABBR_LOWER[token]
    raw_strip = first.strip()
    if raw_strip in ABBR2FULL:
        return raw_strip
    join = raw_strip.replace(" ", "")
    if join.lower() in ABBR_LOWER:
        return ABBR_LOWER[join.lower()]
    return "<UNK>"

# -------- 5. 加载模型 --------
print("[INFO] Loading model …")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# -------- 6A. 分类：保持 enable_thinking=True，只取可见输出 --------
@torch.inference_mode()
def classify(comment: str):
    messages = [
        {"role": "user",   "content": build_prompt(comment)},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True   # 保持充分思考，但不外露
    )
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, max_length=4096).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.6,   # 保持你的参数
        top_p=0.95,
        top_k=20,
    )
    raw_all = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                               skip_special_tokens=True).strip()
    # 仅取可见部分（</think> 之后），不外露思维链
    visible = raw_all.split('</think>')[-1].strip()
    abbr = simple_extract(visible)
    return abbr, visible

# -------- 6B. 生成简短 reason（单独一次短生成，不改变其它超参） --------
def build_reason_prompt(comment: str, abbr: str) -> str:
    full = ABBR2FULL.get(abbr, abbr)
    return (
        f"{PROMPT_HEAD}\n\n"
        f"A label has been chosen: {full} ({abbr}).\n"
        "In one short sentence, justify why the comment fits this label.\n"
        "Do not reveal internal reasoning or steps.\n"
        "Output format: Reason: <short justification>\n\n"
        "Code Comment:\n\"\"\"\n"
        f"{comment}\n\"\"\"\n\n"
        "Answer:"
    )

def extract_reason(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    line = text.strip().splitlines()[0].strip()
    line = re.sub(r'^\s*(?:Reason|原因)\s*[:：]\s*', '', line, flags=re.I).strip()
    # 控制得更短一些（避免跑偏）
    return line[:300]

@torch.inference_mode()
def generate_reason(comment: str, abbr: str) -> str:
    if abbr not in ABBR_SET:
        return ""
    messages = [{"role": "user", "content": build_reason_prompt(comment, abbr)}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, max_length=4096).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=min(REASON_MAX_TOKENS, MAX_NEW_TOKENS),
        temperature=0.6,  # 与分类阶段一致
        top_p=0.95,
        top_k=20,
    )
    raw = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                           skip_special_tokens=True).strip()
    return extract_reason(raw)

# -------- 7. 主流程 --------
df = pd.read_csv(CSV_PATH)
rows, unk_cnt = [], 0

print("[INFO] Classifying …")
for cmt, gold in tqdm(zip(df["Comment Text"], df["ML TD Type"]),
                      total=len(df), ncols=80):
    # 1) 分类（thinking 模式，外显只含缩写）
    abbr, visible = classify(str(cmt))
    final_abbr = abbr if abbr in ABBR_SET else FALLBACK_ABBR
    if abbr == "<UNK>":
        unk_cnt += 1
        if PRINT_UNPARSED:
            with open(RUN_DIR/"unparsed.txt", "a", encoding="utf-8") as f:
                f.write(visible + "\n")

    # 2) 简短理由（summary rationale），一句话
    reason = generate_reason(str(cmt), final_abbr)

    rows.append({
        "comment": cmt,
        "real": ABBR2FULL.get(gold, gold),
        "pred": ABBR2FULL[final_abbr],
        "raw_output": visible,    # 分类阶段可见输出（仍仅是一行缩写）
        "reason": reason,       # 新增：简短理由
        
    })

pred_df = pd.DataFrame(rows)

# 终端打印（保持原有的文本版）
report_text = classification_report(pred_df["real"], pred_df["pred"],
                                    labels=LABELS, zero_division=0, digits=3)
print("\n[INFO] Overall metrics:\n", report_text)

# -------- 8. 保存结果 --------
# 8.1 逐样本预测（包含 reason 列）
pred_df.to_csv(RUN_DIR/"predictions.csv", index=False)

# 8.2 混淆矩阵
pd.DataFrame(
    confusion_matrix(pred_df["real"], pred_df["pred"], labels=LABELS),
    index=LABELS, columns=LABELS
).to_csv(RUN_DIR/"confusion_matrix.csv")

# 8.3 将 classification_report 保存为 CSV（含 Unrecognized 信息）
report_dict = classification_report(pred_df["real"], pred_df["pred"],
                                    labels=LABELS, zero_division=0, output_dict=True)
rep_df = pd.DataFrame(report_dict).transpose()
rep_df.index.name = "label"
rep_df = rep_df.reset_index()
rep_df["info"] = ""
info_line = f"Unrecognized mapped to '{FALLBACK_ABBR}': {unk_cnt}/{len(df)}"
rep_df.loc[len(rep_df)] = {
    "label": "[INFO]",
    "precision": "",
    "recall": "",
    "f1-score": "",
    "support": "",
    "info": info_line
}
rep_df.to_csv(RUN_DIR/"report.csv", index=False)

# 8.4 保存设置
(Path(RUN_DIR/"settings.json")).write_text(json.dumps({
    "csv": CSV_PATH,
    "model": MODEL_PATH,
    "run_dir": str(RUN_DIR),
    "max_new_tokens": MAX_NEW_TOKENS,
    "reason_max_tokens": REASON_MAX_TOKENS,
    "fallback_abbr": FALLBACK_ABBR,
    "thinking_mode_for_classification": True,
    "reason_generated_in_second_pass": True
}, indent=2))

print(f"\n[INFO] Files saved in {RUN_DIR}")
print(f"[INFO] Unrecognized mapped to '{FALLBACK_ABBR}': {unk_cnt}")
gc.collect(); torch.cuda.empty_cache()
