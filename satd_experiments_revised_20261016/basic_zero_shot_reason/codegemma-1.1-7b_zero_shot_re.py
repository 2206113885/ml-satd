#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_satd_gemma2_zero_shot.py  (改版)
-----------------------------------
零样本 · codegemma-1.1-7b-it · 纯对话 prompt
输出目录: results/run-YYYYmmdd-HHMMSS-模型名
保存: predictions.csv, confusion_matrix.csv, report.csv, settings.json
新增: predictions.csv 增加 reason 列（模型给出的简短理由）
"""
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
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_float32_matmul_precision('high')

# ---------- 0. 路径（仅两处需改） ----------
os.environ["CUDA_VISIBLE_DEVICES"] = "3"         # GPU ID
CSV_PATH   = "/data/yangzhifei/pro/ML_SATD/dataset/sampled_label_dataset_normalized_rmvtodo.csv"
MODEL_PATH = "/data/yangzhifei/pro/ML_SATD/models/codegemma-1.1-7b-it"

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

PRINT_UNPARSED  = True
FALLBACK_ABBR   = "Data Conf"
MAX_NEW_TOKENS  = 512  # 原值保留；我们会在解析端限制，仅取两行

# ---------- 1. 标签映射 ----------
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

# -------- 2. Prompt 头（描述，无 few-shot） ----
PROMPT_HEAD = r"""
There are 23 types of Machine Learning (ML) SATD (Self-Admitted Technical Debt):

MACHINE LEARNING KNOWLEDGE (MLKnow): Machine-learning software carries plenty of unique challenges. Uneducated solutions by unaware developers may have to be revisited.

MODEL INTERPRETABILITY (MLInter): Machine-learning models are a black box, causing poor understanding of model functionality; this can lead to unknown behaviour.

MODEL CODE COMPREHENSION (MCC): Model code carries extra legibility concerns not present in traditional software.

DUPLICATE MODEL CODE (DMC): Code duplication frequently occurs in model code.

DUPLICATE FEATURE EXTRACTION CODE (DFEC): Code duplication frequently occurs in feature-extraction code.

WEIGHT CONFIGURATION (Weight Conf): Editing code that involves the weights of a model, or configuring a model's weights directly.

LAYER CONFIGURATION (Layer Conf): Editing code that deals with model layers, or configuring layers themselves.

HYPER-PARAMETER CONFIGURATION (HP Conf): Configuring hyper-parameters of a model, or editing default values of off-the-shelf models.

CODE DEPENDENCY (MLDepend): A needed change is blocked by an external dependency.

GLUE CODE (GC): Supporting code written to interface with other code.

CUSTOM DATA TYPE (CDT): Using data types provided by general-purpose packages can cause extensive inter-operability burden.

MULTIPLE LANGUAGES (ML): Components written in other languages may introduce difficulties in ML development.

UNNECESSARY MODEL CODE (UMC): Model code that bottlenecks execution, is unreachable / deprecated, or is unnecessary and should be removed.

DATA PROCESSING CONFIGURATION (Data Conf): Configuring how data is processed by editing data directly or adding processing steps.

PLAIN OLD DATA TYPE (PODT): Using raw data types in ML causes confusion when interpreting processes.

DATA STORAGE CONFIGURATION (DS Conf): Configuring how data is represented internally or stored externally.

ABSTRACTION (Abs): Lack of abstractions in ML systems causes cascading changes.

BOUNDARY EROSION (BE): Lack of boundaries between subsystems complicates maintenance.

MODEL CODE MODIFIABILITY (MCM): Model code should enable easy maintenance and future modifications.

MODEL CODE REUSABILITY (MCR): Model code should be generalised for reuse in varying situations.

PREDICTION QUALITY (PQ): Workflow changes may affect performance.

MACHINE LEARNING RELIABILITY (MLRelia): Model functionality depends on data quality; robustness measures should be in place.

PROTOTYPE (Proto): Deploying small-scale prototypes into full systems can be dangerous.
""".strip()

# 收紧输出：两行固定格式（Abbreviation + Reason）
PROMPT_TAIL = (
    "Please determine which type of ML Self-Admitted Technical Debt (SATD) the following Code Comment belongs to.\n"
    "Return your answer in EXACTLY two lines and nothing else:\n"
    f"Abbreviation: <ONE from: {', '.join(sorted(ABBR_SET))}>\n"
    "Reason: <<=12 words; short, specific, and grounded in the comment>\n"
).strip()

def build_prompt(comment: str) -> str:
    return (
        f"{PROMPT_HEAD}\n\n{PROMPT_TAIL}\n\n"
        "Code Comment:\n\"\"\"\n"
        f"{comment}\n\"\"\"\n"
    )

# ---------- 3. 解析函数 ----------
_ABBR_PREFIX = re.compile(r'^\s*(abbreviation|abbr)\s*[:：-]\s*', re.I)
_REASON_PREFIX = re.compile(r'^\s*(reason)\s*[:：-]\s*', re.I)

# [R13/R16 原代码保留]
# def simple_extract(raw: str) -> str:
#     token = re.sub(r"[^a-z]", "", str(raw).lower())
#     if token in ABBR_LOWER:
#         return ABBR_LOWER[token]
#     if raw in ABBR2FULL:              # 精确匹配（保留旧逻辑）
#         return raw
#     join = str(raw).replace(" ", "")
#     if join.lower() in ABBR_LOWER:
#         return ABBR_LOWER[join.lower()]
#     return "<UNK>"
# [R13/R16 修订]
def simple_extract(raw: str) -> str:
    return parse_label(raw, LABEL_ABBR)["label"]

# [R13/R16 原代码保留]
# def parse_abbr_and_reason(full_text: str):
#     """
#     解析两行输出：
#       第1行：Abbreviation: <abbr>
#       第2行：Reason: <text>
#     若缺失 Reason，则尝试全文里 'because|reason' 后的内容。
#     返回：(abbr, abbr_raw, reason)
#     """
#     lines = [l.strip() for l in str(full_text).strip().splitlines() if l.strip()]
#     abbr_raw, reason = "", ""
#     if lines:
#         abbr_raw = _ABBR_PREFIX.sub("", lines[0]).strip()
#     if len(lines) >= 2:
#         reason = _REASON_PREFIX.sub("", lines[1]).strip()
#     if not reason:
#         m = re.search(r'(?:because|reason)\s*[:：-]?\s*(.+)$', str(full_text), flags=re.I)
#         if m:
#             reason = m.group(1).strip()
#     if len(reason) > 500:
#         reason = reason[:500].rstrip()
#
#     abbr = simple_extract(abbr_raw if abbr_raw else (lines[0] if lines else ""))
#     return abbr, abbr_raw, reason
# [R13/R16 修订]
def parse_abbr_and_reason(raw: str):
    result = parse_classification(raw, LABEL_ABBR, with_reason=True)
    return result["label"], result["label"], result.get("reason", "")

# ---------- 4. 载入模型 ----------
print("[INFO] Loading model …")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# ---------- 5. classify ----------
# [R13/R16/R23/R24/R29/R31/R33 原代码保留]
# @torch.inference_mode()
# def classify(comment: str):
#     messages = [
#         {"role": "user", "content": build_prompt(comment)},   # codegemma 只要 user
#     ]
#     text = tokenizer.apply_chat_template(messages,
#                                          tokenize=False,
#                                          add_generation_prompt=True)
#     inputs = tokenizer(text, return_tensors="pt",
#                        truncation=True, max_length=4096).to(model.device)
#
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=MAX_NEW_TOKENS,
#     )
#     full = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
#                             skip_special_tokens=True).strip()
#
#     # 解析缩写与理由
#     abbr, abbr_raw, reason = parse_abbr_and_reason(full)
#
#     # raw_first 保留首行（便于排查）
#     raw_first = abbr_raw if abbr_raw else (full.split("\n", 1)[0] if full else "")
#     return abbr, raw_first, reason, full
# [R13/R16/R23/R24/R29/R31/R33 修订]
# R45 OFFICIAL PROFILE: previous function retained in place.
# def classify(comment: str):
#     messages = [{'role': 'user', 'content': build_prompt(comment)}]
#     event = audited_generate(model, tokenizer, messages,
#                              {'max_new_tokens': MAX_NEW_TOKENS}, audit=_REVISION_AUDIT,
#                              max_input_tokens=4096, native_thinking=False,
#                              reasoning_effort=None, stage="classification")
#     parsed = parse_classification(event["answer_text"], LABEL_ABBR, with_reason=True)
#     if event["parse_status"] != "ok":
#         parsed.update(label="<UNK>", parse_ok=False, status=event["parse_status"])
#     return {**parsed, "raw_text": event["raw_text"], "generation_parse_status": event["parse_status"]}

# R45 OFFICIAL PROFILE: generation parameters follow the model and native mode.
def classify(comment: str):
    messages = [{'role': 'user', 'content': build_prompt(comment)}]
    event = audited_generate(model, tokenizer, messages, {'max_new_tokens': MAX_NEW_TOKENS}, audit=_REVISION_AUDIT, max_input_tokens=4096, native_thinking=False, stage='classification', official_profile=True)
    parsed = parse_classification(event["answer_text"], LABEL_ABBR, with_reason=True)
    if event["parse_status"] != "ok":
        parsed.update(label="<UNK>", parse_ok=False, status=event["parse_status"])
    return {**parsed, "raw_text": event["raw_text"], "generation_parse_status": event["parse_status"]}

# ---------- 6. 主流程 ----------
# [R22/R31/R33 原代码保留]
# df = pd.read_csv(CSV_PATH)
# rows, unk_cnt = [], 0
# print("[INFO] Classifying …")
# for cmt, gold in tqdm(zip(df["Comment Text"], df["ML TD Type"]),
#                       total=len(df), ncols=80):
#     abbr, raw_first, reason, full = classify(cmt)
#     if abbr == "<UNK>":
#         unk_cnt += 1
#         if PRINT_UNPARSED:
#             with open(RUN_DIR/"unparsed.txt", "a", encoding="utf-8") as f:
#                 f.write(raw_first + "\n")
#         abbr = FALLBACK_ABBR
#     rows.append({
#         "comment": cmt,
#         "real": ABBR2FULL.get(gold, gold),
#         "pred": ABBR2FULL[abbr],
#         "reason": reason,          # 新增列：模型给出的简短理由
#         "raw_output": raw_first
#     })
#
# pred_df = pd.DataFrame(rows)
#
# # 终端打印文本版
# report_text = classification_report(pred_df["real"], pred_df["pred"],
#                                     labels=LABELS, zero_division=0, digits=3)
# print("\n[INFO] Overall metrics:\n", report_text)
#
# # ---------- 7. 保存结果 ----------
# # 7.1 逐样本预测
# pred_df.to_csv(RUN_DIR/"predictions.csv", index=False)
#
# # 7.2 混淆矩阵（行真列预测，顺序与 LABELS 一致）
# pd.DataFrame(
#     confusion_matrix(pred_df["real"], pred_df["pred"], labels=LABELS),
#     index=LABELS, columns=LABELS
# ).to_csv(RUN_DIR/"confusion_matrix.csv")
#
# # 7.3 把 classification_report 保存为 CSV，并追加 INFO 行
# report_dict = classification_report(pred_df["real"], pred_df["pred"],
#                                     labels=LABELS, zero_division=0, output_dict=True)
# rep_df = pd.DataFrame(report_dict).transpose()
# rep_df.index.name = "label"
# rep_df = rep_df.reset_index()
# rep_df["info"] = ""  # 先占位一列
#
# info_line = f"Unrecognized mapped to '{FALLBACK_ABBR}': {unk_cnt}/{len(df)}"
# rep_df.loc[len(rep_df)] = {
#     "label": "[INFO]",
#     "precision": "",
#     "recall": "",
#     "f1-score": "",
#     "support": "",
#     "info": info_line
# }
# rep_df.to_csv(RUN_DIR/"report.csv", index=False)
#
# # 7.4 运行配置
# (Path(RUN_DIR/"settings.json")).write_text(json.dumps({
#     "csv": CSV_PATH,
#     "model": MODEL_PATH,
#     "model_name": MODEL_NAME,
#     "run_dir": str(RUN_DIR),
#     "max_new_tokens": MAX_NEW_TOKENS,
#     "fallback_abbr": FALLBACK_ABBR,
#     "output_format": "two-lines (Abbreviation / Reason <=12 words)"
# }, indent=2))
#
# print(f"\n[INFO] Files saved in {RUN_DIR}")
# print(f"[INFO] Unrecognized mapped to '{FALLBACK_ABBR}': {unk_cnt}/{len(df)}")
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
    reason = result.get("reason", "")
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
# R45 OFFICIAL PROFILE: previous settings writer retained.
# (RUN_DIR/"settings.json").write_text(json.dumps({
#     **_REVISION_AUDIT.settings(),
#     "model": MODEL_PATH, "csv": CSV_PATH, "run_dir": str(RUN_DIR),
#     "max_new_tokens": MAX_NEW_TOKENS,
#     "reasoning_mode": 'nonthinking',
#     "fallback_abbr": FALLBACK_ABBR,
#     "metrics_policy": "Primary: existing final fallback predictions. Additional failure_as_error metrics mark invalid final parses wrong.",
#     "input_policy": "Full templated input, including demonstrations/comment/code context; actual context with 64 safety tokens; requested generation cap is reduced to remaining context and the effective cap is recorded; no input truncation.",
#     "parser_policy": "Unique complete canonical abbreviation/full-name or the explicitly requested answer field; never search reasoning.",
#     "unknown_count_before_fallback": unk_cnt,
# }, ensure_ascii=False, indent=2), encoding="utf-8")

(RUN_DIR / 'settings.json').write_text(json.dumps({**_REVISION_AUDIT.settings(), 'model': MODEL_PATH, 'csv': CSV_PATH, 'run_dir': str(RUN_DIR), 'max_new_tokens': MAX_NEW_TOKENS, 'reasoning_mode': 'nonthinking', 'fallback_abbr': FALLBACK_ABBR, 'metrics_policy': 'Primary: existing final fallback predictions. Additional failure_as_error metrics mark invalid final parses wrong.', 'input_policy': 'Full templated input, including demonstrations/comment/code context; actual context with 64 safety tokens; requested generation cap is reduced to remaining context and the effective cap is recorded; no input truncation.', 'parser_policy': 'Unique complete canonical abbreviation/full-name or the explicitly requested answer field; never search reasoning.', 'unknown_count_before_fallback': unk_cnt, 'generation_profile_policy': 'R45: official recommendation per model/mode; otherwise fresh checkpoint defaults; actual profiles in runtime_manifest.json and generation_events.jsonl. Existing stage output caps preserved.'}, ensure_ascii=False, indent=2), encoding='utf-8')
print(f"[INFO] Revised results saved to {RUN_DIR}")
