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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
MAX_NEW_TOKENS    = 4096
MAX_INPUT_TOKENS  = 4096

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
PROMPT_HEAD = r"""
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
#     # 规范化直接命中
#     cz = _canon(text)
#     if cz in {_canon(a): a for a in ABBR_SET}:
#         # 把 cz 映回真实缩写
#         for a in ABBR_SET:
#             if _canon(a) == cz:
#                 return a
#     # 正则宽松匹配（允许空白差异）
#     for abbr in ABBR_SET:
#         pat = r'\b' + re.sub(r'\s+', r'\\s*', re.escape(abbr)) + r'\b'
#         if re.search(pat, text, flags=re.IGNORECASE):
#             return abbr
#     # 全称也兜底匹配
#     for full, abbr in FULL2ABBR.items():
#         pat = r'\b' + re.sub(r'\s+', r'\\s*', re.escape(full)) + r'\b'
#         if re.search(pat, text, flags=re.IGNORECASE):
#             return abbr
#     return "<UNK>"
# [R13/R16 修订]
def extract_abbr_from_text(raw: str) -> str:
    return parse_label(raw, LABEL_ABBR)["label"]

# -------- 4. 加载模型（官方推荐：chat template） --------
print("[INFO] Loading model …")

pipe = pipeline(
    "text-generation",
    model=MODEL_PATH,
    torch_dtype="auto",
    device_map="auto",
)

# -------- 5. 分类函数（按官方做法解码+取 final） --------
# [R13/R16/R23/R24/R29/R31/R33 原代码保留]
# @torch.inference_mode()
# def classify(comment: str):
#     messages = [
#         {"role": "system", "content":
#          "You are a strict classifier. Reasoning: low. "
#          "Output ONLY the abbreviation, one line."},
#         {"role": "user", "content": build_prompt(comment)},
#     ]
#
#
#
#     outputs =  pipe(
#     messages,
#     max_new_tokens=MAX_NEW_TOKENS,
#     )
#
#     # 关键：不要跳过特殊标记，便于找到 final 段
#     answer = outputs[0]["generated_text"][-1]['content']
#     raw_answer = answer.split('assistantfinal')[-1]
#     # print(raw_answer)
#     abbr = extract_abbr_from_text(raw_answer)
#
#     return abbr,raw_answer
# [R13/R16/R23/R24/R29/R31/R33 修订]
# R45 OFFICIAL PROFILE: previous function retained in place.
# def classify(comment: str):
#     messages = [{'role': 'system', 'content': 'You are a strict classifier. Output ONLY the abbreviation, one line.'}, {'role': 'user', 'content': build_prompt(comment)}]
#     event = audited_generate(pipe.model, pipe.tokenizer, messages,
#                              {"generation_config": getattr(pipe, "generation_config", pipe.model.generation_config), 'max_new_tokens': MAX_NEW_TOKENS}, audit=_REVISION_AUDIT,
#                              max_input_tokens=None, native_thinking=False,
#                              reasoning_effort=os.environ.get('SATD_REASONING_EFFORT', 'low'), stage="classification")
#     parsed = parse_classification(event["answer_text"], LABEL_ABBR, with_reason=False)
#     if event["parse_status"] != "ok":
#         parsed.update(label="<UNK>", parse_ok=False, status=event["parse_status"])
#     return {**parsed, "raw_text": event["raw_text"], "generation_parse_status": event["parse_status"]}

# R45 OFFICIAL PROFILE: generation parameters follow the model and native mode.
def classify(comment: str):
    messages = [{'role': 'system', 'content': 'You are a strict classifier. Output ONLY the abbreviation, one line.'}, {'role': 'user', 'content': build_prompt(comment)}]
    event = audited_generate(pipe.model, pipe.tokenizer, messages, {'max_new_tokens': MAX_NEW_TOKENS}, audit=_REVISION_AUDIT, max_input_tokens=None, native_thinking=False, stage='classification', official_profile=True)
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
#     # 1) 统一金标到“全称”
#     real_full = normalize_to_full(str(gold))
#     if real_full is None:
#         gold_unmapped += 1
#         # 跳过无法映射的样本，避免评估崩掉
#         continue
#
#     # 2) 推理+解析缩写；再映回全称
#     abbr, raw = classify(str(cmt))
#     pred_full = ABBR2FULL.get(abbr, None)
#     if pred_full is None:
#         unk_cnt += 1
#         if PRINT_UNPARSED:
#             with open(RUN_DIR/"unparsed.txt", "a", encoding="utf-8") as f:
#                 f.write(raw + "\n")
#         pred_full = ABBR2FULL[FALLBACK_ABBR]  # 回退：用 Data Conf 的全称
#
#     rows.append({
#         "comment": cmt,
#         "real": real_full,
#         "pred": pred_full,
#         "raw_output": raw
#     })
#
# pred_df = pd.DataFrame(rows)
#
# # 若全部样本都被跳过，给出友好提示并退出
# if pred_df.empty:
#     print("\n[WARN] No valid samples after label normalization. "
#           "Please check the dataset's label names (column 'ML TD Type').")
# else:
#     # 终端报告
#     report_text = classification_report(
#         pred_df["real"], pred_df["pred"],
#         labels=LABELS, zero_division=0, digits=3
#     )
#     print("\n[INFO] Overall metrics:\n", report_text)
#
#     # -------- 7. 保存结果 --------
#     pred_df.to_csv(RUN_DIR/"predictions.csv", index=False)
#
#     # 混淆矩阵（现在 y_true 一定包含 LABELS 的子集，不会再抛错）
#     cm = confusion_matrix(pred_df["real"], pred_df["pred"], labels=LABELS)
#     pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(RUN_DIR/"confusion_matrix.csv")
#
#     # classification_report → CSV，并附加统计信息
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
#     "reasoning_mode": "gpt_oss_native_reasoning",
#     "fallback_abbr": FALLBACK_ABBR,
#     "metrics_policy": "Primary: existing final fallback predictions. Additional failure_as_error metrics mark invalid final parses wrong.",
#     "input_policy": "Full templated input, including demonstrations/comment/code context; actual context with 64 safety tokens; requested generation cap is reduced to remaining context and the effective cap is recorded; no input truncation.",
#     "parser_policy": "Unique complete canonical abbreviation/full-name or the explicitly requested answer field; never search reasoning.",
#     "unknown_count_before_fallback": unk_cnt,
# }, ensure_ascii=False, indent=2), encoding="utf-8")

(RUN_DIR / 'settings.json').write_text(json.dumps({**_REVISION_AUDIT.settings(), 'model': MODEL_PATH, 'csv': CSV_PATH, 'run_dir': str(RUN_DIR), 'max_new_tokens': MAX_NEW_TOKENS, 'reasoning_mode': 'gpt_oss_native_reasoning', 'fallback_abbr': FALLBACK_ABBR, 'metrics_policy': 'Primary: existing final fallback predictions. Additional failure_as_error metrics mark invalid final parses wrong.', 'input_policy': 'Full templated input, including demonstrations/comment/code context; actual context with 64 safety tokens; requested generation cap is reduced to remaining context and the effective cap is recorded; no input truncation.', 'parser_policy': 'Unique complete canonical abbreviation/full-name or the explicitly requested answer field; never search reasoning.', 'unknown_count_before_fallback': unk_cnt, 'generation_profile_policy': 'R45: official recommendation per model/mode; otherwise fresh checkpoint defaults; actual profiles in runtime_manifest.json and generation_events.jsonl. Existing stage output caps preserved.'}, ensure_ascii=False, indent=2), encoding='utf-8')
print(f"[INFO] Revised results saved to {RUN_DIR}")
