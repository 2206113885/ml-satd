#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在直接 23 类分类的基础上，同时输出“标签产生的简短原因”到 predictions.csv 的 reason 列。
- 不修改任何模型推理参数（temperature/top_p/top_k/max_new_tokens 保持原值）
- 通过提示词要求模型按两行固定格式返回：
    Abbreviation: <缩写>
    Reason: <<=10 词的简短理由>
- 解析器优先从上述键值对抓取；回退到按行解析，确保稳健。
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

# -------- 0. 基础路径（仅此处改） --------
os.environ["CUDA_VISIBLE_DEVICES"] = "2"        # GPU 号
CSV_PATH   = "/data/yangzhifei/pro/ML_SATD/dataset/sampled_label_dataset_normalized_rmvtodo.csv"  # ← 修改
MODEL_PATH = "/data/yangzhifei/pro/ML_SATD/models/Qwen3-8B"                                       # ← 修改

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

# -------- 1. 其它参数（不改动） --------
PRINT_UNPARSED = True
FALLBACK_ABBR  = "Data Conf"
MAX_NEW_TOKENS = 128

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

# —— 改这里：要求模型按两行固定格式输出 缩写 + 简短原因（≤10词）——
PROMPT_TAIL = f"""
Please determine which type of ML Self-Admitted Technical Debt (SATD) the following Code Comment belongs to.
Choose exactly ONE abbreviation from:
{', '.join(sorted(ABBR_SET))}

Return your answer in EXACTLY two lines:
Abbreviation: <ONE abbreviation from the list>
Reason: <very brief justification (<= 80 words), no step-by-step reasoning>
""".strip()

# -------- 3. build_prompt（修正注释包裹 + 两行输出说明） --------
def build_prompt(comment: str) -> str:
    return (
        f"{PROMPT_HEAD}\n\n{PROMPT_TAIL}\n\n"
        "Code Comment:\n\"\"\"\n"
        f"{comment}\n\"\"\"\n"
    )

# -------- 4. 解析函数（增强：能识别 'Abbreviation:' 前缀，并抓取 'Reason:'） --------
# [R13/R16 原代码保留]
# def simple_extract(raw: str) -> str:
#     """从一行或一句文本里尽量提取我们的 23 种缩写之一。"""
#     first = re.split(r"[\r\n]", raw.strip())[0]
#     # 去掉常见前缀
#     first = re.sub(r'^\s*(Abbreviation|Label)\s*:\s*', '', first, flags=re.I).strip()
#     token = re.sub(r"[^a-z]", "", first.lower())
#     if token in ABBR_LOWER:
#         return ABBR_LOWER[token]
#     raw_strip = first.strip()
#     if raw_strip in ABBR2FULL:
#         return raw_strip
#     join = raw_strip.replace(" ", "")
#     if join.lower() in ABBR_LOWER:
#         return ABBR_LOWER[join.lower()]
#     # 再尝试在整句里直接正则找一个缩写（宽松兼容空格差异）
#     for abbr in ABBR_SET:
#         pat = r'\b' + re.sub(r'\s+', r'\\s*', re.escape(abbr)) + r'\b'
#         if re.search(pat, first, flags=re.I):
#             return abbr
#     return "<UNK>"
# [R13/R16 修订]
def simple_extract(raw: str) -> str:
    return parse_label(raw, LABEL_ABBR)["label"]

# [R13/R33 原代码保留]
# def extract_reason(raw: str) -> str:
#     """尽量从模型输出中提取短原因；优先匹配 'Reason:' 行；否则取第二行。"""
#     if not raw:
#         return ""
#     # 优先找 Reason:
#     m = re.search(r'(?:^|\n)\s*Reason\s*:\s*(.+)', raw, flags=re.I | re.DOTALL)
#     if m:
#         reason = m.group(1).strip()
#         # 只取到下一行结束，避免把后续内容带上
#         reason = reason.splitlines()[0].strip()
#         return reason
#     # 否则取第二行（去空行）
#     lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
#     if len(lines) >= 2:
#         # 第二行若仍带前缀，清一下
#         second = re.sub(r'^\s*Reason\s*:\s*', '', lines[1], flags=re.I).strip()
#         return second
#     return ""
# [R13/R33 修订]
def extract_reason(raw: str) -> str:
    if not isinstance(raw, str):
        return ""
    view = visible_answer(raw)
    match = re.search(r"(?mis)^\s*(?:#Reason|Reason)\s*[:：]\s*(.+)", view)
    return match.group(1).strip() if match else ""

# -------- 5. 加载模型（不改参数） --------
print("[INFO] Loading model …")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# -------- 6. classify（不改采样参数） --------
# [R13/R16/R23/R24/R29/R31/R33 原代码保留]
# @torch.inference_mode()
# def classify(comment: str):
#     messages = [
#         {"role": "user",   "content": build_prompt(comment)},
#     ]
#     text = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True,
#         enable_thinking=False
#     )
#     inputs = tokenizer(text, return_tensors="pt",
#                        truncation=True, max_length=4096).to(model.device)
#
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=MAX_NEW_TOKENS,
#         temperature=0.7,
#         top_p=0.8,
#         top_k=20,
#     )
#     raw = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
#                            skip_special_tokens=True).strip()
#
#     # —— 解析缩写（优先从“Abbreviation:”行抓）——
#     # 先找带 key 的形式
#     m = re.search(r'(?:^|\n)\s*Abbreviation\s*:\s*([^\n\r]+)', raw, flags=re.I)
#     if m:
#         abbr_candidate_line = m.group(1).strip()
#     else:
#         # 回退：第一行
#         abbr_candidate_line = re.split(r"[\r\n]", raw, maxsplit=1)[0].strip()
#
#     abbr = simple_extract(abbr_candidate_line)
#     reason = extract_reason(raw)
#
#     return abbr, raw, reason
# [R13/R16/R23/R24/R29/R31/R33 修订]
def classify(comment: str):
    messages = [{'role': 'user', 'content': build_prompt(comment)}]
    event = audited_generate(model, tokenizer, messages,
                             {'max_new_tokens': MAX_NEW_TOKENS, 'temperature': 0.7, 'top_p': 0.8, 'top_k': 20, 'do_sample': True, 'min_p': 0.0}, audit=_REVISION_AUDIT,
                             max_input_tokens=4096, native_thinking=False,
                             reasoning_effort=None, stage="classification")
    parsed = parse_classification(event["answer_text"], LABEL_ABBR, with_reason=True)
    if event["parse_status"] != "ok":
        parsed.update(label="<UNK>", parse_ok=False, status=event["parse_status"])
    return {**parsed, "raw_text": event["raw_text"], "generation_parse_status": event["parse_status"]}

# -------- 7. 主流程 --------
# [R22/R31/R33 原代码保留]
# df = pd.read_csv(CSV_PATH)
# rows, unk_cnt = [], 0
#
# print("[INFO] Classifying …")
# for cmt, gold in tqdm(zip(df["Comment Text"], df["ML TD Type"]),
#                       total=len(df), ncols=80):
#     abbr, raw, reason = classify(cmt)
#     if abbr == "<UNK>":
#         unk_cnt += 1
#         if PRINT_UNPARSED:
#             with open(RUN_DIR/"unparsed.txt", "a", encoding="utf-8") as f:
#                 f.write(raw + "\n")
#         abbr = FALLBACK_ABBR
#         # 若缩写未知，reason 也给个空或沿用原文
#         reason = reason or ""
#
#     rows.append({
#         "comment": cmt,
#         "real": ABBR2FULL.get(gold, gold),
#         "pred": ABBR2FULL[abbr],   # 保存“全称”作为最终预测
#         "reason": reason,          # 新增：简短理由
#         "raw_output": raw          # 保留完整原始输出以便排查
#     })
#
# pred_df = pd.DataFrame(rows)
#
# # 终端打印（保持原有的文本版）
# report_text = classification_report(pred_df["real"], pred_df["pred"],
#                                     labels=LABELS, zero_division=0, digits=3)
# print("\n[INFO] Overall metrics:\n", report_text)
#
# # -------- 8. 保存结果 --------
# # 8.1 逐样本预测（新增 reason 列）
# pred_df.to_csv(RUN_DIR/"predictions.csv", index=False)
#
# # 8.2 混淆矩阵
# pd.DataFrame(
#     confusion_matrix(pred_df["real"], pred_df["pred"], labels=LABELS),
#     index=LABELS, columns=LABELS
# ).to_csv(RUN_DIR/"confusion_matrix.csv")
#
# # 8.3 将 classification_report 保存为 CSV（含 Unrecognized 信息）
# report_dict = classification_report(pred_df["real"], pred_df["pred"],
#                                     labels=LABELS, zero_division=0, output_dict=True)
# rep_df = pd.DataFrame(report_dict).transpose()
# rep_df.index.name = "label"
# rep_df = rep_df.reset_index()
#
# rep_df["info"] = ""
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
# # 8.4 保存设置
# (Path(RUN_DIR/"settings.json")).write_text(json.dumps({
#     "csv": CSV_PATH,
#     "model": MODEL_PATH,
#     "run_dir": str(RUN_DIR),
#     "max_new_tokens": MAX_NEW_TOKENS,
#     "fallback_abbr": FALLBACK_ABBR,
#     "with_reason": True
# }, indent=2))
#
# print(f"\n[INFO] Files saved in {RUN_DIR}")
# print(f"[INFO] Unrecognized mapped to '{FALLBACK_ABBR}': {unk_cnt}")
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
(RUN_DIR/"settings.json").write_text(json.dumps({
    **_REVISION_AUDIT.settings(),
    "model": MODEL_PATH, "csv": CSV_PATH, "run_dir": str(RUN_DIR),
    "max_new_tokens": MAX_NEW_TOKENS,
    "reasoning_mode": 'nonthinking',
    "fallback_abbr": FALLBACK_ABBR,
    "metrics_policy": "Primary: existing final fallback predictions. Additional failure_as_error metrics mark invalid final parses wrong.",
    "input_policy": "Full templated input, including demonstrations/comment/code context; actual context with 64 safety tokens; requested generation cap is reduced to remaining context and the effective cap is recorded; no input truncation.",
    "parser_policy": "Unique complete canonical abbreviation/full-name or the explicitly requested answer field; never search reasoning.",
    "unknown_count_before_fallback": unk_cnt,
}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[INFO] Revised results saved to {RUN_DIR}")
