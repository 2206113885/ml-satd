#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_satd_llama31_zero_shot.py
-----------------------------
零样本 · Meta-Llama-3.1-8B-Instruct · 官方 pipeline 对话模板
输出目录: results/run-YYYYmmdd-HHMMSS-模型名
保存: predictions.csv, confusion_matrix.csv, report.csv, settings.json
"""
# [R13/R16/R17/R22/R23/R24/R29/R31/R33 修订] Shared explicit audit runtime.
import sys as _revision_sys
from pathlib import Path as _RevisionPath
_revision_sys.path.insert(0, str(_RevisionPath(__file__).resolve().parents[1]))
from revision_runtime import RunAudit, audited_generate, parse_label, parse_classification, visible_answer, write_failure_metrics

import re, os, gc, json, torch, transformers
from pathlib import Path
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

# ---------- 0. 路径 ----------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # 如需改 GPU，请改这里
CSV_PATH   = "/data/yangzhifei/pro/ML_SATD/dataset/sampled_label_dataset_normalized_rmvtodo.csv"
MODEL_PATH = "/data/yangzhifei/pro/ML_SATD/models/Llama-3.1-8B-Instruct"

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
FALLBACK_ABBR   = "Data Conf"   # 解析失败回退缩写
MAX_NEW_TOKENS  = 20            # 只需放下一个缩写

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
ABBR_LOWER = {a.lower(): a for a in ABBR_SET}
ABBR_REGEX = re.compile(r"\b(" + "|".join(re.escape(a) for a in ABBR_SET) + r")\b", re.I)

# ---------- 2. Prompt 头（描述，无 few-shot） ----------
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

# [R27 原代码保留]
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
# [R27 修订]
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
   • Understanding & interpretability: model-code readability/comprehension or understanding its implementation ⇒ {{MCC}}; lack of ML knowledge ⇒ {{MLKnow}}; model black-box / explainability concerns ⇒ {{MLInter}}.
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
        "Code Comment:\n\"\"\"\n"
        f"{comment}\n\"\"\"\n\nAbbreviation:"
    )

# ---------- 3. 稳健解析 ----------
# [R13/R16 原代码保留]
# def extract_abbr(raw: str) -> str:
#     first = re.split(r"[\r\n]", raw.strip(), maxsplit=1)[0].strip()
#     first = re.sub(r"(^[*_`]+|[*_`]+$)", "", first)  # 去装饰
#     m = ABBR_REGEX.search(first)
#     if m:
#         return ABBR_LOWER[m.group(1).lower()]
#     token = re.sub(r"[^a-z]", "", first.lower())
#     if token in ABBR_LOWER:
#         return ABBR_LOWER[token]
#     # [R03 原代码保留]
#     # for full, ab in LABEL_ABBR.items():
#     #     if full.split()[0].lower() in token:
#     #         return ab
#     # [R03 修订]
#     full_name_matches = [
#         ab for full, ab in LABEL_ABBR.items()
#         if re.sub(r"[^a-z]", "", full.lower()) == token
#     ]
#     if len(full_name_matches) == 1:
#         return full_name_matches[0]
#     return "<UNK>"
# [R13/R16 修订]
def extract_abbr(raw: str) -> str:
    return parse_label(raw, LABEL_ABBR)["label"]

print("[INFO] Loading model …")




# 4) 用我们设好的 model/tokenizer 构建官方 pipeline
pipeline = transformers.pipeline(
    "text-generation",
    model=MODEL_PATH,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)



# [R22/R24/R31 原代码保留]
# def text_gen(messages, MAX_NEW_TOKENS):
#     return pipeline(
#             messages,
#             max_new_tokens=MAX_NEW_TOKENS,
#             pad_token_id = pipeline.tokenizer.eos_token_id
#             )
# [R22/R24/R31 修订]
def text_gen(messages, MAX_NEW_TOKENS):
    event = audited_generate(pipeline.model, pipeline.tokenizer, messages,
                             {"generation_config": getattr(pipeline, "generation_config", pipeline.model.generation_config), "max_new_tokens": MAX_NEW_TOKENS,
                              "pad_token_id": pipeline.tokenizer.eos_token_id},
                             audit=_REVISION_AUDIT, stage="text_gen")
    return [{"generated_text": [{"role": "assistant", "content": event["answer_text"]}]}]


# ---------- 5. classify（按官方 messages 走 pipeline） ----------
# [R13/R16/R23/R24/R29/R31/R33 原代码保留]
# @torch.inference_mode()
# def classify(comment: str):
#     messages = [
#         {"role": "user",   "content": build_prompt(comment)},
#     ]
#     out = text_gen(messages, MAX_NEW_TOKENS)
#     # out[0]["generated_text"] 是对话消息列表，最后一个是 assistant 的回复
#     assistant_msg = out[0]["generated_text"][-1]
#     raw = assistant_msg.get("content", "").strip()
#     raw_first = raw.split("\n", 1)[0].strip()
#     print(raw_first)
#     return extract_abbr(raw_first), raw_first
# [R13/R16/R23/R24/R29/R31/R33 修订]
def classify(comment: str):
    messages = [{'role': 'user', 'content': build_prompt(comment)}]
    event = audited_generate(pipeline.model, pipeline.tokenizer, messages,
                             {"generation_config": getattr(pipeline, "generation_config", pipeline.model.generation_config), 'max_new_tokens': MAX_NEW_TOKENS, 'pad_token_id': pipeline.tokenizer.eos_token_id}, audit=_REVISION_AUDIT,
                             max_input_tokens=None, native_thinking=False,
                             reasoning_effort=None, stage="classification")
    parsed = parse_classification(event["answer_text"], LABEL_ABBR, with_reason=False)
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
#     abbr, raw_first = classify(cmt)
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
#         "raw_output": raw_first
#     })
#
# pred_df = pd.DataFrame(rows)
#
# # 终端打印
# report_text = classification_report(pred_df["real"], pred_df["pred"],
#                                     labels=LABELS, zero_division=0, digits=3)
# print("\n[INFO] Overall metrics:\n", report_text)
#
# # ---------- 7. 保存结果 ----------
# # 7.1 逐样本预测
# pred_df.to_csv(RUN_DIR/"predictions.csv", index=False)
#
# # 7.2 混淆矩阵（行真列预测）
# pd.DataFrame(
#     confusion_matrix(pred_df["real"], pred_df["pred"], labels=LABELS),
#     index=LABELS, columns=LABELS
# ).to_csv(RUN_DIR/"confusion_matrix.csv")
#
# # 7.3 classification_report 保存为 CSV，并追加 INFO 行
# report_dict = classification_report(pred_df["real"], pred_df["pred"],
#                                     labels=LABELS, zero_division=0, output_dict=True)
# rep_df = pd.DataFrame(report_dict).transpose().reset_index().rename(columns={"index":"label"})
# rep_df["info"] = ""
# info_line = f"Unrecognized mapped to '{FALLBACK_ABBR}': {unk_cnt}/{len(df)}"
# rep_df.loc[len(rep_df)] = {"label":"[INFO]", "precision":"", "recall":"", "f1-score":"", "support":"", "info":info_line}
# rep_df.to_csv(RUN_DIR/"report.csv", index=False)
#
# # 7.4 保存设置
# (Path(RUN_DIR/"settings.json")).write_text(json.dumps({
#     "csv": CSV_PATH,
#     "model": MODEL_PATH,
#     "model_name": MODEL_NAME,
#     "run_dir": str(RUN_DIR),
#     "max_new_tokens": MAX_NEW_TOKENS,
#     "fallback_abbr": FALLBACK_ABBR
# }, indent=2))
#
# print(f"\n[INFO] Files saved in {RUN_DIR}")
# print(f"[INFO] Unrecognized mapped to '{FALLBACK_ABBR}': {unk_cnt}/{len(df)}")
#
# gc.collect(); torch.cuda.empty_cache()
#
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
    "reasoning_mode": 'prompt_cot',
    "fallback_abbr": FALLBACK_ABBR,
    "metrics_policy": "Primary: existing final fallback predictions. Additional failure_as_error metrics mark invalid final parses wrong.",
    "input_policy": "Full templated input, including demonstrations/comment/code context; actual context with 64 safety tokens; requested generation cap is reduced to remaining context and the effective cap is recorded; no input truncation.",
    "parser_policy": "Unique complete canonical abbreviation/full-name or the explicitly requested answer field; never search reasoning.",
    "unknown_count_before_fallback": unk_cnt,
}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[INFO] Revised results saved to {RUN_DIR}")
