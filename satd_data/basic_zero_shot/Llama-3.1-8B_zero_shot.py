#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_satd_llama31_zero_shot.py
-----------------------------
零样本 · Meta-Llama-3.1-8B-Instruct · 官方 pipeline 对话模板
输出目录: results/run-YYYYmmdd-HHMMSS-模型名
保存: predictions.csv, confusion_matrix.csv, report.csv, settings.json
"""

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
RUN_DIR    = Path("results") / f"run-{datetime.now():%Y%m%d-%H%M%S}-{MODEL_NAME}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

PRINT_UNPARSED  = True
FALLBACK_ABBR   = "Data Conf"   # 解析失败回退缩写
MAX_NEW_TOKENS  = 256            # 只需放下一个缩写

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

# 更“收紧”的尾巴，强制只回 23 个缩写之一（不允许 None）
# PROMPT_TAIL = (
#     "Please determine which type of ML SATD the following Code Comment belongs to.\n"
#     "Return exactly ONE of these abbreviations on a single line, with no extra text:\n"
#     f"{', '.join(sorted(ABBR_SET))}"
# ).strip()
PROMPT_TAIL = f"""Please determine which type of ML Self-Admitted Technical Debt (SATD) the following Code Comment belongs to. 
You MUST answer with **only ONE** abbreviation from:
{', '.join(sorted(ABBR_SET))}
""".strip()

def build_prompt(comment: str) -> str:
    return (
        f"{PROMPT_HEAD}\n\n{PROMPT_TAIL}\n\n"
        "Code Comment:\n\"\"\"\n"
        f"{comment}\n\"\"\"\n\nAbbreviation:"
    )

# ---------- 3. 稳健解析 ----------
def extract_abbr(raw: str) -> str:
    first = re.split(r"[\r\n]", raw.strip(), maxsplit=1)[0].strip()
    first = re.sub(r"(^[*_`]+|[*_`]+$)", "", first)  # 去装饰
    m = ABBR_REGEX.search(first)
    if m:
        return ABBR_LOWER[m.group(1).lower()]
    token = re.sub(r"[^a-z]", "", first.lower())
    if token in ABBR_LOWER:
        return ABBR_LOWER[token]
    for full, ab in LABEL_ABBR.items():
        if full.split()[0].lower() in token:
            return ab
    return "<UNK>"

print("[INFO] Loading model …")




# 4) 用我们设好的 model/tokenizer 构建官方 pipeline
pipeline = transformers.pipeline(
    "text-generation",
    model=MODEL_PATH,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)



def text_gen(messages, MAX_NEW_TOKENS):
    return pipeline(
            messages,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id = pipeline.tokenizer.eos_token_id
            )


# ---------- 5. classify（按官方 messages 走 pipeline） ----------
@torch.inference_mode()
def classify(comment: str):
    messages = [
        {"role": "user",   "content": build_prompt(comment)},
    ]
    out = text_gen(messages, MAX_NEW_TOKENS)
    # out[0]["generated_text"] 是对话消息列表，最后一个是 assistant 的回复
    assistant_msg = out[0]["generated_text"][-1]
    raw = assistant_msg.get("content", "").strip()
    raw_first = raw.split("\n", 1)[0].strip()
    return extract_abbr(raw_first), raw_first

# ---------- 6. 主流程 ----------
df = pd.read_csv(CSV_PATH)
rows, unk_cnt = [], 0
print("[INFO] Classifying …")
for cmt, gold in tqdm(zip(df["Comment Text"], df["ML TD Type"]),
                      total=len(df), ncols=80):
    abbr, raw_first = classify(cmt)
    if abbr == "<UNK>":
        unk_cnt += 1
        if PRINT_UNPARSED:
            with open(RUN_DIR/"unparsed.txt", "a", encoding="utf-8") as f:
                f.write(raw_first + "\n")
        abbr = FALLBACK_ABBR
    rows.append({
        "comment": cmt,
        "real": ABBR2FULL.get(gold, gold),
        "pred": ABBR2FULL[abbr],
        "raw_output": raw_first
    })

pred_df = pd.DataFrame(rows)

# 终端打印
report_text = classification_report(pred_df["real"], pred_df["pred"],
                                    labels=LABELS, zero_division=0, digits=3)
print("\n[INFO] Overall metrics:\n", report_text)

# ---------- 7. 保存结果 ----------
# 7.1 逐样本预测
pred_df.to_csv(RUN_DIR/"predictions.csv", index=False)

# 7.2 混淆矩阵（行真列预测）
pd.DataFrame(
    confusion_matrix(pred_df["real"], pred_df["pred"], labels=LABELS),
    index=LABELS, columns=LABELS
).to_csv(RUN_DIR/"confusion_matrix.csv")

# 7.3 classification_report 保存为 CSV，并追加 INFO 行
report_dict = classification_report(pred_df["real"], pred_df["pred"],
                                    labels=LABELS, zero_division=0, output_dict=True)
rep_df = pd.DataFrame(report_dict).transpose().reset_index().rename(columns={"index":"label"})
rep_df["info"] = ""
info_line = f"Unrecognized mapped to '{FALLBACK_ABBR}': {unk_cnt}/{len(df)}"
rep_df.loc[len(rep_df)] = {"label":"[INFO]", "precision":"", "recall":"", "f1-score":"", "support":"", "info":info_line}
rep_df.to_csv(RUN_DIR/"report.csv", index=False)

# 7.4 保存设置
(Path(RUN_DIR/"settings.json")).write_text(json.dumps({
    "csv": CSV_PATH,
    "model": MODEL_PATH,
    "model_name": MODEL_NAME,
    "run_dir": str(RUN_DIR),
    "max_new_tokens": MAX_NEW_TOKENS,
    "fallback_abbr": FALLBACK_ABBR
}, indent=2))

print(f"\n[INFO] Files saved in {RUN_DIR}")
print(f"[INFO] Unrecognized mapped to '{FALLBACK_ABBR}': {unk_cnt}/{len(df)}")

gc.collect(); torch.cuda.empty_cache()

