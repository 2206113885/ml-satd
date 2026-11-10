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

# ===================== 0) 基础路径（仅此处改数据文件路径；推理参数一概不改） =====================
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
CSV_PATH        = "/data/yangzhifei/pro/ML_SATD/dataset/sampled_label_dataset_normalized_rmvtodo.csv"
MODEL_PATH      = "/data/yangzhifei/pro/ML_SATD/models/gpt-oss-20b"   # 或 "openai/gpt-oss-20b"

# 这里放“两个不同模型”的 predictions.csv（来自你之前各自运行）
PRED1_PATH = "/data/yangzhifei/pro/ML_SATD/basic_zero_shot_CoT_reason/results/run-20250908-170030-gpt-oss-20b-36.6/predictions.csv"  # ← 改成你的文件（任意模型1）
PRED2_PATH = "/data/yangzhifei/pro/ML_SATD/basic_zero_shot_CoT_reason/results/run-20250909-192139-Qwen3-14B-35.9/predictions.csv"  # ← 改成你的文件（任意模型2）

MODEL_NAME = Path(MODEL_PATH).name
# [R09-R44 原代码保留]
# RUN_DIR    = Path("results") / f"run-{datetime.now():%Y%m%d-%H%M%S}-{MODEL_NAME}-arbiter"
# [R09-R44 修订]
import sys
import uuid
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from revision_runtime import RunAudit, audited_generate, parse_label, visible_answer, row_identity, write_failure_metrics
REVISION_SEED = int(os.environ.get('SATD_SEED', '42'))
REASONING_EFFORT = os.environ.get('SATD_REASONING_EFFORT', 'medium')
if REASONING_EFFORT not in {'low', 'medium', 'high'}:
    raise ValueError('SATD_REASONING_EFFORT must be low, medium or high')
# [R31 修订] 将新产物固定在修订目录，避免依赖启动工作目录。
# RUN_DIR = Path('results') / f'run-{datetime.now():%Y%m%d-%H%M%S}-{MODEL_NAME}-arbiter'
RUN_DIR = Path(__file__).resolve().parents[1] / "revised_outputs" / f'run-{datetime.now():%Y%m%d-%H%M%S}-{MODEL_NAME}-arbiter'
RUN_DIR = RUN_DIR.with_name(RUN_DIR.name + '-revised-' + datetime.now().strftime('%f') + '-' + uuid.uuid4().hex[:8])
# [R09-R44 原代码保留]
# RUN_DIR.mkdir(parents=True, exist_ok=True)
# [R09-R44 修订]
RUN_DIR.mkdir(parents=True, exist_ok=False)
audit = RunAudit(RUN_DIR, MODEL_PATH, CSV_PATH, seed=REVISION_SEED)

# ===================== 1) 其它参数（保持不变） =====================
PRINT_UNPARSED    = True
FALLBACK_ABBR     = "Data Conf"
MAX_NEW_TOKENS    = 32768
MAX_INPUT_TOKENS  = 4096   

# ===================== 2) 标签映射与工具 =====================
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

# [R09-R44 修订] Complete GPT-OSS final generation and native effort are audited centrally.
def _revision_generate(messages, stage):
    # [R31 原代码保留]
    # event = audited_generate(
    #     pipe.model, pipe.tokenizer, messages, {'max_new_tokens': MAX_NEW_TOKENS},
    #     audit=audit, max_input_tokens=MAX_INPUT_TOKENS, native_thinking=False,
    #     reasoning_effort=REASONING_EFFORT, stage=stage,
    # )
    # [R31 修订] Preserve pipeline task defaults as the explicit base configuration.
    event = audited_generate(
        pipe.model, pipe.tokenizer, messages,
        {
            'generation_config': getattr(pipe, 'generation_config', pipe.model.generation_config),
            'max_new_tokens': MAX_NEW_TOKENS,
        },
        audit=audit,
        max_input_tokens=MAX_INPUT_TOKENS,
        native_thinking=False,
        reasoning_effort=REASONING_EFFORT,
        stage=stage,
    )
    return event.get('answer_text', '')
def _canon(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())

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

# 23类定义（给裁判模型判定参考用）
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

ALLOWED_ABBRS_LINE = ", ".join(sorted(ABBR_SET))

# ===================== 3) 构造“裁判”提示（允许从两者中选其一，或另选 23 类之一） =====================
def build_referee_prompt(comment: str,
                         pred1_full: str, pred2_full: str,
                         pred1_reason: str = None, pred2_reason: str = None) -> str:
    """
    - 仍提供 23 类定义作为依据。
    - 给出两个候选（全称与缩写）与它们各自的一句话 reason（若有）。
    - 让 gpt-oss-20b：优先二选一；若都不合适，可改从 23 类中自选。
    - 强制输出格式：第一行缩写、第二行 Reason: ...（≤256 words）
    """
    p1_abbr = FULL2ABBR.get(pred1_full, FALLBACK_ABBR)
    p2_abbr = FULL2ABBR.get(pred2_full, FALLBACK_ABBR)

    parts = [
        PROMPT_HEAD,
        "Please determine which type of ML Self-Admitted Technical Debt (STAD) the following Code Comment belongs to.",
        "Two models have predicted SATD categories for the code comment below.",
        # "You are an arbiter. Two models have predicted SATD categories for the code comment below.",
        # "Decide the most appropriate label:",
        " • If one of the two candidates matches the definition best, choose it.",
        " • If both are inappropriate, choose a BETTER label from the full set of allowed abbreviations.",
        "",
        "Candidates:",
        f" - Model-1: {p1_abbr} → {pred1_full}" + (f" | reason: {pred1_reason}" if pred1_reason else ""),
        f" - Model-2: {p2_abbr} → {pred2_full}" + (f" | reason: {pred2_reason}" if pred2_reason else ""),
        "",
        "Output format (STRICT):",
        "Line 1: Reason: <a brief justification in <= 256 words>",
        "Line 2: <ABBREVIATION>   (must be one of: " + ALLOWED_ABBRS_LINE + ")",
        "",
        "Code Comment:",
        f"```text\n{comment}\n```",
        "",
        "#Final:"
    ]
    return "\n".join(parts)

# ===================== 4) 解析工具：从 gpt-oss 输出抓取“最终缩写 + reason” =====================
ABBR_CANON_MAP = {_canon(a): a for a in ABBR_SET}
ABBR_REGEX = re.compile(r"\b(" + "|".join(re.escape(a) for a in sorted(ABBR_SET, key=len, reverse=True)) + r")\b", re.I)

# [R09-R44 原代码保留]
# def extract_abbr(text: str) -> str:
#     if not text:
#         return "<UNK>"
#     # 1) 直接匹配缩写（优先精确）
#     m = ABBR_REGEX.search(text)
#     if m:
#         hit = m.group(1)
#         # 归一化回标准大小写/空格
#         for a in ABBR_SET:
#             if _canon(a) == _canon(hit):
#                 return a
#     # 2) 纯字母规范后命中
#     cz = _canon(text)
#     if cz in ABBR_CANON_MAP:
#         return ABBR_CANON_MAP[cz]
#     # 3) 如果写了全称，也容错一下
#     for full, ab in FULL2ABBR.items():
#         if re.search(r'\b' + re.escape(full) + r'\b', text, flags=re.I):
#             return ab
#     return "<UNK>"
# [R09-R44 修订]
def extract_abbr(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()] if isinstance(text, str) else []
    if not lines or len(lines) > 2:
        return '<UNK>'
    if len(lines) == 2 and not re.fullmatch(r'Reason\s*[:：]\s*.+', lines[1], flags=re.I):
        return '<UNK>'
    return parse_label(lines[0], LABEL_ABBR, native_thinking=False)['label']

def extract_reason(text: str) -> str:
    """
    规则：
    - 取 'assistantfinal' 之后的内容中，优先找以 'Reason:' 开头的行
    - 否则尝试第一行之外的剩余文本
    - 长度做一次适度截断（安全）
    """
    if not text:
        return ""
    body = text.split('assistantfinal')[-1]
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    if not lines:
        return ""
    # 搜索 Reason: 行
    for ln in lines:
        if re.match(r'^\s*reason\s*[:\-]\s*(.+)$', ln, flags=re.I):
            return re.sub(r'^\s*reason\s*[:\-]\s*', '', ln, flags=re.I).strip()
    # 否则：把第一行视作缩写行，其余拼接为 reason
    if len(lines) >= 2:
        return " ".join(lines[1:]).strip()
    return ""

# ===================== 5) 读取两侧 predictions.csv（支持不同列名；若缺 reason 也兼容） =====================
# [R09-R44 原代码保留]
# def read_preds(path):
#     """
#     返回 dict: { comment(str) -> (pred_full(str), reason(str or None)) }
#     兼容列名：
#       - comment（必需）
#       - pred（全称）或 fine_abbr/pred_abbr（缩写）
#       - reason（可选）
#     """
#     out = {}
#     p = Path(path)
#     if not p.exists():
#         return out
#     df = pd.read_csv(p)
#     if "comment" not in df.columns:
#         return out
# 
#     for _, r in df.iterrows():
#         cmt = str(r["comment"])
#         # 取预测全称
#         if "pred" in df.columns and isinstance(r["pred"], str):
#             full = r["pred"].strip()
#         elif "fine_abbr" in df.columns and isinstance(r["fine_abbr"], str):
#             full = ABBR2FULL.get(r["fine_abbr"].strip(), None)
#         elif "pred_abbr" in df.columns and isinstance(r["pred_abbr"], str):
#             full = ABBR2FULL.get(r["pred_abbr"].strip(), None)
#         else:
#             full = None
# 
#         if full is None:
#             continue
# 
#         # 取 reason（可没有）
#         rsn = None
#         if "reason" in df.columns and isinstance(r["reason"], str):
#             rsn = r["reason"].strip()
#         out[cmt] = (full, rsn)
#     return out
# [R09-R44 修订]
def read_preds(path):
    """Return source-row keyed predictions; never use comment text as a unique key."""
    prediction_path = Path(path)
    if not prediction_path.exists():
        raise FileNotFoundError(f'Missing arbitration input: {prediction_path}')
    source = pd.read_csv(CSV_PATH)
    predictions = pd.read_csv(prediction_path)
    if 'comment' not in predictions or len(predictions) != len(source):
        raise ValueError(f'Prediction coverage mismatch: {prediction_path}')
    if 'source_row' in predictions:
        numeric_rows = pd.to_numeric(predictions['source_row'], errors='raise')
        if numeric_rows.isna().any() or (numeric_rows % 1 != 0).any():
            raise ValueError('source_row must contain integer positions')
        source_rows = numeric_rows.astype(int)
        if source_rows.duplicated().any() or set(source_rows) != set(range(len(source))):
            raise ValueError('source_row must have exactly one prediction per source row')
        predictions = predictions.assign(source_row=source_rows).set_index('source_row').loc[range(len(source))]
        alignment = 'source_row_with_identity_validation'
    else:
        predictions = predictions.reset_index(drop=True)
        alignment = 'legacy_order_verified_comment_and_available_gold'
    out = {}
    for index, source_row in source.iterrows():
        predicted = predictions.iloc[index]
        if str(predicted['comment']) != str(source_row['Comment Text']):
            raise ValueError(f'Comment mismatch at source row {index} in {prediction_path}')
        identity = row_identity(source_row.to_dict(), index)
        for id_column in ('dataset_id', 'Dataset ID'):
            if id_column in predictions and str(predicted[id_column]) != str(identity['dataset_id']):
                raise ValueError(f'Dataset ID mismatch at source row {index}')
        for gold_column in ('real', 'gold_original', 'original_gold'):
            if gold_column in predictions:
                source_gold = parse_label(str(source_row['ML TD Type']), LABEL_ABBR, native_thinking=False)
                saved_gold = parse_label(str(predicted[gold_column]), LABEL_ABBR, native_thinking=False)
                if not source_gold['parse_ok'] or not saved_gold['parse_ok'] or source_gold['label'] != saved_gold['label']:
                    raise ValueError(f'Gold mismatch at source row {index}')
        value = next((predicted[column] for column in ('pred', 'fine_abbr', 'pred_abbr') if column in predictions), '')
        parsed = parse_label(str(value), LABEL_ABBR, native_thinking=False)
        full = ABBR2FULL[parsed['label']] if parsed['parse_ok'] else None
        detail = predicted.get('reason', None)
        detail = str(detail) if pd.notna(detail) else None
        out[int(index)] = (full, detail)
    # This is a newly verified mapping, not reconstructed historical run provenance.
    mapping = {'prediction_file': str(prediction_path), 'alignment': alignment, 'rows': len(source), 'uses_comment_as_key': False}
    with (RUN_DIR / 'prediction_alignment.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(mapping, ensure_ascii=False) + '\n')
    return out

pred1_map = read_preds(PRED1_PATH)
pred2_map = read_preds(PRED2_PATH)

# ===================== 6) 加载 gpt-oss-20b（保持原样，不改参数） =====================
print("[INFO] Loading model …")
pipe = pipeline(
    "text-generation",
    model=MODEL_PATH,
    torch_dtype="auto",
    device_map="auto",
)
print("Device set to use cuda:0")

# ===================== 7) 一次裁判调用（必要时补打一轮“reason-only”） =====================
# [R09-R44 原代码保留]
# @torch.inference_mode()
# # [R09-R44 原代码保留]
# # [R09-R44 修订]
# def _revision_generate(messages, stage):
#     event = audited_generate(
#         pipe.model, pipe.tokenizer, messages, {'max_new_tokens': MAX_NEW_TOKENS},
#         audit=audit, max_input_tokens=MAX_INPUT_TOKENS, native_thinking=False,
#         reasoning_effort=REASONING_EFFORT, stage=stage,
#     )
#     return event.get('answer_text', '')
# def arbitrate_once(comment: str,
#                    pred1_full: str, pred2_full: str,
#                    pred1_reason: str = None, pred2_reason: str = None):
#     """
#     第一次调用：要求返回两行（缩写 + Reason: ...）。
#     若 reason 缺失，则补打一轮“只给简短 reason”的提示（不改任何推理参数）。
#     """
#     # "role": "system", "content": "You are a strict classifier arbiter. Reasoning: low. "
#     # {"role": "system", "content":
#     #      "You are a strict classifier arbiter. Reasoning: low. "
#     #      "Output MUST be exactly two lines: line1 abbreviation, line2 'Reason: ...'."}
#     messages = [
#         {"role": "user", "content": build_referee_prompt(
#             comment, pred1_full, pred2_full, pred1_reason, pred2_reason
#         )},
#     ]
# 
#     outputs = pipe(messages, max_new_tokens=MAX_NEW_TOKENS)
#     answer  = outputs[0]["generated_text"][-1]['content']
#     final   = answer.split('assistantfinal')[-1]
# 
#     abbr = extract_abbr(final)
#     print(abbr)
#     reason = extract_reason(final)
# 
#     # 若缩写不在 23 类，先置为 <UNK>，稍后统一处理
#     if abbr not in ABBR_SET:
#         abbr = "<UNK>"
# 
#     # 若 reason 缺失，补打一轮只要 reason（不改参数）
#     if not reason:
#         reason = refill_reason(comment, abbr, pred1_full, pred2_full,
#                                pred1_reason, pred2_reason)
# [R09-R44 修订]
def arbitrate_once(comment: str,
                   pred1_full: str, pred2_full: str,
                   pred1_reason: str = None, pred2_reason: str = None):
    """
    第一次调用：要求返回两行（缩写 + Reason: ...）。
    若 reason 缺失，则补打一轮“只给简短 reason”的提示（不改任何推理参数）。
    """
    # "role": "system", "content": "You are a strict classifier arbiter. Reasoning: low. "
    # {"role": "system", "content":
    #      "You are a strict classifier arbiter. Reasoning: low. "
    #      "Output MUST be exactly two lines: line1 abbreviation, line2 'Reason: ...'."}
    messages = [
        {"role": "user", "content": build_referee_prompt(
            comment, pred1_full, pred2_full, pred1_reason, pred2_reason
        )},
    ]

    answer = _revision_generate(messages, 'arbitration')
    final = answer

    abbr = extract_abbr(final)
    print(abbr)
    reason = extract_reason(final)

    # 若缩写不在 23 类，先置为 <UNK>，稍后统一处理
    if abbr not in ABBR_SET:
        abbr = "<UNK>"

    # 若 reason 缺失，补打一轮只要 reason（不改参数）
    if not reason:
        reason = refill_reason(comment, abbr, pred1_full, pred2_full,
                               pred1_reason, pred2_reason)

    return abbr, reason, final
    # [R09-R44 原代码保留] 重复的旧末行不再执行。
    # return abbr, reason, final

# [R09-R44 原代码保留]
# @torch.inference_mode()
# def refill_reason(comment: str, abbr: str,
#                   pred1_full: str, pred2_full: str,
#                   pred1_reason: str = None, pred2_reason: str = None) -> str:
#     """
#     第二次调用：仅要求一条 ≤30 词的 reason。若 abbr 不合法，就让模型先判断
#     更合理的缩写并给出理由（但我们仍只拿 reason 字段）。
#     """
#     allowed = ", ".join(sorted(ABBR_SET))
#     prompt = [
#         PROMPT_HEAD,
#         "You previously omitted the justification. Provide ONLY a brief reason (<= 256 words) for the final label.",
#         f"Previously chosen (may be <UNK>): {abbr}",
#         f"Candidates from models: {FULL2ABBR.get(pred1_full, 'N/A')} ({pred1_full}); "
#         f"{FULL2ABBR.get(pred2_full, 'N/A')} ({pred2_full})",
#         "You may keep the previous choice or switch to a better one based on the definitions, "
#         "but DO NOT output the label here—return ONLY the reason line.",
#         "",
#         "Output format (STRICT):",
#         "Reason: <brief justification>",
#         "",
#         "Code Comment:",
#         f"```text\n{comment}\n```",
#         "",
#         "Allowed abbreviations: " + allowed,
#         "",
#         "#Final:"
#     ]
#     messages = [
#         {"role": "system", "content":
#          "Return ONLY one line starting with 'Reason:'. No extra text."},
#         {"role": "user", "content": "\n".join(prompt)},
#     ]
#     outputs = pipe(messages, max_new_tokens=MAX_NEW_TOKENS)
#     answer  = outputs[0]["generated_text"][-1]['content']
#     final   = answer.split('assistantfinal')[-1]
#     rsn     = extract_reason(final)
#     return rsn if rsn else "Reason: concise justification unavailable."
# [R09-R44 修订]
def refill_reason(comment: str, abbr: str,
                  pred1_full: str, pred2_full: str,
                  pred1_reason: str = None, pred2_reason: str = None) -> str:
    """
    第二次调用：仅要求一条 ≤30 词的 reason。若 abbr 不合法，就让模型先判断
    更合理的缩写并给出理由（但我们仍只拿 reason 字段）。
    """
    allowed = ", ".join(sorted(ABBR_SET))
    prompt = [
        PROMPT_HEAD,
        "You previously omitted the justification. Provide ONLY a brief reason (<= 256 words) for the final label.",
        f"Previously chosen (may be <UNK>): {abbr}",
        f"Candidates from models: {FULL2ABBR.get(pred1_full, 'N/A')} ({pred1_full}); "
        f"{FULL2ABBR.get(pred2_full, 'N/A')} ({pred2_full})",
        "You may keep the previous choice or switch to a better one based on the definitions, "
        "but DO NOT output the label here—return ONLY the reason line.",
        "",
        "Output format (STRICT):",
        "Reason: <brief justification>",
        "",
        "Code Comment:",
        f"```text\n{comment}\n```",
        "",
        "Allowed abbreviations: " + allowed,
        "",
        "#Final:"
    ]
    messages = [
        {"role": "system", "content":
         "Return ONLY one line starting with 'Reason:'. No extra text."},
        {"role": "user", "content": "\n".join(prompt)},
    ]
    answer = _revision_generate(messages, 'posthoc_reason')
    final = answer
    rsn     = extract_reason(final)
    return rsn

# ===================== 8) 主流程：对每条样本进行二次裁判 =====================
df = pd.read_csv(CSV_PATH)

rows = []
unk_cnt = 0
gold_unmapped = 0
missing_side = 0

print("[INFO] Arbitrating between two model predictions …")
# [R09-R44 原代码保留]
# for cmt, gold in tqdm(zip(df["Comment Text"], df["ML TD Type"]),
#                       total=len(df), ncols=80):
#     real_full = normalize_to_full(str(gold))
#     if real_full is None:
#         gold_unmapped += 1
#         continue
# 
#     p1_full, p1_reason = pred1_map.get(str(cmt), (None, None))
#     p2_full, p2_reason = pred2_map.get(str(cmt), (None, None))
# 
#     # 如果任意一侧缺失：当作“缺少一方的候选”，仍交由裁判模型裁断（允许从 23 类自选）
#     if p1_full is None and p2_full is None:
#         # 都缺：我们提供一个“空候选”占位，让裁判直接从 23 类自选
#         p1_full = ABBR2FULL[FALLBACK_ABBR]
#         p2_full = ABBR2FULL[FALLBACK_ABBR]
#         p1_reason = "(missing)"
#         p2_reason = "(missing)"
#         missing_side += 1
#     elif p1_full is None:
#         p1_full = ABBR2FULL[FALLBACK_ABBR]
#         p1_reason = "(missing)"
#         missing_side += 1
#     elif p2_full is None:
#         p2_full = ABBR2FULL[FALLBACK_ABBR]
#         p2_reason = "(missing)"
#         missing_side += 1
# 
#     abbr, reason, raw_final = arbitrate_once(str(cmt), p1_full, p2_full, p1_reason, p2_reason)
# 
#     if abbr not in ABBR_SET:
#         unk_cnt += 1
#         if PRINT_UNPARSED:
#             with open(RUN_DIR/"unparsed.txt", "a", encoding="utf-8") as f:
#                 f.write(str(raw_final) + "\n")
#         abbr = FALLBACK_ABBR
# 
#     rows.append({
#         "comment": cmt,
#         "real": real_full,                            # 真实全称
#         "model1_pred": p1_full,                       # 模型1 全称
#         "model1_reason": p1_reason,                   # 模型1 简短/原始 reason（若有）
#         "model2_pred": p2_full,                       # 模型2 全称
#         "model2_reason": p2_reason,                   # 模型2 简短/原始 reason（若有）
#         "chosen_abbr": abbr,                          # 裁判输出缩写
#         "pred": ABBR2FULL.get(abbr, abbr),            # 转全称
#         "reason": reason,                             # 裁判强制生成的简短理由
#         "raw_output": raw_final                       # 裁判原始输出（便于排查）
#     })
# [R09-R44 修订]
# [R09-R44 原代码保留]
# for (source_row, (cmt, gold)) in enumerate(tqdm(zip(df['Comment Text'], df['ML TD Type']), total=len(df), ncols=80)):
#     audit.set_sample(source_row, df.iloc[source_row].to_dict())
#     real_full = normalize_to_full(str(gold))
#     if real_full is None:
#         raise ValueError(f'Unmapped gold at source row {source_row}: {gold!r}')
#     (p1_full, p1_reason) = pred1_map.get(source_row, (None, None))
#     (p2_full, p2_reason) = pred2_map.get(source_row, (None, None))
#     if p1_full is None and p2_full is None:
#         p1_full = ABBR2FULL[FALLBACK_ABBR]
#         p2_full = ABBR2FULL[FALLBACK_ABBR]
#         p1_reason = '(missing)'
#         p2_reason = '(missing)'
#         missing_side += 1
#     elif p1_full is None:
#         p1_full = ABBR2FULL[FALLBACK_ABBR]
#         p1_reason = '(missing)'
#         missing_side += 1
#     elif p2_full is None:
#         p2_full = ABBR2FULL[FALLBACK_ABBR]
#         p2_reason = '(missing)'
#         missing_side += 1
#     (abbr, reason, raw_final) = arbitrate_once(str(cmt), p1_full, p2_full, p1_reason, p2_reason)
#     raw_pred = abbr
#     parse_ok = raw_pred in ABBR_SET
#     fallback_reason = '' if parse_ok else 'arbiter_label_unparsed'
#     if abbr not in ABBR_SET:
#         unk_cnt += 1
#         if PRINT_UNPARSED:
#             with open(RUN_DIR / 'unparsed.txt', 'a', encoding='utf-8') as f:
#                 f.write(str(raw_final) + '\n')
#         abbr = FALLBACK_ABBR
#     rows.append({
#         **audit.sample_fields(),
#         'comment': cmt,
#         'real': real_full,
#         'model1_pred': p1_full,
#         'model1_reason': p1_reason,
#         'model2_pred': p2_full,
#         'model2_reason': p2_reason,
#         'chosen_abbr': abbr,
#         'pred': ABBR2FULL.get(abbr, abbr),
#         'reason': reason,
#         'raw_output': next((event.get('raw_text', '') for event in audit.events if event.get('stage') == 'arbitration'), ''),
#         'parsed_output': abbr,
#         'raw_pred': raw_pred,
#         'parse_ok': parse_ok,
#         'parse_status': 'ok' if parse_ok else 'final_label_unparsed',
#         'fallback_used': not parse_ok,
#         'fallback_reason': fallback_reason,
#         'generation_events': json.dumps(audit.events, ensure_ascii=False, default=str),
#         'reason_status': 'ok' if reason else 'empty_or_incomplete',
#     })
# [R09-R44 修订] Distinguish candidate-input failure from a valid model prediction.
for (source_row, (cmt, gold)) in enumerate(tqdm(zip(df['Comment Text'], df['ML TD Type']), total=len(df), ncols=80)):
    audit.set_sample(source_row, df.iloc[source_row].to_dict())
    real_full = normalize_to_full(str(gold))
    if real_full is None:
        raise ValueError(f'Unmapped gold at source row {source_row}: {gold!r}')
    (p1_full, p1_reason) = pred1_map.get(source_row, (None, None))
    (p2_full, p2_reason) = pred2_map.get(source_row, (None, None))
    input_side_missing = [name for (name, value) in [('model1', p1_full), ('model2', p2_full)] if value is None]
    if p1_full is None and p2_full is None:
        p1_full = ABBR2FULL[FALLBACK_ABBR]
        p2_full = ABBR2FULL[FALLBACK_ABBR]
        p1_reason = '(missing)'
        p2_reason = '(missing)'
        missing_side += 1
    elif p1_full is None:
        p1_full = ABBR2FULL[FALLBACK_ABBR]
        p1_reason = '(missing)'
        missing_side += 1
    elif p2_full is None:
        p2_full = ABBR2FULL[FALLBACK_ABBR]
        p2_reason = '(missing)'
        missing_side += 1
    (abbr, reason, raw_final) = arbitrate_once(str(cmt), p1_full, p2_full, p1_reason, p2_reason)
    raw_pred = abbr
    parse_ok = raw_pred in ABBR_SET
    fallback_reason = '' if parse_ok else 'arbiter_label_unparsed'
    if abbr not in ABBR_SET:
        unk_cnt += 1
        if PRINT_UNPARSED:
            with open(RUN_DIR / 'unparsed.txt', 'a', encoding='utf-8') as f:
                f.write(str(raw_final) + '\n')
        abbr = FALLBACK_ABBR
    rows.append({
        **audit.sample_fields(),
        'comment': cmt,
        'real': real_full,
        'model1_pred': p1_full,
        'model1_reason': p1_reason,
        'model2_pred': p2_full,
        'model2_reason': p2_reason,
        'chosen_abbr': abbr,
        'pred': ABBR2FULL.get(abbr, abbr),
        'reason': reason,
        'raw_output': next((event.get('raw_text', '') for event in audit.events if event.get('stage') == 'arbitration'), ''),
        'parsed_output': abbr,
        'raw_pred': raw_pred,
        'parse_ok': parse_ok,
        'parse_status': 'ok' if parse_ok else 'final_label_unparsed',
        'fallback_used': not parse_ok,
        'fallback_reason': fallback_reason,
        'generation_events': json.dumps(audit.events, ensure_ascii=False, default=str),
        'reason_status': 'ok' if reason else 'empty_or_incomplete',
        'input_side_missing': ';'.join(input_side_missing),
        'input_side_fallback_used': bool(input_side_missing),
        'decision_source': 'model_arbitration' if audit.events else 'existing_side_policy',
    })

pred_df = pd.DataFrame(rows)

# ===================== 9) 评估与落盘 =====================
if pred_df.empty:
    print("\n[WARN] No valid samples after label normalization.")
else:
    report_text = classification_report(
        pred_df["real"], pred_df["pred"],
        labels=LABELS, zero_division=0, digits=3
    )
    print("\n[INFO] Overall metrics (after arbitration):\n", report_text)

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

    info_lines = [
        f"Unrecognized outputs mapped to '{FALLBACK_ABBR}': {unk_cnt}",
        f"Gold labels unmapped (skipped): {gold_unmapped}",
        f"Samples with a missing model side: {missing_side}",
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

# [R09-R44 修订] Keep policy metrics and additionally count final parse failures as errors.
if not pred_df.empty:
    write_failure_metrics(pred_df, LABELS, RUN_DIR)
(RUN_DIR / 'revision_settings.json').write_text(json.dumps(audit.settings(), ensure_ascii=False, indent=2, default=str), encoding='utf-8')
