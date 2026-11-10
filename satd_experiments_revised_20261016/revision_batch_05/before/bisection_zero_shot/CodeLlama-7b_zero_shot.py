#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型：/data/yangzhifei/pro/ML_SATD/models/CodeLlama-7b-Instruct-hf
方法：对每条评论，对 23 标签逐一做 Yes/No，汇总 Yes 候选；如有多个，再让模型在候选中只选 1 个缩写；
      若没有 Yes，则做一次全标签强制单标签输出；仍解析失败则回退到 FALLBACK_ABBR。
输出：results/run-YYYYmmdd-HHMMSS-模型名 / predictions.csv, confusion_matrix.csv, report.csv, settings.json
"""

import re, os, gc, json
from pathlib import Path
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------- 0. 路径（仅此处需改） ----------
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
CSV_PATH   = "/data/yangzhifei/pro/ML_SATD/dataset/sampled_label_dataset_normalized_rmvtodo.csv"
MODEL_PATH = "/data/yangzhifei/pro/ML_SATD/models/CodeLlama-7b-Instruct-hf"

# 目录名包含时间戳 + 模型名
MODEL_NAME = Path(MODEL_PATH).name
# [R09-R44 原代码保留]
# RUN_DIR    = Path("results") / f"run-{datetime.now():%Y%m%d-%H%M%S}-{MODEL_NAME}"
# [R09-R44 修订]
import sys
import uuid
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from revision_runtime import RunAudit, audited_generate, parse_label, visible_answer, row_identity, write_failure_metrics
REVISION_SEED = int(os.environ.get('SATD_SEED', '42'))
# [R31 修订] 将新产物固定在修订目录，避免依赖启动工作目录。
# RUN_DIR = Path('results') / f'run-{datetime.now():%Y%m%d-%H%M%S}-{MODEL_NAME}'
RUN_DIR = Path(__file__).resolve().parents[1] / "revised_outputs" / f'run-{datetime.now():%Y%m%d-%H%M%S}-{MODEL_NAME}'
RUN_DIR = RUN_DIR.with_name(RUN_DIR.name + '-revised-' + datetime.now().strftime('%f') + '-' + uuid.uuid4().hex[:8])
# [R09-R44 原代码保留]
# RUN_DIR.mkdir(parents=True, exist_ok=True)
# [R09-R44 修订]
RUN_DIR.mkdir(parents=True, exist_ok=False)
audit = RunAudit(RUN_DIR, MODEL_PATH, CSV_PATH, seed=REVISION_SEED)

PRINT_UNPARSED      = True
FALLBACK_ABBR       = "Data Conf"
MAX_YN_TOKENS       = 6      # Yes/No 极短
MAX_PICK_TOKENS     = 16     # 候选选择 1 个
MAX_FALLBACK_TOKENS = 16

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
ABBR2FULL  = {v: k for k, v in LABEL_ABBR.items()}
FULL2ABBR  = {k: v for k, v in LABEL_ABBR.items()}
LABELS     = list(LABEL_ABBR.keys())
ABBR_SET   = set(ABBR2FULL.keys())
ABBR_LOWER = {a.lower(): a for a in ABBR_SET}
ABBR_REGEX = re.compile(r"\b(" + "|".join(re.escape(a) for a in ABBR_SET) + r")\b", re.I)

# ---------- 2. 说明文本 ----------
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
GLUE CODE (GC): Supporting code written to interface with other code.
CUSTOM DATA TYPE (CDT): Using data types provided by general-purpose packages can cause extensive inter-operability burden.
MULTIPLE LANGUAGES (ML): Components written in other languages may introduce difficulties in ML development.
UNNECESSARY MODEL CODE (UMC): Model code that bottlenecks execution, is unreachable / deprecated, or is unnecessary and should be removed.
DATA PROCESSING CONFIGURATION (Data Conf): Configuring how data is processed, by editing the data directly or adding new processing steps.
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

# ---------- 3. Prompt 构造 ----------
def build_yn_prompt(comment: str, full_label: str, abbr: str) -> str:
    """是否属于某标签：回答 Yes 或 No。"""
    return (
        f"{PROMPT_HEAD}\n\n"
        "We are classifying the following code comment into one of the SATD categories.\n\n"
        "Code Comment:\n\"\"\"\n"
        f"{comment}\n\"\"\"\n\n"
        f"Question: Does this comment indicate the category \"{full_label} ({abbr})\"?\n"
        "Answer strictly with Yes or No."
    )

def build_pick_from_candidates_prompt(comment: str, candidates_abbr) -> str:
    """在候选缩写中只选 1 个缩写。"""
    cand_str = ", ".join(candidates_abbr)
    return (
        f"{PROMPT_HEAD}\n\n"
        "Pick the single best matching category for the comment below.\n"
        f"Candidates (abbreviations only): {cand_str}\n\n"
        "Comment:\n\"\"\"\n"
        f"{comment}\n\"\"\"\n\n"
        "Return exactly ONE abbreviation on a single line, with no extra text."
    )

def build_force_pick_any_prompt(comment: str) -> str:
    """从 23 个缩写中强制挑一个（兜底）。"""
    cand_str = ", ".join(sorted(ABBR_SET))
    return (
        f"{PROMPT_HEAD}\n\n"
        "Please determine which type of ML SATD the following Code Comment belongs to.\n"
        "Return exactly ONE of these abbreviations on a single line, with no extra text:\n"
        f"{cand_str}\n\n"
        "Comment:\n\"\"\"\n"
        f"{comment}\n\"\"\"\n\n"
        "Abbreviation:"
    )

# ---------- 4. 解析函数 ----------
# [R08 原代码保留]
# def extract_yes_no(raw: str) -> str:
#     """解析 Yes/No；返回 'yes'/'no'/'unk'。"""
#     first = re.split(r"[\r\n]", raw.strip(), maxsplit=1)[0].strip()
#     first = re.sub(r"(^[*_`]+|[*_`]+$)", "", first)
#     t = re.sub(r"[^a-z\u4e00-\u9fa5]", "", first.lower())
#     if t.startswith(("yes", "y")) or t.startswith("是"):
#         return "yes"
#     if t.startswith(("no", "n")) or t.startswith(("否", "不是")):
#         return "no"
#     return "unk"
# [R08 修订]
# [R09-R44 原代码保留]
# def extract_yes_no(raw: str) -> str:
#     if not isinstance(raw, str):
#         return 'unk'
#     answer = raw.strip()
#     answer = re.sub('(^[*_`]+|[*_`]+$)', '', answer).strip()
#     match = re.fullmatch('(yes|no|y|n|是|否|不是)[.!。！]?', answer, flags=re.I)
#     if match is None:
#         return 'unk'
#     token = match.group(1).lower()
#     if token in ('yes', 'y', '是'):
#         return 'yes'
#     return 'no'
# [R09-R44 修订]
def extract_yes_no(text: str) -> str:
    if not isinstance(text, str):
        return 'unk'
    answer = re.sub(r'(^[*_`]+|[*_`]+$)', '', text.strip()).strip()
    match = re.fullmatch(r'(yes|no|y|n|是|否|不是)[.!。！]?', answer, flags=re.I)
    if match is None:
        return 'unk'
    return 'yes' if match.group(1).lower() in ('yes', 'y', '是') else 'no'

# [R09-R44 原代码保留]
# def extract_abbr(raw: str) -> str:
#     """从一行文本中提取 23 种缩写之一；失败返 '<UNK>'。"""
#     first = re.split(r"[\r\n]", raw.strip(), maxsplit=1)[0].strip()
#     first = re.sub(r"(^[*_`]+|[*_`]+$)", "", first)
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
# [R09-R44 修订]
def extract_abbr(text: str) -> str:
    return parse_label(text, LABEL_ABBR, native_thinking=False)['label']

# ---------- 5. 加载模型 ----------
print("[INFO] Loading model …")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# CodeLlama 部分权重无 pad_token，手动对齐到 eos 以避免 batch/警告
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

# ---------- 6. 统一的“生成首行文本” ----------
def _render_with_template(messages: list) -> str:
    """优先使用 chat_template；若无，则回退到 [INST] ... [/INST] 手写模板。"""
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # 简单回退模板：把所有 role/content 展平成一个 INST 段
        def _join_msgs(msgs):
            return "\n".join([f"{m['role'].upper()}: {m['content']}" for m in msgs])
        body = _join_msgs(messages)
        return f"[INST] {body} [/INST]"

# [R09-R44 原代码保留]
# @torch.inference_mode()
# # [R08 原代码保留]
# # def greedy_generate_first_line(messages, max_new_tokens: int) -> str:
# # [R08 修订]
# def greedy_generate_first_line(messages, max_new_tokens: int, return_full_text: bool=False) -> str:
#     text = _render_with_template(messages)
#     inputs = tokenizer(text, return_tensors="pt",
#                        truncation=True, max_length=4096).to(model.device)
#     out = model.generate(
#         **inputs,
#         max_new_tokens=max_new_tokens,
#         do_sample=True,
#         temperature=0.2,
#         top_p=0.95,
#         eos_token_id=tokenizer.eos_token_id,
#         pad_token_id=tokenizer.pad_token_id,
#     )
#     raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
#                            skip_special_tokens=True).strip()
#     # [R08 原代码保留]
#     # return raw.split("\n", 1)[0].strip()
#     # [R08 修订]
#     if return_full_text:
#         return raw.strip()
#     return raw.split("\n", 1)[0].strip()
# [R09-R44 修订]
# [R45 OFFICIAL PROFILE 原代码保留]
# def greedy_generate_first_line(messages, max_new_tokens: int, return_full_text=True, stage='classification'):
#     event = audited_generate(
#         model, tokenizer, messages, {'max_new_tokens': max_new_tokens, 'do_sample': True, 'temperature': 0.2, 'top_p': 0.95, 'eos_token_id': tokenizer.eos_token_id, 'pad_token_id': tokenizer.pad_token_id},
#         audit=audit, max_input_tokens=4096, native_thinking=False,
#         reasoning_effort=None, stage=stage,
#     )
#     # Full raw response and observed stopping evidence are persisted by RunAudit.
#     return event.get('answer_text', '')
# [R45 OFFICIAL PROFILE 修订]
def greedy_generate_first_line(messages, max_new_tokens: int, return_full_text=True, stage='classification'):
    event = audited_generate(
        model, tokenizer, messages,
        generation_kwargs={'max_new_tokens': max_new_tokens},
        official_profile=True,
        audit=audit,
        max_input_tokens=4096,
        native_thinking=False,
        stage=stage,
    )
    return event.get('answer_text', '')

# ---------- 7. classify（轮询 Yes/No → 从候选选 1 个） ----------
# [R09-R44 原代码保留]
# @torch.inference_mode()
# # [R08 原代码保留]
# # def classify_by_polling(comment: str):
# #     yes_candidates = []
# #
# #     # 7.1 逐标签 Yes/No
# #     for full in LABELS:
# #         abbr = FULL2ABBR[full]
# #         prompt = build_yn_prompt(comment, full, abbr)
# #         messages = [{"role": "user", "content": prompt}]   # CodeLlama 模板常见仅 user
# #         raw_yn = greedy_generate_first_line(messages, MAX_YN_TOKENS)
# #         yn = extract_yes_no(raw_yn)
# #         if yn == "yes":
# #             yes_candidates.append(abbr)
# #
# #     # 7.2 根据候选决定最终结果
# #     if len(yes_candidates) == 1:
# #         final_abbr = yes_candidates[0]
# #         raw_final = final_abbr
# #     elif len(yes_candidates) >= 2:
# #         pick_prompt = build_pick_from_candidates_prompt(comment, yes_candidates)
# #         raw_pick = greedy_generate_first_line([{"role":"user","content":pick_prompt}],
# #                                               MAX_PICK_TOKENS)
# #         pick_abbr = extract_abbr(raw_pick)
# #         if pick_abbr in yes_candidates:
# #             final_abbr, raw_final = pick_abbr, raw_pick
# #         else:
# #             final_abbr, raw_final = yes_candidates[0], raw_pick
# #     else:
# #         # 没有任何 Yes：走全量强制单标签输出
# #         fb_prompt = build_force_pick_any_prompt(comment)
# #         raw_fb = greedy_generate_first_line([{"role":"user","content":fb_prompt}],
# #                                             MAX_FALLBACK_TOKENS)
# #         fb_abbr = extract_abbr(raw_fb)
# #         if fb_abbr in ABBR_SET:
# #             final_abbr, raw_final = fb_abbr, raw_fb
# #         else:
# #             final_abbr, raw_final = FALLBACK_ABBR, raw_fb
# #
# #     return final_abbr, raw_final, yes_candidates
# # [R08 修订]
# def classify_by_polling(comment: str, diagnostics=None):
#     yes_candidates = []
#     unknown_labels = []
#     fallback_reason = ''
#     for full in LABELS:
#         abbr = FULL2ABBR[full]
#         prompt = build_yn_prompt(comment, full, abbr)
#         messages = [{'role': 'user', 'content': prompt}]
#         # [R08 修订] Y/N使用完整响应，以便拒绝后续行中的冲突答案。
#         raw_yn = greedy_generate_first_line(messages, MAX_YN_TOKENS, return_full_text=True)
#         yn = extract_yes_no(raw_yn)
#         if yn == 'yes':
#             yes_candidates.append(abbr)
#         elif yn not in ('yes', 'no'):
#             unknown_labels.append(abbr)
#     if not unknown_labels and len(yes_candidates) == 1:
#         final_abbr = yes_candidates[0]
#         raw_final = final_abbr
#     elif not unknown_labels and len(yes_candidates) >= 2:
#         pick_prompt = build_pick_from_candidates_prompt(comment, yes_candidates)
#         raw_pick = greedy_generate_first_line([{'role': 'user', 'content': pick_prompt}], MAX_PICK_TOKENS)
#         pick_abbr = extract_abbr(raw_pick)
#         if pick_abbr in yes_candidates:
#             (final_abbr, raw_final) = (pick_abbr, raw_pick)
#         else:
#             (final_abbr, raw_final) = (FALLBACK_ABBR, raw_pick)
#             fallback_reason = 'candidate_arbitration_invalid'
#     else:
#         fallback_reason = 'unresolved_votes_full_label' if unknown_labels else 'no_positive_votes_full_label'
#         fb_prompt = build_force_pick_any_prompt(comment)
#         raw_fb = greedy_generate_first_line([{'role': 'user', 'content': fb_prompt}], MAX_FALLBACK_TOKENS)
#         fb_abbr = extract_abbr(raw_fb)
#         if fb_abbr in ABBR_SET:
#             (final_abbr, raw_final) = (fb_abbr, raw_fb)
#         else:
#             (final_abbr, raw_final) = (FALLBACK_ABBR, raw_fb)
#             fallback_reason += ';full_label_parse_failed'
#     if diagnostics is not None:
#         diagnostics.update({
#             'unknown_labels': list(unknown_labels),
#             'fallback_reason': fallback_reason,
#         })
#     return (final_abbr, raw_final, yes_candidates)
# [R09-R44 修订]
def classify_by_polling(comment: str, diagnostics=None):
    """Preserve the three-value interface; audit failures before Data Conf substitution."""
    yes_candidates, unknown_labels, votes = [], [], []
    fallback_reason, selection_route = '', 'single_positive'
    for full, abbr in LABEL_ABBR.items():
        prompt = build_yn_prompt(comment, full, abbr)
        messages = [{'role': 'user', 'content': prompt}]
        answer = greedy_generate_first_line(messages, MAX_YN_TOKENS, stage='yn:' + abbr)
        yn = extract_yes_no(answer)
        event = audit.events[-1] if audit.events else {}
        votes.append({
            'label': abbr,
            'parsed': yn,
            'raw_output': event.get('raw_text', ''),
            'generation_status': event.get('parse_status', 'unavailable'),
            'stop_status': event.get('stop_status', 'unavailable'),
        })
        if yn == 'yes':
            yes_candidates.append(abbr)
        elif yn != 'no':
            unknown_labels.append(abbr)
    raw_final = ''
    raw_pred = '<UNK>'
    if not unknown_labels and len(yes_candidates) == 1:
        raw_pred = yes_candidates[0]
    elif not unknown_labels and len(yes_candidates) >= 2:
        selection_route = 'candidate_arbitration'
        pick_prompt = build_pick_from_candidates_prompt(comment, yes_candidates)
        answer = greedy_generate_first_line([{'role': 'user', 'content': pick_prompt}], MAX_PICK_TOKENS, stage='candidate_arbitration')
        parsed = parse_label(answer, LABEL_ABBR, allowed=yes_candidates, native_thinking=False)
        raw_pred = parsed['label']
        raw_final = audit.events[-1].get('raw_text', '') if audit.events else ''
        if not parsed['parse_ok']:
            fallback_reason = 'candidate_arbitration_invalid'
    else:
        selection_route = 'full_label_unresolved_votes' if unknown_labels else 'full_label_no_positive'
        fallback_reason = 'unresolved_votes_full_label' if unknown_labels else 'no_positive_votes_full_label'
        full_prompt = build_force_pick_any_prompt(comment)
        answer = greedy_generate_first_line([{'role': 'user', 'content': full_prompt}], MAX_FALLBACK_TOKENS, stage='full_label_selection')
        parsed = parse_label(answer, LABEL_ABBR, native_thinking=False)
        raw_pred = parsed['label']
        raw_final = audit.events[-1].get('raw_text', '') if audit.events else ''
        if not parsed['parse_ok']:
            fallback_reason += ';full_label_parse_failed'
    parse_ok = raw_pred in ABBR_SET
    final_abbr = raw_pred if parse_ok else FALLBACK_ABBR
    if diagnostics is not None:
        diagnostics.update({
            'unknown_labels': list(unknown_labels),
            'fallback_reason': fallback_reason,
            'selection_route': selection_route,
            'raw_pred': raw_pred,
            'parse_ok': parse_ok,
            'parse_status': 'ok' if parse_ok else 'final_label_unparsed',
            'fallback_used': not parse_ok,
            'yn_unparsed_count': len(unknown_labels),
            'yn_decisions': votes,
        })
    return final_abbr, raw_final, yes_candidates

# ---------- 8. 主流程 ----------
df = pd.read_csv(CSV_PATH)
rows, unk_cnt = [], 0

print("[INFO] Classifying …")
# [R08 原代码保留]
# for cmt, gold in tqdm(zip(df["Comment Text"], df["ML TD Type"]),
#                       total=len(df), ncols=80):
#     pred_abbr, raw_choice, yes_list = classify_by_polling(cmt)
#     if pred_abbr not in ABBR_SET:
#         unk_cnt += 1
#         if PRINT_UNPARSED:
#             with open(RUN_DIR/"unparsed.txt", "a", encoding="utf-8") as f:
#                 f.write(str(raw_choice) + "\n")
#         pred_abbr = FALLBACK_ABBR
#
#     rows.append({
#         "comment": cmt,
#         "real": ABBR2FULL.get(gold, gold),
#         "pred": ABBR2FULL[pred_abbr],
#         "raw_output": raw_choice,             # 最终决策那一步的原始输出
#         "yes_candidates": ";".join(yes_list), # 该条评论获得 Yes 的缩写集合
#     })
# [R08 修订]
# [R09-R44 原代码保留]
# for (cmt, gold) in tqdm(zip(df['Comment Text'], df['ML TD Type']), total=len(df), ncols=80):
#     poll_diagnostics = {}
#     (pred_abbr, raw_choice, yes_list) = classify_by_polling(cmt, diagnostics=poll_diagnostics)
#     if pred_abbr not in ABBR_SET:
#         unk_cnt += 1
#         if PRINT_UNPARSED:
#             with open(RUN_DIR / 'unparsed.txt', 'a', encoding='utf-8') as f:
#                 f.write(str(raw_choice) + '\n')
#         pred_abbr = FALLBACK_ABBR
#     rows.append({
#         'comment': cmt,
#         'real': ABBR2FULL.get(gold, gold),
#         'pred': ABBR2FULL[pred_abbr],
#         'raw_output': raw_choice,
#         'yes_candidates': ';'.join(yes_list),
#         'unknown_labels': ';'.join(poll_diagnostics.get('unknown_labels', [])),
#         'fallback_reason': poll_diagnostics.get('fallback_reason', ''),
#     })
# [R09-R44 修订]
for source_row, (cmt, gold) in enumerate(tqdm(zip(df['Comment Text'], df['ML TD Type']), total=len(df), ncols=90)):
    audit.set_sample(source_row, df.iloc[source_row].to_dict())
    gold_result = parse_label(str(gold), LABEL_ABBR, native_thinking=False)
    if not gold_result['parse_ok']:
        raise ValueError(f'Unmapped gold label at source row {source_row}: {gold!r}')
    poll_diagnostics = {}
    pred_abbr, raw_choice, yes_list = classify_by_polling(str(cmt), diagnostics=poll_diagnostics)
    if not poll_diagnostics['parse_ok']:
        unk_cnt += 1
        if PRINT_UNPARSED:
            with open(RUN_DIR / 'unparsed.txt', 'a', encoding='utf-8') as f:
                f.write(json.dumps({'source_row': source_row, 'raw_output': raw_choice, 'status': poll_diagnostics['parse_status']}, ensure_ascii=False) + '\n')
    rows.append({
        **audit.sample_fields(),
        'comment': str(cmt),
        'original_gold': gold,
        'real': ABBR2FULL[gold_result['label']],
        'pred': ABBR2FULL[pred_abbr],
        'parsed_output': pred_abbr,
        'raw_output': raw_choice,
        'yes_candidates': ';'.join(yes_list),
        'unknown_labels': ';'.join(poll_diagnostics['unknown_labels']),
        'fallback_reason': poll_diagnostics['fallback_reason'],
        'selection_route': poll_diagnostics['selection_route'],
        'raw_pred': poll_diagnostics['raw_pred'],
        'parse_ok': poll_diagnostics['parse_ok'],
        'parse_status': poll_diagnostics['parse_status'],
        'fallback_used': poll_diagnostics['fallback_used'],
        'yn_unparsed_count': poll_diagnostics['yn_unparsed_count'],
        'yn_decisions': json.dumps(poll_diagnostics['yn_decisions'], ensure_ascii=False),
        'generation_events': json.dumps(audit.events, ensure_ascii=False, default=str),
    })

pred_df = pd.DataFrame(rows)

# 终端打印
report_text = classification_report(pred_df["real"], pred_df["pred"],
                                    labels=LABELS, zero_division=0, digits=3)
print("\n[INFO] Overall metrics:\n", report_text)

# ---------- 9. 保存结果 ----------
# 9.1 逐样本预测
pred_df.to_csv(RUN_DIR/"predictions.csv", index=False)

# 9.2 混淆矩阵（行真列预测）
pd.DataFrame(
    confusion_matrix(pred_df["real"], pred_df["pred"], labels=LABELS),
    index=LABELS, columns=LABELS
).to_csv(RUN_DIR/"confusion_matrix.csv")

# 9.3 classification_report 存为 CSV，并追加 INFO 行
report_dict = classification_report(pred_df["real"], pred_df["pred"],
                                    labels=LABELS, zero_division=0, output_dict=True)
rep_df = pd.DataFrame(report_dict).transpose().reset_index().rename(columns={"index":"label"})
rep_df["info"] = ""
info_line = f"Unrecognized mapped to '{FALLBACK_ABBR}': {unk_cnt}/{len(df)}"
rep_df.loc[len(rep_df)] = {
    "label": "[INFO]", "precision": "", "recall": "", "f1-score": "", "support": "", "info": info_line
}
rep_df.to_csv(RUN_DIR/"report.csv", index=False)

# 9.4 保存设置
# [R45 OFFICIAL PROFILE 原代码保留]
# (Path(RUN_DIR/"settings.json")).write_text(json.dumps({
#     "csv": CSV_PATH,
#     "model": MODEL_PATH,
#     "model_name": MODEL_NAME,
#     "run_dir": str(RUN_DIR),
#     "max_yn_tokens": MAX_YN_TOKENS,
#     "max_pick_tokens": MAX_PICK_TOKENS,
#     "max_fallback_tokens": MAX_FALLBACK_TOKENS,
#     "fallback_abbr": FALLBACK_ABBR,
#     "method": "YN polling -> pick best",
# }, indent=2))
# [R45 OFFICIAL PROFILE 修订]
settings = audit.settings()
settings.update({
    'csv': CSV_PATH,
    'model': MODEL_PATH,
    'model_name': MODEL_NAME,
    'run_dir': str(RUN_DIR),
    'max_yn_tokens': MAX_YN_TOKENS,
    'max_pick_tokens': MAX_PICK_TOKENS,
    'max_fallback_tokens': MAX_FALLBACK_TOKENS,
    'fallback_abbr': FALLBACK_ABBR,
    'method': 'YN polling -> pick best',
    'generation_profile_policy': 'official recommendation; otherwise fresh checkpoint defaults; see generation_events.jsonl',
})
(RUN_DIR / 'settings.json').write_text(json.dumps(settings, ensure_ascii=False, indent=2, default=str), encoding='utf-8')

print(f"\n[INFO] Files saved in {RUN_DIR}")
print(f"[INFO] Unrecognized mapped to '{FALLBACK_ABBR}': {unk_cnt}/{len(df)}")

gc.collect(); torch.cuda.empty_cache()


# [R09-R44 修订] Additional metrics treat final parse failures as errors; primary report retains Data Conf policy.
if not pred_df.empty:
    write_failure_metrics(pred_df, LABELS, RUN_DIR)
(RUN_DIR / 'revision_settings.json').write_text(
    json.dumps(audit.settings(), ensure_ascii=False, indent=2, default=str), encoding='utf-8'
)
