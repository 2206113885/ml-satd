# [R22/R24/R31/R33/R43 原代码保留] 历史入口整体归档；以下原文不参与执行。
# import re, os, gc, json
# from pathlib import Path
# from datetime import datetime
# import pandas as pd
# from tqdm import tqdm
# from sklearn.metrics import classification_report, confusion_matrix
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from collections import Counter
# 
# # -------- 0. 基础路径（仅此处改） --------
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"        # GPU 号
# CSV_PATH   = "/data/yangzhifei/pro/ML_SATD/dataset/sampled_label_dataset_normalized_rmvtodo.csv"  # ← 修改
# MODEL_PATH = "/data/yangzhifei/pro/ML_SATD/models/Qwen3-14B"                                        # ← 修改
# 
# # 目录名包含时间戳 + 模型名
# MODEL_NAME = Path(MODEL_PATH).name
# RUN_DIR    = Path("results") / f"run-{datetime.now():%Y%m%d-%H%M%S}-{MODEL_NAME}"
# RUN_DIR.mkdir(parents=True, exist_ok=True)
# 
# # -------- 1. 其它参数 --------
# PRINT_UNPARSED   = True
# SAVE_COT_TO_CSV  = True     # ← 是否将阶段1生成的推理保存到 predictions.csv 的 'cot' 列
# FALLBACK_ABBR    = "Data Conf"
# MAX_NEW_TOKENS   = 2048     # 充足即可，避免 32768 带来的极慢与OOM
# CTX_MAX_TOKENS   = 4096
# 
# # Zero-shot 推荐：阶段2贪婪；阶段1若不开启自一致，也可贪婪；若开启自一致，对阶段1做采样
# USE_SELF_CONSISTENCY = True
# SC_SAMPLES = 5
# SC_TEMP    = 0.7
# SC_TOP_P   = 0.95
# SC_TOP_K   = 50
# 
# # ----------------- 标签映射 -----------------
# LABEL_ABBR = {
#     "Machine Learning Knowledge":"MLKnow","Model Interpretability":"MLInter",
#     "Model Code Comprehension":"MCC","Duplicate Model Code":"DMC",
#     "Duplicate Feature Extraction Code":"DFEC","Weight Configuration":"Weight Conf",
#     "Layer Configuration":"Layer Conf","Hyper-parameter Configuration":"HP Conf",
#     "Code Dependency":"MLDepend","Glue Code":"GC","Custom Data Type":"CDT",
#     "Multiple Languages":"ML","Unnecessary Model Code":"UMC",
#     "Data Processing Configuration":"Data Conf","Plain Old Data Type":"PODT",
#     "Data Storage Configuration":"DS Conf","Abstraction":"Abs","Boundary Erosion":"BE",
#     "Model Code Modifiability":"MCM","Model Code Reusability":"MCR",
#     "Prediction Quality":"PQ","Machine Learning Reliability":"MLRelia","Prototype":"Proto",
# }
# ABBR2FULL  = {v:k for k,v in LABEL_ABBR.items()}
# LABELS     = list(LABEL_ABBR.keys())
# ABBR_SET   = set(ABBR2FULL.keys())
# ABBR_LOWER = {a.lower():a for a in ABBR_SET}
# 
# # 预编译缩写匹配（忽略大小写，缩写里的空格可被任意空白匹配）
# def _esc(a: str) -> str:
#     return re.escape(a).replace(r"\ ", r"\s+")
# ABBR_PATTERN = re.compile(r"\b(" + "|".join(sorted((_esc(a) for a in ABBR_SET), key=len, reverse=True)) + r")\b", re.IGNORECASE)
# 
# # ============ 2. Prompt 头（仅描述，无示例） ============
# PROMPT_HEAD = r"""
# There are 23 types of Machine Learning (ML) SATD (Self-Admitted Technical Debt):
# 
# MACHINE LEARNING KNOWLEDGE (MLKnow): Machine-learning software carries plenty of unique challenges. Uneducated solutions by unaware developers may have to be revisited.
# 
# MODEL INTERPRETABILITY (MLInter): Machine-learning models are a black box, causing poor understanding of model functionality; this can lead to unknown behaviour.
# 
# MODEL CODE COMPREHENSION (MCC): Model code carries extra legibility concerns not present in traditional software (e.g., poorly-named temporary matrix variables).
# 
# DUPLICATE MODEL CODE (DMC): Code duplication frequently occurs in model code.
# 
# DUPLICATE FEATURE EXTRACTION CODE (DFEC): Code duplication frequently occurs in feature-extraction code.
# 
# WEIGHT CONFIGURATION (Weight Conf): Editing code that involves the weights of a model, or configuring a model's weights directly.
# 
# LAYER CONFIGURATION (Layer Conf): Editing code that deals with model layers, or configuring layers themselves.
# 
# HYPER-PARAMETER CONFIGURATION (HP Conf): Configuring hyper-parameters of a model, or editing default values of off-the-shelf models.
# 
# CODE DEPENDENCY (MLDepend): A needed change is blocked by an external dependency (library / subsystem).
# 
# GLUE CODE (GC): Supporting code written to interface with other code, inhibiting improvements due to dependent peculiarities.
# 
# CUSTOM DATA TYPE (CDT): Using data types provided by general-purpose packages can cause extensive inter-operability burden.
# 
# MULTIPLE LANGUAGES (ML): Components written in other languages may introduce difficulties in ML development.
# 
# UNNECESSARY MODEL CODE (UMC): Model code that bottlenecks execution, is unreachable / deprecated, or is unnecessary and should be removed.
# 
# DATA PROCESSING CONFIGURATION (Data Conf): Configuring how data is processed, by editing the data directly or adding new processing steps.
# 
# PLAIN OLD DATA TYPE (PODT): Using raw data types in ML causes confusion when interpreting processes.
# 
# DATA STORAGE CONFIGURATION (DS Conf): Configuring how data is represented internally or stored externally.
# 
# ABSTRACTION (Abs): Lack of abstractions in ML systems causes cascading changes when a component is modified.
# 
# BOUNDARY EROSION (BE): Lack of boundaries between subsystems complicates maintenance and isolates changes poorly.
# 
# MODEL CODE MODIFIABILITY (MCM): Model code should enable easy maintenance and future modifications.
# 
# MODEL CODE REUSABILITY (MCR): Model code should be generalised for reuse in varying situations.
# 
# PREDICTION QUALITY (PQ): Workflow changes may affect performance.
# 
# MACHINE LEARNING RELIABILITY (MLRelia): Model functionality depends on data quality; robustness measures should be in place.
# 
# PROTOTYPE (Proto): Deploying small-scale prototypes into full systems can be dangerous.
# """.strip()
# 
# ABBR_LIST_TEXT = ", ".join(sorted(ABBR_SET))
# PROMPT_TAIL = f"""Please determine which type of ML Self-Admitted Technical Debt (SATD) the following Code Comment belongs to.
# Let's think step by step. You should first give detailed explain after the #Reason: and then give the answer after the #Answer: with only ONE abbreviation from:
# {ABBR_LIST_TEXT}
# """.strip()
# 
# 
# 
# # -------- 4. 解析函数（稳健版） --------
# def extract_abbr(text: str) -> str:
#     # 先用缩写正则抓取第一个合法命中
#     m = ABBR_PATTERN.search(text)
#     if m:
#         token = re.sub(r"\s+", " ", m.group(1)).strip()   # 标准化内部空白
#         # 归一化大小写到标准形式
#         key = token.lower().replace(" ", "")
#         for a in ABBR_SET:
#             if a.lower().replace(" ", "") == key:
#                 return a
#     # 回退到宽松解析（你的旧逻辑增强）
#     first = re.split(r"[\r\n]", text.strip())[0]
#     token = re.sub(r"[^a-z]", "", first.lower())
#     if token in ABBR_LOWER:
#         return ABBR_LOWER[token]
#     raw_strip = first.strip()
#     if raw_strip in ABBR2FULL:
#         return raw_strip
#     join = raw_strip.replace(" ", "")
#     if join.lower() in ABBR_LOWER:
#         return ABBR_LOWER[join.lower()]
#     return "<UNK>"
# 
# # -------- 5. 加载模型 --------
# print("[INFO] Loading model …")
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
# # 解码器-only模型建议左侧padding，避免警告和影响生成
# try:
#     tokenizer.padding_side = "left"
# except Exception:
#     pass
# 
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_PATH,
#     device_map="auto",
#     torch_dtype="auto"
# )
# 
# def _encode_chat(prompt: str):
#     # 使用 chat_template，但不启用“隐式思维标签”，我们显式用自然语言引导CoT
#     messages = [{"role":"user", "content": prompt}]
#     text = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True,
#         # enable_thinking=False  # 保持默认；不依赖 <think> 特性
#     )
#     return tokenizer(text, return_tensors="pt",
#                      truncation=True, max_length=CTX_MAX_TOKENS).to(model.device)
# 
# @torch.inference_mode()
# def _gen(inputs, max_new_tokens=512, temperature=0.0, top_p=1.0, top_k=0):
#     return model.generate(
#         **inputs,
#         max_new_tokens=max_new_tokens,
#         do_sample=(temperature > 0),
#         temperature=temperature,
#         top_p=top_p,
#         top_k=top_k,
#         pad_token_id=tokenizer.eos_token_id
#     )
# 
# # -------- 6. 两阶段 CoT 推理 --------
# @torch.inference_mode()
# def classify_one(comment: str):
#     # --- 阶段1：推理 ---
#     if USE_SELF_CONSISTENCY:
#         votes, reasons = [], []
#         for _ in range(SC_SAMPLES):
#             inp1 = _encode_chat(build_cot_stage1(comment))
#             out1 = _gen(inp1, max_new_tokens=512, temperature=SC_TEMP, top_p=SC_TOP_P, top_k=SC_TOP_K)
#             cot = tokenizer.decode(out1[0][inp1["input_ids"].shape[1]:], skip_special_tokens=True).strip()
#             # 阶段2：基于每次采样的推理取最终缩写（贪婪）
#             inp2 = _encode_chat(build_cot_stage2(comment, cot))
#             out2 = _gen(inp2, max_new_tokens=64, temperature=0.0)
#             ans = tokenizer.decode(out2[0][inp2["input_ids"].shape[1]:], skip_special_tokens=True)
#             abbr = extract_abbr(ans)
#             if abbr != "<UNK>":
#                 votes.append(abbr)
#                 reasons.append((abbr, cot))
#         if votes:
#             winner, _ = Counter(votes).most_common(1)[0]
#             # 取与胜出缩写相对应的一条推理存档
#             picked_reason = next((c for a,c in reasons if a==winner), "")
#             return winner, picked_reason, ans
#         # 若全是UNK，继续走单次贪婪作为回退
#     # --- 单次贪婪 CoT ---
#     inp1 = _encode_chat(build_cot_stage1(comment))
#     out1 = _gen(inp1, max_new_tokens=512, temperature=0.0)
#     cot = tokenizer.decode(out1[0][inp1["input_ids"].shape[1]:], skip_special_tokens=True).strip()
# 
#     inp2 = _encode_chat(build_cot_stage2(comment, cot))
#     out2 = _gen(inp2, max_new_tokens=64, temperature=0.0)
#     ans = tokenizer.decode(out2[0][inp2["input_ids"].shape[1]:], skip_special_tokens=True)
# 
#     abbr = extract_abbr(ans)
#     return abbr, cot, ans
# 
# # -------- 7. 主流程 --------
# df = pd.read_csv(CSV_PATH)
# rows, unk_cnt = [], 0
# 
# print("[INFO] Classifying …")
# for cmt, gold in tqdm(zip(df["Comment Text"], df["ML TD Type"]),
#                       total=len(df), ncols=80):
#     abbr, cot_reasoning, raw_answer = classify_one(cmt)
#     if abbr == "<UNK>":
#         unk_cnt += 1
#         if PRINT_UNPARSED:
#             with open(RUN_DIR/"unparsed.txt", "a", encoding="utf-8") as f:
#                 f.write(raw_answer.strip() + "\n")
#         abbr = FALLBACK_ABBR
#         cot_reasoning = cot_reasoning if SAVE_COT_TO_CSV else ""
# 
#     row = {
#         "comment": cmt,
#         "real": ABBR2FULL.get(gold, gold),
#         "pred": ABBR2FULL[abbr],
#         "raw_answer": raw_answer.strip()
#     }
#     if SAVE_COT_TO_CSV:
#         row["cot"] = cot_reasoning.strip()
#     rows.append(row)
# 
# pred_df = pd.DataFrame(rows)
# 
# # 终端打印（保持原有的文本版）
# report_text = classification_report(pred_df["real"], pred_df["pred"],
#                                     labels=LABELS, zero_division=0, digits=3)
# print("\n[INFO] Overall metrics:\n", report_text)
# 
# # -------- 8. 保存结果 --------
# # 8.1 逐样本预测
# pred_df.to_csv(RUN_DIR/"predictions.csv", index=False)
# 
# # 8.2 混淆矩阵
# pd.DataFrame(
#     confusion_matrix(pred_df["real"], pred_df["pred"], labels=LABELS),
#     index=LABELS, columns=LABELS
# ).to_csv(RUN_DIR/"confusion_matrix.csv")
# 
# # 8.3 classification_report 保存为 CSV，并追加未解析统计
# report_dict = classification_report(pred_df["real"], pred_df["pred"],
#                                     labels=LABELS, zero_division=0, output_dict=True)
# rep_df = pd.DataFrame(report_dict).transpose().reset_index()
# rep_df.rename(columns={"index":"label"}, inplace=True)
# rep_df["info"] = ""
# info_line = f"Unrecognized mapped to '{FALLBACK_ABBR}': {unk_cnt}/{len(df)}"
# rep_df.loc[len(rep_df)] = {"label":"[INFO]","precision":"","recall":"","f1-score":"","support":"","info":info_line}
# rep_df.to_csv(RUN_DIR/"report.csv", index=False)
# 
# # 8.4 保存设置
# (Path(RUN_DIR/"settings.json")).write_text(json.dumps({
#     "csv": CSV_PATH,
#     "model": MODEL_PATH,
#     "run_dir": str(RUN_DIR),
#     "max_new_tokens": MAX_NEW_TOKENS,
#     "fallback_abbr": FALLBACK_ABBR,
#     "use_self_consistency": USE_SELF_CONSISTENCY,
#     "sc_samples": SC_SAMPLES,
#     "sc_temp": SC_TEMP,
#     "sc_top_p": SC_TOP_P,
#     "sc_top_k": SC_TOP_K,
#     "save_cot_to_csv": SAVE_COT_TO_CSV
# }, indent=2))
# 
# print(f"\n[INFO] Files saved in {RUN_DIR}")
# print(f"[INFO] Unrecognized mapped to '{FALLBACK_ABBR}': {unk_cnt}/{len(df)}")
# gc.collect(); 
# try:
#     torch.cuda.empty_cache()
# except Exception:
#     pass

# [R22/R24/R31/R33/R43 修订] 新两阶段显式CoT入口；旧原型及缺失构造器不复用。
"""Prospective explicit CoT/self-consistency protocol for local Qwen3.

Native thinking is explicitly disabled in both stages. Complete stage outputs,
sample IDs, trial seeds and effective configurations are recorded for every vote.
This does not establish historical model/template versions or repair old scores.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from auxiliary_protocol import (ABBREVIATIONS, file_sha256, load_labeled,
                                new_output_dir, stable_sample_seed, write_json)
from qwen_auxiliary_runtime import LABEL_DEFINITIONS, QwenRuntime, exact_abbreviation


def build_cot_stage1(comment):
    return (LABEL_DEFINITIONS + '\n\nTarget comment:\n' + json.dumps(comment, ensure_ascii=False) +
            '\nExplain the evidence for the single best ML-SATD category. This stage returns an explanation; a separate stage chooses the abbreviation.')


def build_cot_stage2(comment, explanation):
    return (LABEL_DEFINITIONS + '\n\nTarget comment:\n' + json.dumps(comment, ensure_ascii=False) +
            '\n\nEvidence from the explanation stage:\n' + explanation +
            '\n\nChoose the single best label for the target comment. Return one abbreviation only: ' + ', '.join(ABBREVIATIONS))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-csv', required=True, type=Path)
    parser.add_argument('--model-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=20261016)
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--input-cap', type=int, default=4096)
    parser.add_argument('--reason-tokens', type=int, default=512)
    parser.add_argument('--answer-tokens', type=int, default=64)
    args = parser.parse_args()
    if args.samples < 1 or args.reason_tokens < 1 or args.answer_tokens < 1:
        raise ValueError('Sample count and token budgets must be positive')
    rows = load_labeled(args.input_csv)
    output = new_output_dir(args.output_dir)
    runtime = QwenRuntime(args.model_dir, args.input_cap)
    write_json(output / 'settings.json', {
        'protocol': 'prospective explicit CoT + self-consistency v1',
        'input_sha256': file_sha256(args.input_csv), 'model_identity': runtime.identity,
        'loaded_generation_config': runtime.loaded_generation_config,
        'stage1_config': runtime.generation_config(args.reason_tokens).to_dict(),
        'stage2_config': runtime.generation_config(args.answer_tokens, greedy=True).to_dict(),
        'native_thinking': False, 'samples': args.samples, 'seed': args.seed,
        'seed_policy': 'SHA256([seed, sample_id, trial]); stage2 greedy',
        'vote_policy': 'valid votes only; a tied maximum or no votes is a strict failure',
        'failure_policy': 'strict failures count wrong; separate historical Data Conf fallback comparison',
        'runner_sha256': file_sha256(Path(__file__)),
        'runtime_sha256': file_sha256(Path(__file__).with_name('qwen_auxiliary_runtime.py')),
        'helper_sha256': file_sha256(Path(__file__).with_name('auxiliary_protocol.py')),
    })
    results = []
    with (output / 'trials.jsonl').open('w', encoding='utf-8') as trials_out, (output / 'predictions.jsonl').open('w', encoding='utf-8') as predictions:
        for row in rows:
            trials, votes = [], []
            for trial in range(args.samples):
                record = {'sample_id': row['sample_id'], 'trial': trial,
                          'seed': stable_sample_seed(args.seed, row['sample_id'], trial)}
                try:
                    reason_prompt = build_cot_stage1(row['comment'])
                    # [R24/R43 原代码保留]
                    # reason = runtime.generate(reason_prompt, args.reason_tokens, record['seed'])
                    # [R24/R43 修订]
                    record['stage1_prompt'] = reason_prompt
                    reason = runtime.generate(reason_prompt, args.reason_tokens, record['seed'])
                    record.update(stage1=reason, stage1_prompt=reason_prompt)
                    answer_prompt = build_cot_stage2(row['comment'], reason['raw_response'])
                    # [R24/R43 原代码保留]
                    # answer = runtime.generate(answer_prompt, args.answer_tokens, record['seed'], greedy=True)
                    # [R24/R43 修订]
                    record['stage2_prompt'] = answer_prompt
                    answer = runtime.generate(answer_prompt, args.answer_tokens, record['seed'], greedy=True)
                    label = exact_abbreviation(answer['raw_response'])
                    record.update(stage1=reason, stage2=answer, stage1_prompt=reason_prompt,
                                  stage2_prompt=answer_prompt, predicted_label=label,
                                  parse_ok=label is not None, parse_error=None if label else 'unrecognized_output')
                    if label:
                        votes.append(label)
                except ValueError as exc:
                    if not str(exc).startswith('input_budget_exceeded'):
                        raise
                    # [R24 原代码保留]
                    # record.update(predicted_label=None, parse_ok=False, parse_error='input_budget_exceeded')
                    # [R24 修订]
                    record.update(predicted_label=None, parse_ok=False, parse_error='input_budget_exceeded',
                                  input_failure_detail=str(exc))
                trials_out.write(json.dumps(record, ensure_ascii=False) + '\n')
                trials_out.flush()
                trials.append(record)
            counts = Counter(votes)
            winners = [key for key, value in counts.items() if value == max(counts.values())] if counts else []
            predicted = winners[0] if len(winners) == 1 else None
            result = {**row, 'predicted_label': predicted, 'parse_ok': predicted is not None,
                      'raw_votes': votes, 'vote_counts': dict(counts),
                      'fallback_reason': '' if predicted else ('tied_votes' if winners else 'no_valid_votes'),
                      'legacy_policy_prediction': predicted or 'Data Conf'}
            predictions.write(json.dumps(result, ensure_ascii=False) + '\n')
            predictions.flush()
            results.append(result)
    write_json(output / 'summary.json', {
        'n': len(results), 'strict_accuracy': sum(r['predicted_label'] == r['gold_label'] for r in results) / len(results),
        'legacy_fallback_accuracy': sum(r['legacy_policy_prediction'] == r['gold_label'] for r in results) / len(results),
        'strict_failures': sum(not r['parse_ok'] for r in results),
    })


if __name__ == '__main__':
    main()
