"""Source editor only. Never imports or calls experimental modules."""
from pathlib import Path
import ast
import hashlib
import json
import re

BASE = Path('/Users/yizho/Desktop/satd_experiments_revised_20261016')
NOTES = BASE / 'revision_batch_03'
manifest = json.loads((BASE/'revision_batch_01/source_copy_manifest.json').read_text())
files = [row['file'] for row in manifest['code_files'] if row['file'].split('/')[0].startswith('basic_')]


def preserve(block, replacement, tag):
    old = ''.join('# '+line if line.strip() else '#\n' for line in block.splitlines(keepends=True))
    if old and not old.endswith('\n'):
        old += '\n'
    return '# ['+tag+' 原代码保留]\n'+old+'# ['+tag+' 修订]\n'+replacement.rstrip()+'\n'


def kw_for_function(func, functions):
    if func is None:
        return {'max_new_tokens': 'MAX_NEW_TOKENS'}
    for node in ast.walk(func):
        if isinstance(node, ast.Call) and ((isinstance(node.func,ast.Attribute) and node.func.attr=='generate') or (isinstance(node.func,ast.Name) and node.func.id in ['pipe','pipeline'])):
            return {k.arg: ast.unparse(k.value) for k in node.keywords if k.arg is not None and k.arg not in ['return_full_text','return_tensors']}
    if 'text_gen' in functions:
        return kw_for_function(functions['text_gen'], {})
    return {'max_new_tokens': 'MAX_NEW_TOKENS'}


def dict_source(values):
    return '{'+', '.join(repr(k)+': '+v for k,v in values.items())+'}'


prepared = []
for rel in files:
    path = BASE/rel
    text = path.read_text()
    lines = text.splitlines(keepends=True)
    tree = ast.parse(text)
    funcs = {n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    assigned = {t.id:n for n in tree.body if isinstance(n,ast.Assign) for t in n.targets if isinstance(t,ast.Name)}
    assert 'LABEL_ABBR' in assigned, rel
    classifier = funcs['classify']
    original_classify = ''.join(lines[classifier.lineno-1:classifier.end_lineno])
    is_gpt = 'gpt-oss' in path.name
    is_qwen3 = 'Qwen3' in path.name
    is_coder = 'Qwen2.5' in path.name
    post_reason = path.parent.name in ['basic_few_shot_CoT_reason','basic_zero_shot_CoT_reason']
    inline_reason = path.parent.name in ['basic_zero_shot_reason','basic_zero_shot_CoT_output_reason']
    native = is_qwen3 and any(isinstance(n,ast.keyword) and n.arg=='enable_thinking' and isinstance(n.value,ast.Constant) and n.value.value is True for n in ast.walk(classifier))
    model_expr, tok_expr = ('pipe.model','pipe.tokenizer') if 'pipe' in assigned else ('pipeline.model','pipeline.tokenizer') if 'pipeline' in assigned else ('model','tokenizer')
    gen = kw_for_function(classifier, funcs)
    gen.setdefault('max_new_tokens','MAX_NEW_TOKENS')
    if is_qwen3:
        gen.update(do_sample='True',temperature='0.6' if native else '0.7',top_p='0.95' if native else '0.8',top_k='20',min_p='0.0')
    effort_match = re.search(r'Reasoning:\s*(low|medium|high)', original_classify, re.I)
    effort_default = effort_match.group(1).lower() if effort_match else 'medium'
    effort = "os.environ.get('SATD_REASONING_EFFORT', "+repr(effort_default)+")" if is_gpt else 'None'
    reason_gen = kw_for_function(funcs.get('generate_reason'), {}) if post_reason else {}
    if post_reason:
        if is_qwen3:
            reason_gen.update(do_sample='True',temperature='0.7',top_p='0.8',top_k='20',min_p='0.0')
        reason_gen.setdefault('max_new_tokens','MAX_NEW_TOKENS')
    legacy_caps = [n.value.value for n in ast.walk(classifier) if isinstance(n,ast.keyword) and n.arg=='max_length' and isinstance(n.value,ast.Constant) and isinstance(n.value.value,int)]
    legacy_cap = str(legacy_caps[0]) if legacy_caps else 'None'
    replacement_classify = f'''def classify(comment: str):
    messages = [{{"role": "user", "content": build_prompt(comment)}}]
    event = audited_generate({model_expr}, {tok_expr}, messages,
                             {dict_source(gen)}, audit=_REVISION_AUDIT,
                             max_input_tokens={legacy_cap}, native_thinking={native!r},
                             reasoning_effort={effort}, stage="classification")
    parsed = parse_classification(event["answer_text"], LABEL_ABBR, with_reason={inline_reason!r})
    if event["parse_status"] != "ok":
        parsed.update(label="<UNK>", parse_ok=False, status=event["parse_status"])
    return {{**parsed, "raw_text": event["raw_text"], "generation_parse_status": event["parse_status"]}}
'''
    # Preserve any non-reasoning system instructions; the actual effort goes
    # into the GPT-OSS chat template rather than a body-string convention.
    message_assign = next((n for n in classifier.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='messages' for t in n.targets)), None)
    if message_assign is not None:
        msg_source = ast.unparse(message_assign.value)
        msg_source = re.sub(r'Reasoning:\s*(?:low|medium|high)\.?\s*', '', msg_source, flags=re.I)
        replacement_classify = replacement_classify.replace('[{"role": "user", "content": build_prompt(comment)}]', msg_source, 1)
    edits=[]
    for name, f in funcs.items():
        replacement = None
        tag = 'R13/R16/R23/R24/R29/R31/R33'
        if name=='classify':
            replacement = replacement_classify
        elif name in ['simple_extract','extract_abbr','extract_abbr_from_text','extract_tagged_answer']:
            replacement = f'def {name}(raw: str) -> str:\n    return parse_label(raw, LABEL_ABBR)["label"]\n'
            tag = 'R13/R16'
        elif name=='parse_abbr_and_reason':
            replacement = '''def parse_abbr_and_reason(raw: str):
    result = parse_classification(raw, LABEL_ABBR, with_reason=True)
    return result["label"], result["label"], result.get("reason", "")
'''
            tag='R13/R16'
        elif name=='_extract_visible':
            replacement='def _extract_visible(raw: str) -> str:\n    return visible_answer(raw)\n'
            tag='R13/R23'
        elif name in ['extract_reason','extract_reason_from_text','parse_reason']:
            replacement=f'''def {name}(raw: str) -> str:
    if not isinstance(raw, str):
        return ""
    view = visible_answer(raw)
    match = re.search(r"(?mis)^\\s*(?:#Reason|Reason)\\s*[:：]\\s*(.+)", view)
    return match.group(1).strip() if match else ""
'''
            tag='R13/R33'
        elif name=='generate_reason':
            replacement=f'''def generate_reason(comment: str, final_abbr: str) -> str:
    event = audited_generate({model_expr}, {tok_expr},
                             [{{"role": "user", "content": build_reason_prompt(comment, final_abbr)}}],
                             {dict_source(reason_gen)}, audit=_REVISION_AUDIT,
                             native_thinking=False,
                             reasoning_effort={"os.environ.get('SATD_REASONING_EFFORT', 'low')" if is_gpt else 'None'},
                             stage="posthoc_reason")
    if event["parse_status"] != "ok":
        return ""
    visible = event["answer_text"].strip()
    return re.sub(r"(?i)^\\s*(?:#Reason|Reason)\\s*[:：]\\s*", "", visible).strip()
'''
            tag='R13/R21/R23/R24/R29/R31'
        elif name=='text_gen':
            replacement=f'''def text_gen(messages, MAX_NEW_TOKENS):
    event = audited_generate({model_expr}, {tok_expr}, messages,
                             {{"max_new_tokens": MAX_NEW_TOKENS,
                              "pad_token_id": {tok_expr}.eos_token_id}},
                             audit=_REVISION_AUDIT, stage="text_gen")
    return [{{"generated_text": [{{"role": "assistant", "content": event["answer_text"]}}]}}]
'''
            tag='R22/R24/R31'
        if replacement is not None:
            start=min([f.lineno]+[d.lineno for d in f.decorator_list])-1
            edits.append((start,f.end_lineno,preserve(''.join(lines[start:f.end_lineno]),replacement,tag)))
    # Fix paired comment delimiters and the omitted MCC rule in live strings.
    bp=funcs['build_prompt']
    bp_source=''.join(lines[bp.lineno-1:bp.end_lineno])
    if 'Comment:\\n' in bp_source and '\\n\\"\\"\\"' in bp_source and 'Comment:\\n\\"\\"\\"' not in bp_source:
        revised=bp_source.replace('Comment:\\n','Comment:\\n\\"\\"\\"\\n')
        edits.append((bp.lineno-1,bp.end_lineno,preserve(bp_source,revised,'R36')))
    for n in tree.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PROMPT_TAIL' for t in n.targets):
            segment=''.join(lines[n.lineno-1:n.end_lineno])
            if 'Understanding & interpretability:' in segment and 'code readability/comprehension' not in segment:
                updated=segment.replace('Understanding & interpretability:', 'Understanding & interpretability: model-code readability/comprehension or understanding its implementation ⇒ {{MCC}};')
                edits.append((n.lineno-1,n.end_lineno,preserve(segment,updated,'R27')))
    main_node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='df' for t in n.targets) and isinstance(n.value,ast.Call))
    main=f'''# [R22/R31/R33 修订] Per-row identity and pre-fallback diagnostics; old policy report stays primary.
_REVISION_AUDIT = RunAudit(RUN_DIR, MODEL_PATH, CSV_PATH)
df = pd.read_csv(CSV_PATH)
rows = []
unk_cnt = 0
for source_row, source in tqdm(df.iterrows(), total=len(df), ncols=80):
    _REVISION_AUDIT.set_sample(source_row, source.to_dict())
    gold_result = parse_label(str(source["ML TD Type"]), LABEL_ABBR)
    if not gold_result["parse_ok"]:
        raise ValueError(f"Invalid gold label at source row {{source_row}}; no rows are silently skipped")
    cmt = str(source["Comment Text"])
    result = classify(cmt)
    raw_pred = result["label"]
    parse_ok = bool(result["parse_ok"])
    final_abbr = raw_pred if parse_ok else FALLBACK_ABBR
    unk_cnt += int(not parse_ok)
    reason = {"generate_reason(cmt, final_abbr)" if post_reason else 'result.get("reason", "")'}
    rows.append({{
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
    }})
pred_df = pd.DataFrame(rows)
pred_df.to_csv(RUN_DIR/"predictions.csv", index=False)
report_text = classification_report(pred_df["real"], pred_df["pred"], labels=LABELS, zero_division=0, digits=3)
print(report_text)
(RUN_DIR/"report.txt").write_text(report_text, encoding="utf-8")
report_dict = classification_report(pred_df["real"], pred_df["pred"], labels=LABELS, zero_division=0, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose().rename_axis("label").reset_index()
report_df["info"] = ""
report_df.loc[len(report_df)] = {{"label": "[INFO]", "info": f"Parsing failures before fallback: {{unk_cnt}} / {{len(df)}}"}}
report_df.to_csv(RUN_DIR/"report.csv", index=False)
pd.DataFrame(confusion_matrix(pred_df["real"], pred_df["pred"], labels=LABELS), index=LABELS, columns=LABELS).to_csv(RUN_DIR/"confusion_matrix.csv")
write_failure_metrics(pred_df, LABELS, RUN_DIR)
(RUN_DIR/"settings.json").write_text(json.dumps({{
    **_REVISION_AUDIT.settings(),
    "model": MODEL_PATH, "csv": CSV_PATH, "run_dir": str(RUN_DIR),
    "max_new_tokens": MAX_NEW_TOKENS,
    "reasoning_mode": {repr('native_thinking' if native else 'prompt_cot' if 'CoT' in path.parent.name else 'nonthinking')},
    "fallback_abbr": FALLBACK_ABBR,
    "metrics_policy": "Primary: existing final fallback predictions. Additional failure_as_error metrics mark invalid final parses wrong.",
    "input_policy": "Full templated input, including demonstrations/comment/code context; actual context minus output reserve and 64 safety tokens; no truncation.",
    "parser_policy": "Unique complete canonical abbreviation/full-name or the explicitly requested answer field; never search reasoning.",
    "unknown_count_before_fallback": unk_cnt,
}}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[INFO] Revised results saved to {{RUN_DIR}}")
'''
    edits.append((main_node.lineno-1,len(lines),preserve(''.join(lines[main_node.lineno-1:]),main,'R22/R31/R33')))
    header_end=0
    if lines and lines[0].startswith('#!'):header_end=1
    while header_end<len(lines) and re.search(r'coding[:=]',lines[header_end]):header_end+=1
    if tree.body and isinstance(tree.body[0],ast.Expr) and isinstance(tree.body[0].value,ast.Constant) and isinstance(tree.body[0].value.value,str):header_end=max(header_end,tree.body[0].end_lineno)
    for n in tree.body:
        if isinstance(n,ast.ImportFrom) and n.module=='__future__':header_end=max(header_end,n.end_lineno)
    bootstrap='''# [R13/R16/R17/R22/R23/R24/R29/R31/R33 修订] Shared explicit audit runtime.
import sys as _revision_sys
from pathlib import Path as _RevisionPath
_revision_sys.path.insert(0, str(_RevisionPath(__file__).resolve().parents[1]))
from revision_runtime import RunAudit, audited_generate, parse_label, parse_classification, visible_answer, write_failure_metrics
'''
    edits.append((header_end,header_end,bootstrap))
    # Revisions refer to the original line coordinates. Overlap is forbidden.
    occupied=[]
    for start,end,_ in edits:
        if start!=end:
            assert not any(start<oldend and end>oldstart for oldstart,oldend in occupied),(rel,start,end)
            occupied.append((start,end))
    newlines=list(lines)
    for start,end,content in sorted(edits, key=lambda e:e[0],reverse=True):
        newlines[start:end]=[content]
    revised=''.join(newlines)
    ast.parse(revised,filename=rel)
    prepared.append((path,revised,{'file':rel,'sha256_before':hashlib.sha256(text.encode()).hexdigest(),'sha256_after':hashlib.sha256(revised.encode()).hexdigest(),'native_thinking':native,'posthoc_reason':post_reason,'inline_reason':inline_reason,'generation_kwargs':gen,'gpt_effort_default':effort_default if is_gpt else None,'original_code_commented_in_place':True}))
for path,revised,_ in prepared:
    path.write_text(revised)
(NOTES/'basic_changes.json').write_text(json.dumps({'files_changed':len(prepared),'validation':'Static AST/source editing only; no experiment/module/function/model/tokenizer/test execution','changes':[r for _,_,r in prepared]},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'basic_files_revised':len(prepared),'ast_passed':len(prepared)}))
