#!/usr/bin/env python3
"""Read-only AST audit of original Qwen experiments and small official configs."""
import argparse
import ast
import csv
import hashlib
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = BASE.parent
SOURCES = BASE / 'qwen_sources'
REPOS = ['Qwen/Qwen3-8B', 'Qwen/Qwen3-14B', 'Qwen/Qwen2.5-Coder-7B-Instruct']
DIRS = ['basic_zero_shot','basic_few_shot','bisection_zero_shot','bisection_few_shot',
        'tree_zero_shot','tree_few_shot','basic_zero_shot_reason','basic_zero_shot_CoT',
        'basic_few_shot_CoT','basic_zero_shot_CoT_reason','basic_few_shot_CoT_reason','bisection_CoT']

def sha(data): return hashlib.sha256(data).hexdigest()
def get(url):
    with urllib.request.urlopen(url, timeout=40) as response:
        return response.read()
def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def fetch_sources():
    SOURCES.mkdir(parents=True, exist_ok=True)
    for repo in REPOS:
        metadata=json.loads(get('https://huggingface.co/api/models/'+repo))
        revision=metadata['sha']
        stem=repo.split('/')[-1]
        source={'repo':repo,'revision':revision,'retrieved_at':datetime.now(timezone.utc).isoformat(),
                'metadata_url':'https://huggingface.co/api/models/'+repo,'files':{}}
        for filename in ['generation_config.json','config.json','tokenizer_config.json','README.md']:
            url=f'https://huggingface.co/{repo}/raw/{revision}/{filename}'
            raw=get(url)
            obj=None if filename.endswith('.md') else json.loads(raw)
            info={'url':url,'sha256':sha(raw),'bytes':len(raw)}
            if filename=='generation_config.json':
                dump(SOURCES/f'{stem}.generation_config.json',obj)
                source['generation_config']=obj
            elif filename=='config.json':
                subset={k:obj.get(k) for k in ['model_type','architectures','max_position_embeddings','rope_scaling','transformers_version','torch_dtype','dtype']}
                dump(SOURCES/f'{stem}.model_config_subset.json',subset)
                source['model_config_subset']=subset
            elif filename=='tokenizer_config.json':
                template=obj.get('chat_template','')
                (SOURCES/f'{stem}.chat_template.jinja').write_text(template,encoding='utf-8')
                source['chat_template']={'mentions_enable_thinking':'enable_thinking' in template,
                                         'mentions_think_marker':'<think>' in template,
                                         'sha256':sha(template.encode())}
            else:
                text=raw.decode()
                # Short reproducible evidence excerpts, avoiding whole-card duplication.
                lines=text.splitlines()
                anchors=['For thinking mode, use','For non-thinking mode, we suggest','Adequate Output Length',
                         'max_new_tokens=512','max_new_tokens=32768']
                excerpts=[{'line_1based':i+1,'text':line} for i,line in enumerate(lines) if any(a in line for a in anchors)]
                dump(SOURCES/f'{stem}.model_card_excerpts.json',excerpts)
                source['model_card_url']=f'https://huggingface.co/{repo}/blob/{revision}/README.md'
            source['files'][filename]=info
        dump(SOURCES/f'{stem}.source_manifest.json',source)

def resolve(node, consts):
    try: return ast.literal_eval(node)
    except Exception: pass
    if isinstance(node,ast.Name) and node.id in consts: return consts[node.id]
    if isinstance(node,ast.BinOp) and isinstance(node.op,(ast.Add,ast.Sub)):
        left,right=resolve(node.left,consts),resolve(node.right,consts)
        if isinstance(left,(int,float)) and isinstance(right,(int,float)):
            return left+right if isinstance(node.op,ast.Add) else left-right
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in ('min','max'):
        args=[resolve(x,consts) for x in node.args]
        if all(isinstance(x,(int,float)) for x in args): return (min if node.func.id=='min' else max)(args)
    return {'expression':ast.unparse(node)}

def inspect_script(path,sources):
    raw=path.read_bytes(); text=raw.decode(); tree=ast.parse(text)
    consts={}
    for node in tree.body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name): consts[target.id]=resolve(node.value,consts)
    parents={child:parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    def function(n):
        while n in parents:
            n=parents[n]
            if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)): return n.name
        return '<module>'
    method=lambda n:n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id if isinstance(n.func,ast.Name) else ast.unparse(n.func)
    calls=[]; seed_calls=[]; templates=[]; truncations=[]
    for node in sorted((n for n in ast.walk(tree) if isinstance(n,ast.Call)),key=lambda n:n.lineno):
        name=method(node); kwargs={k.arg:resolve(k.value,consts) for k in node.keywords if k.arg}
        record={'line':node.lineno,'function':function(node),'method':name,'kwargs':kwargs}
        if name=='generate': calls.append(record)
        if name=='apply_chat_template': templates.append(record)
        if name in ('manual_seed','manual_seed_all','seed','set_seed'): seed_calls.append(record)
        if 'truncation' in kwargs or 'max_length' in kwargs: truncations.append(record)
    repo=next(repo for repo in REPOS if repo.split('/')[-1].split('-Instruct')[0] in path.name)
    official=sources[repo]['generation_config']
    for call in calls:
        supplied=call['kwargs']
        call['conditional_effective_if_unmodified_official_config']={k:supplied.get(k,official.get(k,'library default/not in file')) for k in
            ['do_sample','temperature','top_p','top_k','min_p','repetition_penalty','max_new_tokens']}
        call['explicit_overrides']=list(supplied)
        call['scope_note']='Conditional only: historical /data model folder generation_config.json and package versions were not saved.'
    constants={k:v for k,v in consts.items() if k in ['MODEL_PATH','DO_SAMPLE','TEMPERATURE','TOP_P','TOP_K','FALLBACK_ABBR','COARSE_MAX_LEN','FINE_MAX_LEN'] or k.startswith('MAX_') or k=='REASON_MAX_TOKENS'}
    comments=[{'line':i+1,'text':line.strip()} for i,line in enumerate(text.splitlines()) if line.lstrip().startswith('#') and any(k in line for k in ['贪婪','greedy','do_sample','enable_thinking'])]
    return {'path':str(path.relative_to(DATA)),'sha256':sha(raw),'repo':repo,'constants':constants,
            'generation_calls':calls,'chat_template_calls':templates,'input_tokenization_calls':truncations,
            'seed_calls':seed_calls,'comment_only_claims':comments,
            'runtime_generation_config_mutation':bool(re.search(r'model\.generation_config\s*[.=]',text)),
            'loads_pinned_revision':any(k.arg=='revision' for n in ast.walk(tree) if isinstance(n,ast.Call) for k in n.keywords)}

def binary_output_evidence():
    path=DATA/'bisection_CoT/Qwen3-8B_few_shot.py'
    selected=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('_visible_only','parse_yesno')]
    namespace={'re':re}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace)
    examples=[]
    for name,raw in [
        ('normal_complete_pair','<think>The answer could be yes, but that is ruled out.</think>\nNo'),
        ('closing_marker_only','The answer could be yes, but that is ruled out.\n</think>\nNo'),
        ('truncated_unclosed_thinking','<think>The answer could be yes, but further examination is needed')]:
        examples.append({'type':name,'raw':raw,'source_parser_result':namespace['parse_yesno'](raw),
                         'expected_final_answer':'no' if name!='truncated_unclosed_thinking' else None,
                         'synthetic_counterexample_not_an_actual_model_prediction':True})
    results=[]
    for path in sorted((DATA/'bisection_CoT/results').rglob('predictions.csv')):
        if not any(model in str(path.parent) for model in ('Qwen3-8B','Qwen3-14B')):continue
        rows=list(csv.DictReader(path.open()))
        settings=json.loads((path.parent/'settings.json').read_text())
        reports=list(csv.DictReader((path.parent/'report.csv').open()))
        avg=next((r for r in reports if r.get('label')=='weighted avg'),None)
        results.append({'path':str(path.relative_to(DATA)),'n':len(rows),'columns':list(rows[0]),
            'raw_output_all_already_parsed_abbreviations':all(r['raw_output'] in set(['MLKnow','MLInter','MCC','DMC','DFEC','Weight Conf','Layer Conf','HP Conf','MLDepend','GC','CDT','ML','UMC','Data Conf','PODT','DS Conf','Abs','BE','MCM','MCR','PQ','MLRelia','Proto']) for r in rows),
            'raw_output_max_chars':max(len(r['raw_output']) for r in rows),
            'raw_output_with_think_markers':sum('<think>' in r['raw_output'] or '</think>' in r['raw_output'] for r in rows),
            'reason_equal_open_think_marker':sum(r.get('reason')=='<think>' for r in rows),
            'max_new_tokens_settings':settings.get('max_new_tokens'),
            'temperature_settings':settings.get('temperature'),'top_p_settings':settings.get('top_p'),
            'weighted_report':avg,'extra_raw_logs':[x.name for x in path.parent.iterdir() if x.name not in ['predictions.csv','settings.json','report.csv','confusion_matrix.csv']]})
    return {'parser_source':str((DATA/'bisection_CoT/Qwen3-8B_few_shot.py').relative_to(DATA)),
            'parser_lines':[282,318],'reason_call_and_storage_lines':[421,433],
            'proof':examples,'stored_result_runs':results,'stored_classifications':sum(r['n'] for r in results),
            'stored_reason_equal_open_think_marker':sum(r['reason_equal_open_think_marker'] for r in results),
            'pinned_official_template_fact':'For enable_thinking=True, prompt ends with assistant newline; opening <think> is generated, not prefilled. Complete paired output is parsed correctly by the regex.',
            'impact_boundary':'Post-hoc reasons are demonstrably unusable for 3139/3144 rows. Classification raw Y/N/arbitration completions and finish reasons were not saved, so affected classification count and corrected F1 cannot be recovered offline.'}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fetch-official',action='store_true');args=parser.parse_args()
    if args.fetch_official: fetch_sources()
    sources={}
    for path in SOURCES.glob('*.source_manifest.json'):
        obj=json.loads(path.read_text());sources[obj['repo']]=obj
    if set(sources)!=set(REPOS): raise ValueError('Need all three source manifests; run --fetch-official first.')
    scripts=[inspect_script(p,sources) for d in DIRS for p in sorted((DATA/d).glob('Qwen*.py'))]
    result={'audit_date':'2026-10-16','scope':'Read-only current source audit; no experiment reruns and no model weight downloads.',
            'historical_limitation':'Current scripts, historical result settings, official current pinned configs are distinct evidence. No original /data model config snapshot or exact package lock was recovered.',
            'script_count':len(scripts),'official_sources':sources,'scripts':scripts}
    result['binary_cot_output_evidence']=binary_output_evidence()
    result['findings']=[
        {'id':'Q01','status':'paper_wording_incorrect','summary':'No Qwen source sets temperature=0. Main Qwen3 profiles match the official mode-specific sampling guidance; Qwen2.5 vanilla inherits its shipped config. All three official configs have do_sample=true.',
         'historical_caveat':'An omitted do_sample inherits loaded model configuration; absence alone does not prove greedy decoding. The original server /data model folders were not recovered.'},
        {'id':'Q02','status':'confirmed_current_source_inconsistency','paths':['bisection_zero_shot/Qwen3-14B_zero_shot.py:188-194','bisection_zero_shot/Qwen3-8B_zero_shot.py:188-194'],
         'summary':'14B binary zero-shot explicitly forces do_sample=False; 8B uses mode-specific sampling parameters and no override. It is a decoding-policy difference, not merely a comment typo.',
         'action':'Document the actual difference if preserving historical experiments. If claiming controlled strategy/model comparisons, rerun the affected condition under a predeclared matched decoding policy.'},
        {'id':'Q03','status':'confirmed_logging_config_mismatch','paths':[s['path'] for s in scripts if s['path'].startswith(('tree_zero_shot/','tree_few_shot/'))],
         'summary':'All six tree scripts declare DO_SAMPLE=False and save it to settings, but the generate do_sample=DO_SAMPLE argument is commented out. Six stored Qwen tree settings files say false, without proving the runtime used greedy decoding.',
         'action':'Correct logging to serialize effective model.generation_config merged with call overrides. Preserve original settings as historical artifacts; do not silently replace them.'},
        {'id':'Q04','status':'comparison_confound','paths':['basic_zero_shot/Qwen3-8B_zero_shot_re.py:140-154','basic_zero_shot_CoT/Qwen3-8B_zero_shot_CoT.py:135-150','basic_zero_shot_CoT/Qwen3-8B-CoT-new.py:151-166'],
         'summary':'Qwen3 nonthinking versus thinking profiles also change temperature .7 to .6, top_p .8 to .95, and max output 20 to 32768. Manual-rule CoT scripts also enable native thinking, so comparison to nonthinking baseline does not isolate the hand-written rules.',
         'action':'Describe recommended-profile comparisons rather than an isolated thinking causal effect; compare native-thinking vanilla against native-thinking manual rules to isolate the latter.'},
        {'id':'Q05','status':'confirmed_non_native_flag_and_parameter_deviation','paths':['basic_zero_shot_CoT/Qwen2.5-Coder-7B-CoT-new.py:151-166','basic_few_shot_CoT/Qwen2.5-Coder-7B-CoT-new.py:362-378','basic_zero_shot_CoT_reason/Qwen2.5-Coder-7B-CoT.py:154-216','basic_few_shot_CoT_reason/Qwen2.5-Coder-7B-CoT.py:154-216'],
         'summary':'Official Qwen2.5-Coder template contains neither enable_thinking nor think markers. Passing enable_thinking=True is inert for that template, not a native reasoning switch. Four classification calls instead explicitly use .6/.95/20, deviating from its shipped .7/.8/20 default.',
         'action':'Keep prompt-based reasoning distinct from native thinking. Custom decoding is allowable, but cannot be described as uniformly following shipped model defaults.'},
        {'id':'Q06','status':'auxiliary_reason_profile_deviation','paths':[s['path'] for s in scripts if s['repo'].startswith('Qwen/Qwen3') and '_CoT_reason/' in s['path']],
         'summary':'Four Qwen3 auxiliary generate_reason calls disable native thinking but retain .6/.95/20, instead of nonthinking .7/.8/20 guidance. They run after label selection and do not feed back into the classifier.',
         'action':'Disclose as auxiliary explanation settings; do not infer a classification-score effect.'},
        {'id':'Q07','status':'confirmed_auxiliary_output_failure_with_unquantified_classification_risk','paths':['bisection_CoT/Qwen3-8B_few_shot.py:35-41','bisection_CoT/Qwen3-8B_few_shot.py:282-318','bisection_CoT/Qwen3-8B_few_shot.py:341-362','bisection_CoT/Qwen3-8B_few_shot.py:421-433','bisection_CoT/Qwen3-14B_few_shot.py:282-318'],
         'summary':'Binary CoT classification stages allow 2048 output tokens and auxiliary explanations only 96, while keeping thinking enabled. 3139 of 3144 saved auxiliary reasons equal <think>. Complete paired thinking is removed correctly, but malformed/unfinished blocks are not rejected and can be scanned as final decisions.',
         'action':'Do not claim saved reasons are usable explanations. For reliable classification validation, save full raw output/finish reason and reject unfinished thinking; a targeted corrected evaluation is needed if final-answer parsing is central, because old raw classifier completions were discarded.'},
        {'id':'Q08','status':'reproducibility_gap','summary':'Across all 38 original Qwen sources: 0 runtime seed-setting calls and 0 pinned model revision arguments; historical settings omit the resolved full generation config. Stochastic single runs are not exactly reproducible from current sources alone.',
         'action':'Report per-condition effective decoding and versions, set and record seeds, and use repeated stochastic runs or paired uncertainty estimates for new claims.'},
        {'id':'Q09','status':'input_budget_caveat','summary':'Original sources truncate inputs: common vanilla/CoT cap4096, vanilla few-shot cap65536 or16384, binary few-shot cap65536, hierarchy coarse cap2048/fine cap8192 with left truncation. A cap is not proof actual truncation happened.',
         'action':'Audit actual prompt token lengths before claiming no loss; do not equate different input limits with confirmed lost targets.'}
    ]
    dump(BASE/'qwen_official_audit.json',result)
    print(json.dumps({'scripts':len(scripts),'generation_calls':sum(len(x['generation_calls']) for x in scripts),'seed_calls':sum(len(x['seed_calls']) for x in scripts)}))

if __name__=='__main__':main()
