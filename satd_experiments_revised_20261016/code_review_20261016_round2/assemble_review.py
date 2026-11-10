"""Curate independent reviews into a file-by-file manual action table."""
import ast
import copy
import csv
import hashlib
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
PREV=ROOT/'paper_generation_audit_20261016'

def read(name,base=HERE):return json.loads((base/name).read_text())
def write(name,data):(HERE/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def items(name):
    d=read(name)
    return d['findings'] if isinstance(d,dict) else d

raw=[]
for name in ['inference_findings.json','data_findings.json','prompt_findings.json','pilot_findings.json']:
    for f in items(name):
        f=copy.deepcopy(f);f['review_source']=name;raw.append(f)
by_id={f['id']:f for f in raw}
aliases={
    'PROMPT-03':'INF-01','DATA-02':'INF-02','DATA-09':'INF-03',
    'DATA-18':'INF-05','DATA-17':'INF-08',
    'DATA-10':'INF-14','PROMPT-10':'INF-14','PROMPT-11':'INF-14',
    'DATA-19':'PROMPT-08',
}
groups=[]
for f in raw:
    if f['id'] in aliases:continue
    f['source_ids']=[f['id']]
    f['supporting_reviews']=[]
    groups.append(f)
lookup={f['id']:f for f in groups}
for old,new in aliases.items():
    extra=by_id[old];target=lookup[new]
    target['source_ids'].append(old)
    target['locations'].extend(extra['locations'])
    target['supporting_reviews'].append(extra)

# Preserve important distinctions instead of treating every design choice as a bug.
lookup['INF-05'].update(severity='P2',recommendation='保留回退前标签、parse_status、fallback_reason及最终策略预测；披露既有回退规则。新实验预先规定计分策略，可另报失败计错指标、原回退指标和回退贡献；不覆盖旧表。')
lookup['INF-07']['status']='design-choice'
lookup['INF-03'].update(
    title='多词缩写的兼容性归一化分支不一致',
    impact='大小写/标点兼容分支对单词与多词缩写行为不同；若设计为严格缩写协议，拒绝额外格式可以保留，但应统一规则并记录格式失败。当前先导实验300条均为可解析输出。',
    recommendation='先明确严格或兼容协议。若兼容，以同一canonical函数构建字典键并解析输出；若严格，使用完整合法缩写精确校验并删掉不一致的宽松兜底。')
lookup['INF-14'].update(
    title='Yes/No解析接受普通单词或优先搜索yes',
    description='前缀匹配会把Not sure判成No、Yesterday判成Yes；Qwen原生thinking解析器还在全文先找yes，非标准否定解释可能翻转为Yes。',
    recommendation='先提取受支持的最终答案字段，再严格匹配完整Yes/No（或预先定义的Y/N）。不在思考/解释全文优先搜索yes；不确定、双答案及普通前缀词返回invalid。')
lookup['INF-04']['recommendation']='仅对预期原生thinking的条件检查思考结束标记和最终答案段；未闭合时保存独立失败，不扫描思考中的候选。非thinking模型不应被要求输出</think>。保留完整raw与生成停止状态。'
lookup['INF-08']['description']='四个Qwen binary脚本将最终raw也替换成缩写；其余27个binary脚本保留最终步骤文本或候选，但逐标签Yes/No原文未持久化。'
lookup['INF-08']['impact']='无法完整重放23个投票、候选裁决和解析过程。不能据此判定已有标签F1无效，但历史截断/格式失败频率无法可靠恢复。'
lookup['INF-11']['locations']=[l for l in lookup['INF-11']['locations'] if l['file']!='basic_few_shot/CodeLlama-7b_few_shot.py']
lookup['INF-02']['impact']+=' 当前保存的Vanilla原文中未找到“单独完整全称却存错类”实例；可复现逻辑问题不等于已量化历史分数影响。'
lookup['DATA-04']['replay_evidence_file']='data_observed_tree_parse_failures.json'
lookup['DATA-04']['impact']+=' 当前parser重放保存的tree raw可复现UNK而日志为0；历史源文件hash缺失，不能将重放率当作确定的历史运行失败率。'

frozen=['pilot_context_20261016/reference/frozen_prompt.py','pilot_context_20261016/reference_thinking_stopped/frozen_prompt.py']
for rel in frozen:
    tree=ast.parse((ROOT/rel).read_text())
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name=='simple_extract':lookup['INF-03']['locations'].append({'file':rel,'line':node.lineno,'end_line':node.end_lineno})
        if isinstance(node,ast.FunctionDef) and node.name=='build_prompt':lookup['PROMPT-12']['locations'].append({'file':rel,'line':node.lineno,'end_line':node.end_lineno})

def add(id,title,status,severity,description,impact,recommendation,locations,evidence):
    groups.append(dict(id=id,title=title,status=status,severity=severity,description=description,impact=impact,recommendation=recommendation,locations=locations,evidence=evidence,source_ids=[id],supporting_reviews=[],review_source='root_additional_review'))

add('ROOT-01','Qwen2.5的enable_thinking=True不启用原生思考','confirmed','P2',
    '四个Qwen2.5-Coder CoT脚本传enable_thinking=True，但该官方模板不读取该变量；提示式CoT仍可以使用。',
    '模式开关不生效，不能把该条件描述为Qwen3式原生thinking。显式.6/.95属于自定义采样配置，本身合法。',
    '移除对该flag生效的依赖；将prompt CoT与原生thinking分别记录。若保留自定义参数，明确写入有效配置。',
    [{'file':f,'line':line,'end_line':line+5} for f,line in [
        ('basic_few_shot_CoT/Qwen2.5-Coder-7B-CoT-new.py',362),
        ('basic_few_shot_CoT_reason/Qwen2.5-Coder-7B-CoT.py',154),
        ('basic_zero_shot_CoT/Qwen2.5-Coder-7B-CoT-new.py',151),
        ('basic_zero_shot_CoT_reason/Qwen2.5-Coder-7B-CoT.py',154)]],
    {'source':'../paper_generation_audit_20261016/qwen_official_audit.json Q05','official_template_contains_enable_thinking':False})

gpt=read('gptoss_official_audit.json',PREV)
gpt_locs=[]
for script in gpt['scripts']:
    for call in script.get('generation_calls',[]):gpt_locs.append({'file':script['script'],'line':call['line'],'end_line':call.get('end_line',call['line'])})
add('ROOT-02','GPT-oss原生reasoning effort没有可核实的显式控制','current-risk','P2',
    '部分普通消息正文写Reasoning:low/high，pipeline调用没有明确配置原生Harmony级别，也未保存渲染后系统元数据。',
    '不能由正文文字证明全部运行使用high；实际历史模式仍需原模板和渲染prompt确认。标准pipeline聊天接口本身合法。',
    '使用对应版本模板或官方Harmony渲染器明确设置原生级别，保存渲染后的prompt头和模型/库版本；移除仅凭正文宣称high的实验记录。',
    gpt_locs,{'source':'../paper_generation_audit_20261016/gptoss_official_audit.json','official':'https://developers.openai.com/cookbook/articles/gpt-oss/run-transformers'})

reason_locs=[]
for f in ['bisection_CoT/Qwen3-8B_few_shot.py','bisection_CoT/Qwen3-14B_few_shot.py','test1.py','test3.py']:
    for i,line in enumerate((ROOT/f).read_text().splitlines(),1):
        if ('MAX_NEW_TOKENS_REASON' in line or 'MAX_REASON_TOKENS' in line or ('reason_out' in line and '_gen(' in line)):
            reason_locs.append({'file':f,'line':i,'end_line':i})
add('ROOT-03','Qwen binary事后解释大量只保存了<think>','confirmed','P2',
    '事后解释仍走开启thinking的生成函数，短解释预算没有保证最终解释生成。8个binary-CoT run的3144条解释中3139条仅为<think>，另2个native binary run也有未闭合标记。',
    '这些字段不能作为有效解释。解释在标签确定后生成，不改变分类预测；不能据此判定F1无效。合理限制长度本身不是bug。',
    '为事后短解释使用独立的非thinking调用，或实现限制思考后继续生成答案；校验非空、无未闭合思考标记并记录停止状态。分类与解释预算分别保存。',
    reason_locs,{'source':'../paper_generation_audit_20261016/qwen_official_audit.json binary_cot_output_evidence','coT_exact_think_reasons':3139,'coT_reason_total':3144})

add('ROOT-04','旧ICL文件末尾语法错误','confirmed','P1',
    'to_csv调用后直接附有多余de，整文件无法通过Python解析。',
    '该旧原型当前不能执行；不说明主文其他方法脚本失败。',
    '若需复用，先移除多余字符并做语法检查；若仅归档，将文件移出可执行入口，保留历史来源说明。',
    [{'file':'ICL_Ob_qwen-8b.py','line':517,'end_line':517}],{'test':'ast.parse raises SyntaxError at line517; no original script executed'})
add('ROOT-05','旧ICL新脚本在定义前使用缺失名称','confirmed','P1',
    '第33行调用add_context(df, INPUT)，文件未定义或导入add_context，也未定义INPUT。',
    '在原服务器数据读取成功后仍会NameError，后续生成和评估无法执行；该文件属于旧原型。',
    '若复用，显式定义或导入add_context，并把INPUT改为有校验的配置参数；先做无模型的数据构建检查。',
    [{'file':'ICL_Ob_qwen-8b_new.py','line':33,'end_line':33}],{'test':'AST name-binding and import inspection; no original script executed'})

# Sort actionable, directly evidenced items first, and keep archived prototypes later.
rank={'P1':0,'P2':1,'P3':2}
status_rank={'confirmed':0,'current-risk':1,'needs-history':2,'design-choice':3}
preferred=['PROMPT-01','INF-01','INF-02','DATA-01','PROMPT-07','PROMPT-02','INF-06','INF-14','DATA-03','DATA-04']
def ordering(f):
    if f['id'] in preferred:return (0,preferred.index(f['id']),0)
    only_legacy=(all(l['file'].startswith('ICL_') or l['file']=='original_COT.py' or l['file'].startswith('dataset/') for l in f['locations'])
                 and any(l['file'].startswith('ICL_') or l['file']=='original_COT.py' for l in f['locations']))
    return (2 if only_legacy else 1,rank.get(f['severity'],2),status_rank[f['status']],f['id'])
groups.sort(key=ordering)
labels={'confirmed':'已确认','current-risk':'风险','needs-history':'待核实','design-choice':'设计选择'}
inventory=read('all_scripts_inventory.json',PREV)
inv={r['script']:r for r in inventory}
pilot_cov=read('pilot_coverage.json')
code_hashes={r['script']:r['sha256'] for r in inventory}|{r['file']:r['sha256'] for r in pilot_cov}
hash_changes=[f for f,h in code_hashes.items() if hashlib.sha256((ROOT/f).read_bytes()).hexdigest()!=h]
assert not hash_changes,hash_changes

all_rows=[]
for idx,f in enumerate(groups,1):
    f['issue_id']=f'R{idx:02d}';f['status_label']=labels[f['status']]
    unique={}
    for l in f['locations']:
        path=ROOT/l['file']; assert path.is_file(),l
        count=len(path.read_text(encoding='utf-8-sig').splitlines())
        assert 1<=l['line']<=count,(f['id'],l,count)
        key=(l['file'],l['line'],l.get('end_line',l['line']))
        unique[key]=l
    f['locations']=sorted(unique.values(),key=lambda l:(l['file'],l['line']))
    per_file=defaultdict(list)
    for l in f['locations']:per_file[l['file']].append(l)
    f['affected_files']=len(per_file)
    for file,locs in per_file.items():
        line_display='; '.join(str(l['line']) if l.get('end_line',l['line'])==l['line'] else f"{l['line']}-{l['end_line']}" for l in locs)
        recommendation=f['recommendation']
        if '/reference' in file:
            recommendation='保留已用于历史run的冻结文件，勿原地修改；在新runner/新解析模块中修复并使用新run目录。'+recommendation
        scope='数据集' if file.endswith('.csv') else ('先导实验' if file.startswith('pilot_context_') else inv.get(file,{}).get('role','旧原型/辅助工具'))
        all_rows.append(dict(issue_id=f['issue_id'],source_id=f['id'],severity=f['severity'],status=f['status'],status_label=f['status_label'],file=file,line=locs[0]['line'],line_display=line_display,title=f['title'],description=f['description'],impact=f['impact'],recommendation=recommendation,manual_status='待核实' if f['status'] in ('current-risk','needs-history') else '待决定' if f['status']=='design-choice' else '待处理',scope=scope))

role_labels={
 'root_experimental_or_legacy_candidate_not_run_linked':'旧原型（未直接关联论文run）',
 'method_family_result_producer_candidate':'论文方法族候选脚本',
 'arbiter_experiment_not_in_active_manuscript_tables':'辅助仲裁实验',
 'data_preprocessing_utility':'预处理工具','model_download_utility_auth_values_redacted':'模型下载工具',
 'random_baseline_result_producer':'随机基线'}
coverage=[]
for file,h in sorted(code_hashes.items()):
    if file in inv:
        role=role_labels.get(inv[file]['role'],inv[file]['role'])
        areas='prompt/示例/标签映射；生成/分词/输出解析；数据/gold/指标/保存（无相关逻辑者按用途标不适用）'
    else:
        role='先导实验冻结参考' if '/reference' in file else '先导实验测试' if '/test_' in file else '先导实验脚本'
        areas='选样与来源核验；配对prompt；MLX配置/恢复/解析；指标与无模型测试'
    issue_ids=sorted({r['issue_id'] for r in all_rows if r['file']==file})
    coverage.append(dict(file=file,role=role,reviewed_areas=areas,issue_ids=issue_ids,sha256=h))

non_findings=read('data_non_findings.json')
payload=dict(date='2026-10-16',code_files_reviewed=len(coverage),original_files=114,pilot_files=10,
    finding_count=len(groups),status_counts=dict(Counter(f['status_label'] for f in groups)),
    priority_counts=dict(Counter(f['severity'] for f in groups)),file_issue_rows=len(all_rows),
    source_hash_changes=hash_changes,groups=groups,locations=all_rows,coverage=coverage,
    verified_checks=non_findings,model_inference_performed=False,original_code_modified=False,
    scope_note='确认指当前源码/当前分词器反例/保存产物行为，不等于已证明该源码版本导致历史分数变化。没有运行原始模型脚本。')
write('review_summary.json',payload)
with (HERE/'manual_fix_checklist.csv').open('w',newline='',encoding='utf-8-sig') as f:
    fields=['问题编号','优先级','判断','文件','行号','问题','修改建议','影响与边界','处理状态','修改备注']
    writer=csv.writer(f);writer.writerow(fields)
    for r in all_rows:writer.writerow([r['issue_id'],r['severity'],r['status_label'],r['file'],r['line_display'],r['title'],r['recommendation'],r['impact'],r['manual_status'],''])

E=html.escape
def link(file,line):return '<a href="'+E(str(ROOT/file)+':'+str(line),quote=True)+'">'+E(file)+':'+str(line)+'</a>'
parts=['''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>实验代码审计与修改清单</title><style>
body{font:15px/1.65 system-ui,sans-serif;margin:30px auto;max-width:1600px;padding:0 24px;color:#233346;background:#f7f9fc}h1{font-size:28px}h2{margin-top:32px}a{color:#13629d}table{border-collapse:collapse;width:100%;background:white}th,td{border:1px solid #d9e0e9;padding:10px;text-align:left;vertical-align:top;overflow-wrap:anywhere}th{background:#243e59;color:white;position:sticky;top:72px}input,select{font:inherit;padding:8px;border:1px solid #a9bbcc;border-radius:4px}#controls{position:sticky;top:0;background:#f7f9fc;padding:12px 0;z-index:2;display:flex;gap:10px;flex-wrap:wrap}input{min-width:370px}.note{padding:15px;background:#eaf1fa}.p1{color:#a53226;font-weight:600}.muted{color:#607087}details{margin:7px 0}summary{cursor:pointer}pre{white-space:pre-wrap;word-break:break-word;font-size:12px;background:#edf2f7;padding:12px}.count{margin:15px 0}button{font:inherit;padding:8px 14px;cursor:pointer}</style><h1>实验代码审计与修改清单</h1>''']
parts.append('<p>2026-10-16 · '+str(len(coverage))+'个脚本（114原始＋10先导） · '+str(len(groups))+'类事项 · '+str(len(all_rows))+'个文件事项</p>')
parts.append('<div class="note"><b>先修确认问题，再核实条件风险。</b>“已确认”表示当前源码、分词器反例或保存产物已证实该行为；不代表全部历史分数受影响。合法非零温度、合理输出上限和明确的回退策略不直接判为程序错误。原脚本未修改、未运行模型推理。冻结参考和旧结果应保留，新实现使用新run目录。</div>')
parts.append('<p>从保存预测重算的224个论文P/R/F1三元组全部吻合；158个结果文件均有393行，评论/gold顺序无错配；108份常规标签映射覆盖相同23类。主方法固定示例未检出完整测试注释重叠；旧ICL问题已单列。</p>')
parts.append('<div id="controls"><input id="q" placeholder="搜索文件、编号、问题或修改建议"><select id="status"><option value="">全部判断</option>'+''.join('<option>'+v+'</option>' for v in labels.values())+'</select><select id="priority"><option value="">全部优先级</option><option>P1</option><option>P2</option><option>P3</option></select><button id="reset">清空筛选</button></div><p class="count" id="count"></p><table><thead><tr><th>编号/优先级</th><th>判断</th><th>文件与位置</th><th>问题</th><th>修改建议</th><th>影响与边界</th></tr></thead><tbody id="body">')
for r in all_rows:
    search=' '.join(str(v) for v in r.values())
    parts.append('<tr data-text="'+E(search.lower(),quote=True)+'" data-status="'+E(r['status_label'],quote=True)+'" data-priority="'+r['severity']+'"><td class="'+('p1' if r['severity']=='P1' else '')+'">'+r['issue_id']+'<br>'+r['severity']+'</td><td>'+r['status_label']+'</td><td>'+link(r['file'],r['line'])+'<br><span class="muted">'+E(r['line_display'])+'</span></td><td>'+E(r['title'])+'</td><td>'+E(r['recommendation'])+'</td><td>'+E(r['impact'])+'</td></tr>')
parts.append('</tbody></table><h2>各类事项的完整依据</h2>')
for f in groups:
    evidence=f.get('evidence',f.get('evidence/test',{}))
    parts.append('<details id="'+f['issue_id']+'"><summary>'+f['issue_id']+' · '+E(f['title'])+' · '+f['status_label']+' · '+str(f['affected_files'])+'个文件</summary><p>'+E(f['description'])+'</p><pre>'+E(json.dumps(evidence,ensure_ascii=False,indent=2))+'</pre><p>交叉审查记录：'+E(', '.join(f['source_ids']))+'</p></details>')
parts.append('<h2>交付表格</h2><p><a href="outputs/01a1439c-689e-72b1-82c0-2c50695a0417/实验代码审计与修改清单.xlsx">Excel修改清单</a> · <a href="manual_fix_checklist.csv">CSV修改清单</a> · <a href="review_summary.json">完整审计JSON与124文件覆盖记录</a></p>')
parts.append('''<script>
const rows=[...document.querySelectorAll('#body tr')];
function apply(){let n=0;const q=document.querySelector('#q').value.toLowerCase(),s=document.querySelector('#status').value,p=document.querySelector('#priority').value;for(const row of rows){const ok=(!q||row.dataset.text.includes(q))&&(!s||row.dataset.status===s)&&(!p||row.dataset.priority===p);row.hidden=!ok;n+=ok;}document.querySelector('#count').textContent=`显示 ${n} / ${rows.length} 个文件事项`;}
for(const id of ['q','status','priority'])document.querySelector('#'+id).addEventListener('input',apply);
document.querySelector('#reset').onclick=()=>{for(const id of ['q','status','priority'])document.querySelector('#'+id).value='';apply();};apply();
</script></html>''')
(HERE/'manual_fix_checklist.html').write_text('\n'.join(parts))
print(json.dumps({k:payload[k] for k in ['code_files_reviewed','finding_count','status_counts','priority_counts','file_issue_rows','source_hash_changes']},ensure_ascii=False))
