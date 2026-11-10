"""Assemble the read-only generation audit; never import original experiment scripts."""
import csv
import hashlib
import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PAPER = Path('/Users/yizho/Desktop/Fine_grained_ML_SATD_Classification_with_Large_Language_Models_An_Empirical_Study')

def read(name):
    return json.loads((HERE / name).read_text())

def save(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

inventory = read('all_scripts_inventory.json')
observations = read('local_observations.json')
summary = read('inventory_summary.json')
qwen = read('qwen_official_audit.json')
nonqwen = read('nonqwen_official_audit.json')
gpt = read('gptoss_official_audit.json')

# Include the arbiter experiment, whose model is identified by MODEL_PATH rather
# than the inventory's filename-based family label. It is outside active tables.
existing = {r['script'] for r in gpt['scripts']}
for row in inventory:
    model_path = row.get('configuration_constants', {}).get('MODEL_PATH', {}).get('value', '')
    if isinstance(model_path, str) and 'gpt-oss-20b' in model_path and row['script'] not in existing:
        gpt['scripts'].append({
            'script': row['script'], 'sha256': row['sha256'],
            'backend': 'transformers.pipeline', 'generation_calls': row.get('generation_calls', []),
            'model_configuration_constants': row.get('configuration_constants', {}),
            'explicit_reasoning_effort_parameter': False,
            'scope_role': row.get('role'),
            'sampling_assessment': 'No explicit sampling overrides; historical effective pipeline/checkpoint configuration was not saved.',
            'reasoning_assessment': 'No explicit native reasoning-effort control; arbiter experiment is outside active manuscript tables.'
        })
gpt['checked_original_scripts'] = len(gpt['scripts'])
save('gptoss_official_audit.json', gpt)

sources = [
    ('Qwen3-8B 官方模型卡：分模式采样建议', 'https://huggingface.co/Qwen/Qwen3-8B'),
    ('Qwen3-14B 官方模型卡', 'https://huggingface.co/Qwen/Qwen3-14B'),
    ('Qwen2.5-Coder 官方生成配置', 'https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct/blob/c03e6d358207e414f1eca0bb1891e29f1db0e242/generation_config.json'),
    ('Qwen 官方 thinking budget：限制思考后继续生成答案', 'https://qwen.readthedocs.io/en/v3.0/getting_started/thinking_budget.html'),
    ('Transformers 生成配置：max_new_tokens 与采样', 'https://huggingface.co/docs/transformers/main_classes/text_generation'),
    ('Transformers 4.51.3 聊天模板：避免重复特殊 token', 'https://huggingface.co/docs/transformers/v4.51.3/en/chat_templating'),
    ('CodeLlama 官方参考示例：温度 0.2 / top-p 0.95', 'https://github.com/meta-llama/codellama/blob/main/example_instructions.py'),
    ('GPT-oss 官方 Transformers 指南', 'https://developers.openai.com/cookbook/articles/gpt-oss/run-transformers'),
    ('OpenAI Harmony：推理级别与消息格式', 'https://developers.openai.com/cookbook/articles/openai-harmony'),
]
for model in nonqwen['official_models']:
    sources.append((model['repo'] + ' 官方模型卡', model['model_card_url']))

models = [
    ['Qwen3-8B / Qwen3-14B', '主要 non-thinking：T=.7, p=.8, k=20；thinking：T=.6, p=.95, k=20。', '与官方分模式建议一致；并非温度 0。14B binary zero-shot 有显式 greedy 例外。'],
    ['Qwen2.5-Coder-7B-Instruct', '省略参数的调用依赖配置；部分 CoT 显式 .6/.95/20。', '官方随附配置是采样 .7/.8/20。自定义参数可以使用，但不能称为统一默认；enable_thinking 对该官方模板不生效。'],
    ['CodeLlama-7B-Instruct', '10 个脚本均显式 do_sample=True, T=.2, p=.95。', '与官方参考示例一致（示例不是强制标准）。另有重复 BOS 格式问题。'],
    ['CodeGemma-1.1-7B-it', '多数调用未显式覆盖采样参数。', '未找到统一强制温度。官方完整配置访问受限，不能确认历史有效默认值。'],
    ['Gemma-2-9B-it', '多数调用未显式覆盖采样参数。', '同上；层次零样本当前代码实际绕过聊天模板。'],
    ['Gemma-3-12B-it', '多数调用未显式覆盖采样参数。', '同上；官方示例中的 greedy 不等于所有实验必须 greedy。'],
    ['Llama-3.1-8B-Instruct', '多数调用未显式覆盖采样参数。', '官方示例允许 pipeline 继承配置；本地历史配置未恢复，不能写成确定温度 0。'],
    ['GPT-oss-20B', 'pipeline 调用未显式覆盖 T/p/k/do_sample；部分消息正文写 Reasoning: low/high。', '自动聊天模板路线有官方支持。正文文字不足以证明原生 high 级别已配置；未找到唯一强制温度。'],
]

findings = [
    {'id':'A01','category':'论文表述可纠正','finding':'论文 main.tex:668 的“全部默认且温度为0”与当前显式参数矛盾。','evidence':'basic_zero_shot/Qwen3-8B_zero_shot_re.py:140; basic_zero_shot_CoT/Qwen3-8B-CoT-new.py:151','action':'改为按模型、模式、方法报告实际设置；去掉以统一温度0保证复现的说法。'},
    {'id':'A02','category':'有效参数待核实','finding':'tree 多个脚本声明并保存 DO_SAMPLE=False，但传给 generate 的对应参数已注释；日志不等于有效配置。','evidence':'tree_zero_shot/Qwen3-8B_zero_shot.py; qwen_official_audit.json Q03; nonqwen_official_audit.json NQ3','action':'恢复原服务器配置、环境版本及源文件；保留旧日志，不把缺失值补写成0或False。'},
    {'id':'A03','category':'合法但条件不同','finding':'Qwen3-14B binary zero-shot 显式 do_sample=False，而相邻8B条件使用采样配置。','evidence':'bisection_zero_shot/Qwen3-14B_zero_shot.py:188','action':'可如实披露；若声称只改变策略或模型的受控比较，需针对该条件核实或补充对照。'},
    {'id':'A04','category':'预算与截断','finding':'max_new_tokens 是思考与答案共享的总输出上限；现有代码没有统一的“结束思考后继续生成最终答案”机制。不同上限本身不是bug。','evidence':'bisection_CoT/Qwen3-8B_few_shot.py:339; test3.py:225','action':'准确报告512/2048/32768等实际条件；若需限制思考，应另留答案预算并验证最终答案是否完成。'},
    {'id':'A05','category':'确认解释失败；分类影响未知','finding':'8组 binary-CoT 的3144条事后解释中3139条仅为<think>；另2组native binary thinking各393条解释含未闭合标记。解释在标签之后生成，不反馈预测。','evidence':'qwen_official_audit.json binary_cot_output_evidence; local_observations.json; bisection_CoT/Qwen3-8B_few_shot.py:421','action':'不能把这些字段用作有效解释；不据此宣称F1失效。未闭合思考仍可能被分类parser扫描，历史完整分类raw缺失，无法量化。'},
    {'id':'A06','category':'当前分词器已复现','finding':'CodeLlama先render模板再默认tokenizer编码，出现双BOS。原basic prompt为889 tokens，对照直接模板分词为888，前缀分别[1,1,...]和[1,...]。','evidence':'nonqwen_sources/codellama_bos_reproduction.json; basic_zero_shot/CodeLlama-7b_zero_shot.py','action':'复现基于官方分词器和Transformers5.20.0；先确认历史分词器/版本。仅改温度描述不能解决输入格式差异，影响幅度尚未测量。'},
    {'id':'A07','category':'当前源码确认格式问题','finding':'Gemma2层次零样本渲染text后实际编码原始prompt，绕过chat template；max_len形参也未使用。','evidence':'tree_zero_shot/gemma-2-9b_zero_shot.py:278','action':'恢复对应历史源文件与运行配置；若论文该结果由此实现产生，需修正实现后针对该条件验证。'},
    {'id':'A08','category':'原生模式表述待核实','finding':'GPT-oss正文Reasoning:high并非原生Harmony级别配置的充分证据；Qwen2.5模板中的enable_thinking=True也不启用原生模式。','evidence':'gptoss_official_audit.json; qwen_official_audit.json Q05; main.tex:878','action':'核对渲染后prompt与原生元数据；将提示式推理与原生thinking分开描述。'},
    {'id':'A09','category':'比较结论边界','finding':'Qwen3 thinking比较同时改变温度、top-p和输出上限；manual CoT同时启用native thinking。','evidence':'qwen_official_audit.json Q04','action':'可描述为各模式配置的整体比较；不能将差异全部因果归于thinking开关或手写规则。'},
    {'id':'A10','category':'历史复现记录不足','finding':'论文指标映射到158个run，12个无settings；146份settings只有8份记录do_sample、10份记录T/p/k，未记录seed及原生模式参数。','evidence':'local_observations.json','action':'先回收原服务器generation_config/tokenizer_config、软件版本与源文件；不能由当前官方默认值反推全部历史值。'},
    {'id':'A11','category':'覆盖边界','finding':'GPT-5示例合成来自交互工作流，缺少生成代码及机器可读参数；不能纳入本地推理参数已验证的结论。','evidence':'workflow_coverage.json; main.tex:779','action':'补充真实工作流和可取得的记录，不捏造温度。'},
    {'id':'A12','category':'辅助或旧代码','finding':'tree_few_shot中Gemma2命名脚本实际选择Gemma3；根ICL_Ob_qwen-8b.py末尾多余de导致SyntaxError。','evidence':'tree_few_shot/gemma-2-9b_few_shot.py:14; ICL_Ob_qwen-8b.py:517','action':'当前没有直接证据将这两个文件关联到活跃论文指标；不要自动算作论文结果失败。'},
]

hash_changes = []
for row in inventory:
    path = ROOT / row['script']
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
        hash_changes.append(row['script'])

conclusion = {
    'date':'2026-10-16', 'original_script_count':len(inventory),
    'generation_call_count':summary['generation_calls'],
    'active_prf_triples':224, 'matched_active_prf_triples':224,
    'active_unique_runs':158, 'metric_match_is_historical_source_proof':False,
    'no_original_script_was_executed':True,
    'original_script_hash_changes':hash_changes,
    'gptoss_original_scripts_including_arbiter':gpt['checked_original_scripts'],
    'models':models, 'findings':findings, 'official_sources':sources,
    'decision':'温度问题主要是论文表述不符，没有仅因非零温度而全量重跑的依据。但模板、有效配置日志及模式识别存在独立问题，不能保证只改一段文字就完成。',
    'truncation_clarification':'限制长输出合理。生成后删除思考，不等于限制思考预算；max_new_tokens会停止整个生成。必须区分事后解释失败与主分类答案未完成，后者历史发生率无法由已解析CSV恢复。'
}
save('audit_summary.json', conclusion)
with (HERE/'findings.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=['id','category','finding','evidence','action'])
    writer.writeheader();writer.writerows(findings)

suggested = '''DRAFT WORDING FOR REVIEW — do not claim that unresolved historical configurations have been verified.

Replacement for the blanket temperature statement (main.tex:668):
Generation settings varied across models and experimental conditions; a common zero temperature was not enforced. The principal Qwen3 non-thinking configurations used temperature 0.7, top-p 0.8, and top-k 20, whereas the thinking configurations used temperature 0.6, top-p 0.95, and top-k 20. CodeLlama used sampling with temperature 0.2 and top-p 0.95. Parameters not explicitly overridden in an inference call were inherited from the loaded checkpoint and inference implementation. Condition-specific overrides and generation limits should be listed separately.

Add an explicit exception, if the historical source confirms the current implementation:
The Qwen3-14B binary zero-shot implementation explicitly disabled sampling.

Length-control wording (accurate for the current one-stage implementation):
We capped total generated tokens to limit inference cost. This cap applied jointly to reasoning and final-answer tokens. Where supported, completed thinking blocks were removed during post-processing before extracting the predicted label. This procedure did not implement a separate thinking-token budget followed by a guaranteed final-answer stage.

Comparison wording:
The thinking and non-thinking conditions used mode-specific decoding configurations and output limits. Their comparison therefore measures the performance of the complete configurations rather than the isolated causal effect of enabling thinking.

Items that must be resolved before final submission:
1. Recover original effective checkpoint/pipeline configs and library/tokenizer versions; do not substitute today's defaults for historical values.
2. Verify the GPT-oss rendered Harmony reasoning-effort metadata before retaining the claim that all runs used high effort.
3. Verify whether the current Gemma2 hierarchical template bypass and CodeLlama duplicate-BOS path were present in the runs used for the paper.
4. Do not describe the largely '<think>' auxiliary reason fields as usable explanations; they were generated after, and did not change, the selected labels.
5. Report truncation/completion status only if supported by full saved outputs; processed label-only CSVs cannot establish it.
'''
(HERE/'suggested_manuscript_text.txt').write_text(suggested)

E=html.escape
def table(headers, rows):
    return '<table><thead><tr>'+''.join('<th>'+E(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'
def local(name,label=None):
    return '<a href="'+E(name,quote=True)+'">'+E(label or name)+'</a>'
parts=['''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>ML-SATD 生成设置审计 · 2026-10-16</title><style>
body{font:16px/1.7 system-ui,sans-serif;color:#1e293b;max-width:1200px;margin:36px auto;padding:0 24px;background:#f8fafc}h1{font-size:29px}h2{margin-top:36px}table{border-collapse:collapse;width:100%;font-size:14px;background:white}th,td{border:1px solid #d5dce5;padding:10px;vertical-align:top;text-align:left}th{background:#e9eff7}a{color:#1261a0}code,pre{font-family:ui-monospace,monospace}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#edf2f7;padding:14px;font-size:12px}details{border:1px solid #d5dce5;background:white;padding:10px;margin:8px 0}summary{cursor:pointer;overflow-wrap:anywhere}.lead{background:#e9f3ff;border-left:5px solid #2875ad;padding:18px}.note{background:#fff5df;padding:16px}small{color:#546174}</style><h1>ML-SATD 论文生成设置审计</h1><p>2026-10-16 · 当前代码、保存结果与官方资料的交叉核查</p>''']
parts.append('<div class="lead"><b>'+E(conclusion['decision'])+'</b><p>已检查114个原始Python脚本、126个生成调用；论文活跃表格的224个P/R/F1三元组全部在预期结果族中找到唯一的三位小数指标匹配，涉及158个run。指标匹配定位的是候选结果记录，不能证明当前源文件就是历史运行版本，也不构成对实验设计或全部指标计算正确性的独立认证。</p></div>')
parts.append('<p>原始脚本与论文均未改动；未运行原实验、未加载模型权重。仅用官方CodeLlama分词器做了输入格式复现。原脚本SHA-256复验变化数：'+str(len(hash_changes))+'。</p>')
parts.append('<h2>你补充的“限制过长思考”</h2><p>限制输出长度是合理选择，预算不同本身不是程序错误。需要区分：①生成后删掉已经完成的思考；②达到max_new_tokens时停止整个输出；③限制思考后，显式继续生成最终答案。原脚本主要采用①和②，未实现统一的③。</p><p>分类上限有512、2048、32768等不同条件，不能统一写成一个预算。binary-CoT的解释上限96，且仍开启thinking，3144条解释中3139条仅为&lt;think&gt;。解释在分类标签确定后生成，因此这项已确认失败不等于F1无效。分类parser只删除闭合思考块，未闭合时可能扫描思考中的yes/no或候选标签；旧记录主要保留已解析结果，无法恢复其实际发生率。</p><p>Qwen官方给出的thinking-budget示例会限制第一阶段思考，然后接续生成最终答案；这与直接缩短总输出上限不同。不能通过改论文文字将现有单阶段实现描述为该两阶段机制。</p>')
parts.append('<h2>逐模型核查</h2>'+table(['模型（共9个）','当前代码','与官方资料的关系'],[[E(v) for v in row] for row in models]))
parts.append('<h2>发现及处置边界</h2>'+table(['编号','判断','证据与含义','处理'],[[E(r['id']),E(r['category']),E(r['finding'])+'<br><small>'+E(r['evidence'])+'</small>',E(r['action'])] for r in findings]))
parts.append('<h2>先改什么、再核实什么</h2><ol><li>删除“所有温度为0、由此保证复现”的表述；按条件记录真实采样策略、参数及上限。温度非零本身不要求全量重跑。</li><li>回收原服务器的源文件、generation_config.json、tokenizer_config.json和软件版本，确定哪些当前源码问题实际进入了论文run。</li><li>如果格式问题确实进入对应run，再针对CodeLlama输入格式、Gemma2层次条件及必要的Qwen解析条件做验证；当前审计未重跑，也没有证明其数值影响。</li><li>若无法恢复完整原始输出，如实报告thinking完成状态不可核实，收窄关于纯thinking效应、解释质量和严格可复现性的表述。</li></ol>')
parts.append('<h2>论文与run证据</h2><p>234个原始run中23个缺settings。论文涉及的158个run中12个缺settings；146份现存settings只有8份包含do_sample、10份包含温度/top-p/top-k，没有保存seed、enable_thinking或reasoning_effort。不能用缺失项推断默认值。</p><p>附录table10的加权结果重复table11；“layered hybrid”标题不证明另一次独立混合实验。GPT-5示例合成缺少对应生成脚本，不在本地推理参数已验证的范围内。</p>')
parts.append('<h2>逐文件清单</h2><p>下列清单包括主方法、解释辅助脚本、旧探索脚本、仲裁实验、随机基线及工具；是否关联活跃表格已在JSON/CSV映射中单独标记。</p>')
for row in inventory:
    payload = {k:row.get(k) for k in ['script','sha256','parse_status','role','model_family','active_manuscript_candidate_triples','active_manuscript_candidate_tables','generation_calls','pipeline_factory_or_invocation','static_limitations']}
    parts.append('<details><summary>'+E(row['script'])+' · '+E(str(row.get('model_family') or row.get('role','')))+' · '+str(len(row.get('generation_calls',[])))+'个生成调用</summary><pre>'+E(json.dumps(payload,ensure_ascii=False,indent=2))+'</pre></details>')
parts.append('<h2>审计产物</h2><ul>')
for name, label in [('audit_summary.json','总体结论（JSON）'),('findings.csv','发现清单（CSV）'),('suggested_manuscript_text.txt','英文方法表述草稿'),('all_scripts_inventory.csv','114个原始脚本清单'),('table_runmapping.csv','全部表格到结果的候选映射'),('run_evidence_inventory.csv','234个运行记录'),('qwen_official_audit.json','Qwen逐调用证据'),('nonqwen_official_audit.json','Gemma/Llama等逐调用证据'),('gptoss_official_audit.json','GPT-oss逐调用证据'),('nonqwen_sources/codellama_bos_reproduction.json','CodeLlama分词器复现')]:
    parts.append('<li>'+local(name,label)+'</li>')
parts.append('</ul><h2>官方来源</h2><p>官方推荐、随附默认配置、参考示例和接口强制约束分别判断；不把示例数字当作唯一合法设置。Google和Llama四个完整配置文件访问受限（401），已明确保留不确定性，没有用第三方镜像推定旧配置。公开来源快照、revision及SHA保存在sources子目录。</p><ul>')
for label, url in sources:
    parts.append('<li><a href="'+E(url,quote=True)+'">'+E(label)+'</a></li>')
parts.append('</ul><p><small>静态审计不能恢复未保存的历史有效参数、完整生成token、停止原因和随机轨迹。该报告区分当前源码事实、保存产物事实与尚未验证的历史推断。</small></p></html>')
(HERE/'audit_report.html').write_text('\n'.join(parts))
print(json.dumps({'report':str(HERE/'audit_report.html'),'audited_scripts':len(inventory),'gptoss_scripts':gpt['checked_original_scripts'],'findings':len(findings),'original_hash_changes':hash_changes},ensure_ascii=False))
