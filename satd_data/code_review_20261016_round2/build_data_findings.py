#!/usr/bin/env python3
"""Build review findings from original static AST and model-free parser tests."""
import ast
import collections
import copy
import csv
import importlib.util
import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = BASE.parent
spec = importlib.util.spec_from_file_location('helpers', BASE / 'audit_data_helpers.py')
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
coverage = json.loads((BASE / 'data_coverage.json').read_text())
tests = json.loads((BASE / 'data_parser_tests.json').read_text())
trees, sources = {}, {}
for row in coverage:
    if row['status'] != 'reviewed':
        continue
    path = DATA / row['file']
    source = path.read_text(encoding='utf-8-sig')
    sources[row['file']] = source
    try:
        trees[row['file']] = ast.parse(source)
    except SyntaxError as error:
        trees[row['file']] = ast.parse('\n'.join(source.splitlines()[:error.lineno - 1]))


def location(file, node=None, line=None, end=None):
    return {'file': file, 'line': node.lineno if node is not None else line, 'end_line': node.end_lineno if node is not None else (end or line)}


def unique_locations(items):
    return list({(r['file'], r['line'], r['end_line']): r for r in items}.values())


def function_locations(names, files=None):
    return [location(file, node) for file, tree in trees.items() if files is None or file in files for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in names]


def matching_lines(predicate):
    return [location(file, line=i) for file, source in sources.items() for i, line in enumerate(source.splitlines(), 1) if predicate(line) and not line.lstrip().startswith('#')]


findings = []


def finding(id, title, status, severity, description, impact, recommendation, locations, evidence):
    findings.append({'id': id, 'title': title, 'status': status, 'severity': severity, 'description': description, 'impact': impact, 'recommendation': recommendation, 'locations': unique_locations(locations), 'evidence/test': evidence})


crashes = [r for r in tests if r['input'] == '' and r.get('exception') == 'IndexError']
finding('DATA-01', '层级解析器遇空输出会直接崩溃', 'confirmed', 'P2',
    'extract_coarse/extract_fine在raw是字符串时直接访问raw.splitlines()[0]；空字符串经过类型检查但没有任何首行。',
    '模型首步EOS或空content会触发IndexError；当前脚本在整批结束才保存预测，异常可使未落盘进度丢失。是否曾影响历史运行需日志确认。',
    '先检查非空字符串，再获取首个非空行；空输出返回显式invalid，并保存失败状态。',
    [location(r['file'], line=r['line'], end=r['end_line']) for r in crashes],
    {'test': '只编译原AST字符串解析函数，调用parser("")；extract_fine额外传入合法候选列表。', 'reproduced_exceptions': len(crashes), 'functions': dict(collections.Counter(r['function'] for r in crashes)), 'evidence_file': 'data_parser_tests.json'})

wrong_full = [r for r in tests if r['function'] == 'extract_abbr' and r['input'] == 'Model Code Reusability' and r.get('output') == 'MLInter']
finding('DATA-02', '全称兜底按首词把多个Model类别误映射到MLInter', 'confirmed', 'P2',
    'extract_abbr全称回退只判断full.split()[0]是否包含在输出文本中；多个全称共用Model首词，先出现的Model Interpretability因此截获Model Code Reusability等完整有效类名。',
    '这不是保守地拒绝非协议格式，而是把语义明确的完整标签变成另一类；可改变候选仲裁和最终预测。历史触发频次需重放已存输出统计。',
    '用规范化完整全称→唯一缩写映射；删除首词猜测，无法唯一映射时标invalid。',
    [location(r['file'], line=r['line'], end=r['end_line']) for r in wrong_full],
    {'input': 'Model Code Reusability', 'expected_if_full_names_are_supported': 'MCR', 'actual': 'MLInter', 'affected_functions': len(wrong_full), 'evidence_file': 'data_parser_tests.json'})

binary_stub, tree_stub = [], []
for file, tree in trees.items():
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name not in {'classify_by_polling', 'classify_two_stage'}:
            continue
        env = helpers.parser_environment(tree)
        env.update({'greedy_generate_first_line': lambda *a, **k: 'unparseable', 'generate_first_line': lambda *a, **k: 'unparseable', 'generate_once': lambda *a, **k: '<UNK>', 'build_yn_prompt': lambda *a: '', 'build_pick_from_candidates_prompt': lambda *a: '', 'build_force_pick_any_prompt': lambda *a: '', 'build_prompt_coarse': lambda *a: '', 'build_prompt_fine': lambda *a: '', 'print': lambda *a, **k: None})
        for key in ['MAX_NEW_TOKENS', 'MAX_YN_TOKENS', 'MAX_PICK_TOKENS', 'MAX_FALLBACK_TOKENS', 'COARSE_MAX_LEN', 'FINE_MAX_LEN']:
            env.setdefault(key, 32)
        candidate = copy.deepcopy(node)
        candidate.decorator_list = []
        exec(compile(ast.Module(body=[candidate], type_ignores=[]), '<model_free_control_flow>', 'exec'), env)
        result = env[node.name]('synthetic comment')
        if node.name == 'classify_by_polling':
            binary_stub.append({'file': file, 'line': node.lineno, 'end_line': node.end_lineno, 'invalid_all_model_outputs': True, 'returned_label': result[0], 'post_return_unknown_check_would_count': result[0] not in env['ABBR_SET']})
        else:
            tree_stub.append({'file': file, 'line': node.lineno, 'end_line': node.end_lineno, 'invalid_both_model_outputs': True, 'coarse': result[0], 'returned_fine': result[2], 'post_return_unknown_fine_check_would_count': result[2] not in env['FINE_ABBR_SET']})
(BASE / 'data_control_flow_tests.json').write_text(json.dumps({'binary': binary_stub, 'hierarchical': tree_stub}, ensure_ascii=False, indent=2) + '\n')
finding('DATA-03', 'Binary内部已回退，外部unk_cnt漏记真实解析失败', 'confirmed', 'P2',
    'classify_by_polling在无法解析时已返回合法Data Conf，主循环只在返回值不属于ABBR_SET时递增unk_cnt；非法仲裁又可能默认第一候选而不单独计数。',
    '报告中Unrecognized mapped...:0不能证明无解析失败；失败率与格式可靠性无法由该计数复原。最终带回退标签的F1公式本身未因此算错。',
    'helper返回raw_pred、parse_ok、fallback_used、fallback_reason和各阶段状态；分别统计无候选、非法仲裁、最终解析失败及回退。',
    [location(r['file'], line=r['line'], end=r['end_line']) for r in binary_stub] + [r for r in matching_lines(lambda line: 'if pred_abbr not in ABBR_SET' in line) if r['file'] in {x['file'] for x in binary_stub}],
    {'model_stub': '每次生成都返回unparseable；未运行任何模型或原模块。', 'reproduced_helpers': len(binary_stub), 'all_invalid_return_legal_label_and_not_counted': all(not r['post_return_unknown_check_would_count'] for r in binary_stub), 'evidence_file': 'data_control_flow_tests.json'})
finding('DATA-04', '层级unknown_fine在默认回退后统计，遗漏全部细类解析失败', 'confirmed', 'P2',
    'classify_two_stage将<UNK>替换为FALLBACK_FINE_BY_COARSE中的合法标签后才返回；主循环随后检查fine_abbr是否属于FINE_ABBR_SET，因此正常路径的unknown_fine始终为0。',
    '保存的unknown_fine=0不代表第二阶段全部可解析；会掩盖层级模型格式失败和默认类贡献。最终预测标签仍可用于报告带回退的指标。',
    '在替换前记录fine_parse_ok/fine_raw_label和fallback_used，再将状态返回主循环；保留回退前后两个字段。',
    [location(r['file'], line=r['line'], end=r['end_line']) for r in tree_stub] + matching_lines(lambda line: 'if fine_abbr not in FINE_ABBR_SET' in line),
    {'model_stub': '两个阶段均返回<UNK>。', 'reproduced_helpers': len(tree_stub), 'all_invalid_return_legal_fine_and_not_counted': all(not r['post_return_unknown_fine_check_would_count'] for r in tree_stub), 'evidence_file': 'data_control_flow_tests.json'})

coarse_locations = matching_lines(lambda line: 'pred_df["pred_coarse"] = pred_df["pred"].map' in line)
coarse_differences = []
for dirname in ['tree_zero_shot', 'tree_few_shot']:
    for file in sorted((DATA / dirname).rglob('predictions.csv')):
        with file.open(encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        different = sum(r['coarse_pred'] != r['pred_coarse'] for r in rows)
        if different:
            coarse_differences.append({'file': str(file.relative_to(DATA)), 'different_stage1_and_derived_coarse': different})
finding('DATA-05', 'Coarse指标评估最终细类回映，不等同于第一阶段coarse输出', 'design-choice', 'P2',
    '粗类报告使用pred_df.pred映射到FINE2COARSE_FULL得到pred_coarse，没有直接评估已保存的coarse_pred；阶段1未知时，阶段2全局选择仍能得到合法粗类。',
    '若论文将表解释为第一阶段性能，会混入第二阶段补救；当前zero-shot Qwen2.5-Coder有9条、CodeGemma有23条差异。若本意是最终细类聚合后的粗类指标，该公式可以保留但需改名。',
    '分别报告stage1_coarse与final_fine_aggregated_coarse，明确无效粗类按错处理；保留当前最终fine指标。',
    coarse_locations,
    {'csv_differences': coarse_differences, 'no_automatic_relabeling': True})

csv.field_size_limit(50_000_000)
with (DATA / 'dataset/sampled_label_dataset_normalized_rmvtodo.csv').open(encoding='utf-8-sig') as stream:
    dataset = list(csv.DictReader(stream))
by_id = collections.Counter(r['Dataset ID'] for r in dataset)
by_comment = collections.Counter(r['Comment Text'] for r in dataset)
finding('DATA-06', '实际393条数据仍含重复ID和重复注释', 'confirmed', 'P2',
    '最终CSV共有393行、392个唯一Dataset ID、391个唯一Comment Text；aafedbhaj的同一条记录重复，TODO: make efficient另在两个不同项目出现。',
    '当前结果对同一样本重复计权；未来按行划分示例/训练/测试会使重复或克隆跨集合。重复量较小，不应夸大成整个实验无效。',
    '先按Dataset ID、项目/路径/提交与规范化注释建立身份清单，冻结去重版本和旧结果敏感性分析；跨项目相同文本按研究目标单独标记。',
    [location('dataset/sampled_label_dataset_normalized_rmvtodo.csv', line=1)],
    {'rows': len(dataset), 'unique_ids': len(by_id), 'unique_comments': len(by_comment), 'duplicate_ids': {k:v for k,v in by_id.items() if v>1}, 'duplicate_comments': {k:v for k,v in by_comment.items() if v>1}})

join_locations = function_locations({'_read_preds', 'read_preds'}, {'battle/test.py','battle/gpt_battle.py'})
finding('DATA-07', '仲裁预测以comment作唯一键，重复注释会静默覆盖', 'confirmed', 'P2',
    '两个battle脚本将每行预测写入out[cmt]字典，再按comment查找。数据中存在重复comment，不同项目/样本的预测和reason可被最后一行替换。',
    '仲裁实验无法保持一一对应；当前重复文本已经满足覆盖条件，但具体历史仲裁预测是否改变需逐行比较。此实验未列为主文最佳结果来源。',
    '所有预测保存Dataset ID及项目/路径/提交；连接前assert key唯一并检查一对一覆盖，重复键明确报错。',
    join_locations,
    {'counterexample': '两行(comment="same",pred="A")与(comment="same",pred="B")进入out[cmt]后只有最后的B；不同Dataset ID不会被保留。', 'actual_duplicate_comment_groups': {k:v for k,v in by_comment.items() if v>1}})

legacy_locations = matching_lines(lambda line: "'satd_type'" in line and ('[[' in line or 'rename_column' in line))
with (DATA / 'dataset/OBrien_789_v2.csv').open(encoding='utf-8-sig') as stream:
    legacy_data = list(csv.DictReader(stream))
finding('DATA-08', '旧ICL代码仍把传统satd_type当作23类任务gold', 'confirmed', 'P2',
    'ICL_Ob_qwen-8b系列保留satd_type列并重命名label，而其后定义的预测候选是23个ML细粒度全称；OBrien_789_v2.csv的satd_type实际为传统Requirement/Code/Design等。',
    '即使修复旧脚本其余运行问题，该gold与候选不在同一标签空间，所得评估不能视为23类ML-SATD结果。没有证据表明这些旧脚本产出了主文已匹配的结果。',
    '将旧代码隔离为历史材料；复用时显式选择ml_satd_type并执行相同预处理，强制assert gold均属于候选集合。',
    legacy_locations,
    {'legacy_rows': len(legacy_data), 'satd_type_values': dict(collections.Counter(r['satd_type'] for r in legacy_data)), 'scope': 'Legacy root ICL scripts only; not an assertion against current method-family reports.'})

strict_cases = [r for r in tests if r['function'] in {'simple_extract','parse_abbr','extract_abbr'} and r['input'] in {'Data Conf.','**Data Conf**'} and r.get('output') == '<UNK>']
finding('DATA-09', '严格格式及宽松归一化策略不统一，需明确协议', 'design-choice', 'P3',
    '部分simple_extract先移除非字母，但ABBR_LOWER的多词key仍保留空格；Data Conf.或markdown格式因此可能返回UNK，而单词缩写的同类格式可被接受。',
    '纯缩写协议下拒绝附加标点可以是合理设计，不应把这些案例算硬bug；问题在于不同模型族的格式容忍程度不同且回退率影响评估。',
    '先明确是严格协议还是兼容常见格式；若兼容，则统一用同一canonical函数构建key和解析输出，并单独报告格式失败率。',
    [location(r['file'], line=r['line'], end=r['end_line']) for r in strict_cases],
    {'counterexamples_in_parser_tests': len(strict_cases), 'evidence_file': 'data_parser_tests.json', 'classification': 'Design choice/improvement, not a hard bug merely because punctuation is rejected.'})

yesno_cases = [r for r in tests if r['function'] in {'extract_yes_no','parse_yesno'} and r['input'] == 'Yesterday' and r.get('output') == 'yes']
finding('DATA-10', 'Yes/No宽松前缀会把普通单词或不确定输出判成投票', 'confirmed', 'P2',
    'extract_yes_no/parse_yesno使用startswith("y")、startswith("n")或startswith("yes")等前缀，未要求单词结束；Yesterday被解析为yes，Not sure被解析为no。',
    '无效或不确定输出会被当作正常二元投票，改变候选集且无法进入unknown统计；历史触发频率不能只凭反例推断。',
    '用带单词边界的严格Yes/No语法，明确单字母及中文合法形式；不确定、混合Yes/No和其他词返回unknown。',
    [location(r['file'], line=r['line'], end=r['end_line']) for r in yesno_cases],
    {'input': 'Yesterday', 'actual': 'yes', 'expected': 'unknown/invalid', 'affected_functions': len(yesno_cases), 'secondary_input': 'Not sure', 'secondary_actual': 'no', 'evidence_file': 'data_parser_tests.json'})

finding('DATA-11', '随机基线无seed且单次结果具有抽样波动', 'design-choice', 'P3',
    'random_classify.py明确不设随机种子，从23类均匀抽取一次预测；各次run保存实际预测但未定义汇总协议。',
    '单次随机F1不具确定性，把它当精确固定门槛可能改变较弱模型是否超过随机基线的判断；均匀随机本身是合法基线。',
    '固定并报告多组seed及均值/区间；或补充解析期望accuracy并明确F1采用模拟分布。保留已有run作为实现核对。',
    [location('random_classify/random_classify.py', line=47, end=60),location('random_classify/random_classify.py', line=89, end=96)],
    {'source': 'pyrandom.choice(ABBR_LIST); settings note explicitly No seed set.'})

finding('DATA-12', '预处理只保留type2的行却未将其提升为主标签', 'current-risk', 'P3',
    'filter_rows保留恰有一个标签的行，包括ML TD Type为空、ML TD Type 2非空的行；后续分类脚本仅读取ML TD Type。',
    '将来遇到secondary-only输入会把合法单标签保留为缺失gold；当前本地825原始CSV无该模式，不能声称已影响393条结果。',
    '过滤后先把唯一非空标签合并成单一gold列，再assert非空且属于23类；写出每步ID清单。',
    [location('data_process.py', line=110, end=115)],
    {'counterexample': {'ML TD Type': None, 'ML TD Type 2': 'Abs'}, 'filter_result': 'kept with primary gold still null', 'observed_current_dataset_secondary_only_rows': 0})

save_id_locations = []
for file, tree in trees.items():
    if file.startswith('ICL_') or file == 'data_process.py':
        continue
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
            if 'comment' in keys and 'real' in keys and 'pred' in keys and not any(k in keys for k in ['sample_id','dataset_id','Dataset ID']):
                save_id_locations.append(location(file,node))
finding('DATA-13', '预测记录未保存Dataset ID，难以可靠连接及审计', 'current-risk', 'P2',
    '结果行通常只包含comment、real、pred和输出文本，没有保留原CSV的Dataset ID/项目/路径/引入提交。',
    '顺序一致时当前158个论文run可逐行核对，但排序、跳过gold或跨文件仲裁后，重复注释无法可靠区分；文本键覆盖问题已在battle实现中出现。',
    '新run统一保存Dataset ID、输入行号、项目/路径/提交及原始gold；合并采用ID并assert一致；历史run保存可验证的顺序映射。',
    save_id_locations,
    {'source_dataset_has_stable_id': True, 'observed_unique_ids': len(by_id), 'observed_prediction_identity_field': 'comment only'})

def canon_comment(text):
    return re.sub(r'\s+', ' ', text.replace('\\/', '/').replace('\\"', '"').replace(' || ', '\n').replace(',', ';')).strip()

demo_hits = []
for file, tree in trees.items():
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        value = node.value
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and isinstance(value.func.value, ast.Constant):
            value = value.func.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str) and len(value.value) > 500:
            text = canon_comment(value.value)
            hits = [r['Dataset ID'] for r in dataset if len(canon_comment(r['Comment Text'])) >= 20 and canon_comment(r['Comment Text']) in text]
            if hits:
                demo_hits.append({'file': file, 'line': node.lineno, 'end_line': node.end_lineno, 'matching_final_dataset_ids': hits})
(BASE / 'data_demo_overlap_checks.json').write_text(json.dumps({'method': 'Active top-level literal prompt strings only; whitespace and known CSV serialization normalized; complete target comment >=20 characters required.', 'hits': demo_hits, 'current_main_method_family_hits': [r for r in demo_hits if not r['file'].startswith('ICL_')], 'limitation': 'No overlap in embedded strings does not prove synthetic-generation seed independence or model pretraining independence.'},ensure_ascii=False,indent=2)+'\n')
finding('DATA-14', '旧ICL固定示例含评估数据注释，须与主流程隔离', 'needs-history', 'P2',
    '旧ICL固定init_prompt_for_OBrien中嵌入最终数据集13条完整注释示例；这些固定示例不随测试fold排除。当前主文方法族的活动长字符串未检出相同完整注释重叠。',
    '复用旧ICL会把部分待测实例直接写进基础prompt；但旧文件还有运行/标签问题，且没有证据表明它们产出当前主文结果。不能据此指控主文few-shot污染。',
    '隔离legacy目录；任何真实示例仅从独立训练/开发集合选择，并记录示例ID与每个fold排除清单。补交GPT-5生成示例的种子与来源清单。',
    [location(r['file'],line=r['line'],end=r['end_line']) for r in demo_hits],
    {'evidence_file': 'data_demo_overlap_checks.json', 'overlap_per_legacy_prompt': {r['file']:len(r['matching_final_dataset_ids']) for r in demo_hits}, 'scope': 'Legacy only; no confirmed overlap in current main method-family literal prompts.'})

folds_by_repo, folds_by_text, folds_by_id = collections.defaultdict(set),collections.defaultdict(set),collections.defaultdict(set)
for row in legacy_data:
    folds_by_repo[row['repo_name']].add(row['fold'])
    folds_by_text[row['comment_text']].add(row['fold'])
    folds_by_id[row['dataset_id']].add(row['fold'])
fold_locations = []
for file in ['ICL_Ob_qwen-8b.py','ICL_Ob_qwen-8b_v0.py']:
    for node in ast.walk(trees[file]):
        if isinstance(node,ast.For) and 'test_fold' in ast.unparse(node.target):
            fold_locations.append(location(file,node))
finding('DATA-15', '旧ICL按已有fold划分仍有同ID和同文本跨集合', 'confirmed', 'P2',
    '旧ICL直接采用OBrien_789_v2.csv的fold做train/test。该文件136/299个项目跨多个fold，11组相同文本及3个Dataset ID跨fold。',
    '最近邻ICL可从训练集合取到测试实例的重复记录，且这不是跨项目划分。仅针对旧ICL复用风险，不等于当前主文零/少样本流程已经发生训练测试泄漏。',
    '先按ID/规范化文本/克隆分组，再按项目分组划分；对每fold显式assert ID、重复文本和项目隔离。',
    fold_locations+[location('dataset/OBrien_789_v2.csv',line=1)],
    {'repos_total':len(folds_by_repo),'repos_across_folds':sum(len(v)>1 for v in folds_by_repo.values()),'comment_groups_across_folds':sum(len(v)>1 for v in folds_by_text.values()),'dataset_ids_across_folds':{k:sorted(v) for k,v in folds_by_id.items() if len(v)>1}})

finding('DATA-16', '预处理版本链无法仅从当前原始CSV和入口脚本重建', 'needs-history', 'P2',
    '本地sampled_label_dataset.csv为825条，而normalized为402条，含14个不在825条中的新ID；data_process.py默认读取已经normalized的CSV，真正标准化/过滤/保存步骤在main中注释掉，删除9条TODO的步骤也未作为可运行流水线保存。',
    '无法从当前交付源文件一键重建393条相同输入；这可能是原始版本未完整交付，而非现有结果计算错误。',
    '确认用于论文的原始数据版本，保存输入/输出hash；提供独立可复现预处理入口和每步保留/排除ID及原因，不直接改写源CSV。',
    [location('data_process.py',line=6,end=8),location('data_process.py',line=125,end=138),location('dataset/sampled_label_dataset.csv',line=1),location('dataset/sampled_label_dataset_normalized.csv',line=1)],
    {'raw825_rows':825,'raw825_unlabeled':363,'raw825_dual_label':74,'raw825_single_label':388,'normalized402_new_ids_not_in_raw825':14,'normalized402_minus_final393':9,'evidence_file':'data_integrity_checks.json'})

raw_abbr_locations = []
for file in ['test1.py','test3.py','bisection_CoT/Qwen3-8B_few_shot.py','bisection_CoT/Qwen3-14B_few_shot.py']:
    for node in ast.walk(trees[file]):
        if isinstance(node,ast.Dict):
            pairs={k.value:v for k,v in zip(node.keys,node.values) if isinstance(k,ast.Constant)}
            if {'comment','pred','raw_output'} <= set(pairs):
                raw_abbr_locations.append(location(file,node))
finding('DATA-17', '部分Binary run的raw_output实际只存最终选定缩写', 'confirmed', 'P2',
    'test1/test3及Qwen Binary CoT在仲裁、全标签回退等分支把final_raw设为已解析缩写，逐标签yn_raw_all虽收集但未写入预测行。',
    '现存raw_output不能重放23个Yes/No、候选裁决及解析器；真实输出格式错误、思考未闭合和第一候选回退无法从该列完整审计。标签F1不因此自动无效，但历史错误率无法可靠恢复。',
    '分别保存每阶段未处理输出、可见输出、解析结果、候选集合、fallback及token/终止状态；raw_output只用于真实原文，最终标签单独保存。',
    raw_abbr_locations,
    {'source_observation':'final_raw = picked/final_abbr; yn_raw_all not included in saved row','scope':'Four Qwen Binary scripts; other polling helpers also omit individual Y/N raw outputs, but this finding targets the strongest explicit replacement.'})

fallback_locations=matching_lines(lambda line:'FALLBACK_ABBR' in line and '=' in line and ('"Data Conf"' in line or "'Data Conf'" in line))
finding('DATA-18', '失败回退到Data Conf并计入F1属于明确的评估策略', 'design-choice', 'P2',
    '多数分类脚本把无法解析的输出替换为最大类Data Conf，再以正常预测计入classification_report。',
    '这是可合法选择的部署策略，不能称F1公式算错；但模型自身分类能力与回退贡献混合，格式失败较多的模型可能得到部分正确预测。跨模型比较需披露该规则。',
    '保留既有带回退结果用于复现，同时另报invalid-as-wrong指标、格式失败率与fallback贡献；不要在不标注新协议时直接覆盖旧表。',
    fallback_locations,
    {'current_majority_class':'Data Conf','current_majority_support':sum(r['ML TD Type']=='Data Conf' for r in dataset),'current_total':len(dataset),'distinction':'Policy choice; DATA-03/DATA-04 concern incorrect logging of that policy, not the policy itself.'})

finding('DATA-19', '辅助bin_test解释解析截断所有含空格的合法缩写', 'confirmed', 'P2',
    'bin_test.explain的正则只允许[A-Za-z]+；合法输出Data Conf - reason不能匹配后，split(None,1)只取Data作为预测，Layer Conf、Weight Conf、HP Conf、DS Conf同样受影响。',
    '会把合法多词缩写变成不在候选集合的词；主流程没有校验chosen合法性，confusion_matrix显式labels又不包含这些截断词。该辅助原型没有与当前主文结果建立来源关联。',
    '使用共享的完整缩写映射/正则并校验候选集合；对未知输出保留invalid状态；如果不再使用该原型则隔离为legacy。',
    [location('bisection_zero_shot/bin_test.py',line=128,end=153),location('bisection_zero_shot/bin_test.py',line=169,end=194)],
    {'model_free_stub':'chat返回Data Conf - the processing step needs changes','actual_abbreviation':'Data','expected_abbreviation':'Data Conf','evidence_file':'data_auxiliary_bin_test_cases.json','scope':'Auxiliary bin_test only; not attributed to manuscript runs.'})

if (BASE/'data_observed_tree_parse_failures.json').exists():
    next(r for r in findings if r['id']=='DATA-04')['evidence/test']['current_parser_replay_file']='data_observed_tree_parse_failures.json'
if (BASE/'data_observed_fullname_misparse.json').exists():
    observed=json.loads((BASE/'data_observed_fullname_misparse.json').read_text())
    next(r for r in findings if r['id']=='DATA-02')['evidence/test']['observed_unambiguous_current_main_vanilla_fullname_misparses']=len(observed)
    next(r for r in findings if r['id']=='DATA-02')['evidence/test']['impact_caveat']='Logic defect reproduced; no unambiguous standalone-fullname misparse was found in the current active Vanilla saved outputs.'

(BASE / 'data_findings.json').write_text(json.dumps(findings,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'findings':len(findings),'status_counts':dict(collections.Counter(r['status'] for r in findings)),'locations_by_id':{r['id']:len(r['locations']) for r in findings},'control_flow_stub_counts':{'binary':len(binary_stub),'hierarchical':len(tree_stub)}},indent=2))
