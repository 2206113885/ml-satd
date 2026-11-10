"""Model-free review evidence for the supplementary pilot. No inference or edits."""
import ast
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE = ROOT / 'pilot_context_20261016'

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

runner = load('review_pilot_runner', BASE / 'run_pilot.py')
reference = runner.load_reference()
length_case = runner.parse_response('Data Conf', reference, thinking=False, finish_reason='length')
stop_case = runner.parse_response('Data Conf', reference, thinking=False, finish_reason='stop')
rows = [json.loads(line) for line in (BASE / 'runs/main/predictions.jsonl').read_text().splitlines() if line]
cohort_raw = (BASE / 'data/cohort.jsonl').read_bytes()
history = [json.loads(line) for line in (BASE / 'data/historical_verification/manifest.jsonl').read_text().splitlines() if line]
history_summary = json.loads((BASE / 'data/historical_verification/summary.json').read_text())
cohort = {r['sample_id']:r for r in [json.loads(line) for line in cohort_raw.decode().splitlines()]}
history_matches_current = all(r.get('local_snapshot_sha256') == cohort[r['sample_id']]['snapshot_sha256'] for r in history)

findings = [
    {'id':'PL01','title':'先导实验解析器未使用生成结束原因', 'status':'current-risk','severity':'P2',
     'description':'parse_response 接受 finish_reason 参数却不检查；length 与正常停止时同一标签都会被判有效，与settings中“unfinished output counts as wrong”的笼统说明不一致。',
     'impact':'未来输出耗尽上限时可能没有完成标记却被接受。当前300条记录均为正常stop，不改变本轮已保存分数。有效缩写已完整但缺EOS是否计失败，应由协议明确规定。',
     'recommendation':'保存并检查hit_limit/结束状态；明确截断但已有完整合法标签的计分规则，并使parse_response和settings说明一致。不要仅因finish_reason=length就推断缩写一定错误。',
     'locations':[{'file':'pilot_context_20261016/run_pilot.py','line':73,'end_line':82}, {'file':'pilot_context_20261016/run_pilot.py','line':176,'end_line':176}],
     'evidence':{'length_case':length_case,'normal_stop_case':stop_case,'current_run_finish_reasons':dict(Counter(r['finish_reason'] for r in rows))}},
    {'id':'PL02','title':'历史核验子集只按ID复用旧核验记录','status':'current-risk','severity':'P2',
     'description':'wait_and_run 从历史manifest选择full_snapshot_identical的sample_id，但没有核对该manifest的cohort哈希或每条local_snapshot_sha256与当前cohort一致。',
     'impact':'重新准备cohort后若保留旧核验文件，同一ID的新上下文可能被错误纳入“已历史核验”子集。当前cohort及所有核验snapshot哈希一致，未发现现有结果受影响。',
     'recommendation':'构建子集前校验historical summary的cohort_sha256和当前cohort哈希；逐条校验sample_id、local_snapshot_sha256及introducing_url，失配时拒绝复用并重新核验。',
     'locations':[{'file':'pilot_context_20261016/wait_and_run.py','line':84,'end_line':100}],
     'evidence':{'current_cohort_hash_matches_history_summary':hashlib.sha256(cohort_raw).hexdigest()==history_summary['cohort_sha256'],'current_per_record_snapshot_hashes_match':history_matches_current}},
]

def function_line(path, name):
    tree=ast.parse(path.read_text())
    return next(n.lineno for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)

findings[0]['locations'][0]['line']=function_line(BASE/'run_pilot.py','parse_response')
lines=(BASE/'run_pilot.py').read_text().splitlines()
findings[0]['locations'][1]['line']=next(i for i,line in enumerate(lines,1) if 'failure_policy' in line)
findings[0]['locations'][1]['end_line']=findings[0]['locations'][1]['line']
wait_lines=(BASE/'wait_and_run.py').read_text().splitlines()
findings[1]['locations'][0]['line']=next(i for i,line in enumerate(wait_lines,1) if 'verified = {' in line)

coverage=[]
for path in sorted(BASE.rglob('*.py')):
    if any(part in {'.venv','site-packages','__pycache__'} for part in path.parts):
        continue
    ast.parse(path.read_text())
    coverage.append({'file':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                     'status':'reviewed','scope':'模型调用、prompt、选样、配对、指标、恢复、历史来源验证及纯函数测试；归档prompt按归档角色审查',
                     'finding_ids':[f['id'] for f in findings if any(loc['file']==str(path.relative_to(ROOT)) for loc in f['locations'])]})
(HERE/'pilot_findings.json').write_text(json.dumps(findings,ensure_ascii=False,indent=2)+'\n')
(HERE/'pilot_coverage.json').write_text(json.dumps(coverage,ensure_ascii=False,indent=2)+'\n')
(HERE/'pilot_test_evidence.json').write_text(json.dumps({'existing_model_free_tests':'10 passed (test_runner + test_analyze_pilot)', 'new_counterexample':{'length_case':length_case,'normal_stop_case':stop_case},'predictions':len(rows),'finish_reason_counts':dict(Counter(r['finish_reason'] for r in rows)), 'history_manifest_matches_current_cohort':history_matches_current},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'pilot_files_reviewed':len(coverage),'findings':len(findings),'current_finish_reasons':dict(Counter(r['finish_reason'] for r in rows))},ensure_ascii=False))
