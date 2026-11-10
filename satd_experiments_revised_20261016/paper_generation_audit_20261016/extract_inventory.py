#!/usr/bin/env python3
"""Read-only AST inventory and manuscript-to-report matching. Never imports scripts.

Only audit outputs are written. Authentication-related values are always
redacted, including nested dictionaries. No model inference or downloads occur.
"""
import ast
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent
DATA = OUT.parent
PAPER = Path('/Users/yizho/Desktop/Fine_grained_ML_SATD_Classification_with_Large_Language_Models_An_Empirical_Study')
EXCLUDE = {'models', 'pilot_context_20261016', OUT.name, '__pycache__', '.git', '.venv', 'venv'}
SENSITIVE = {'token', 'api_key', 'apikey', 'access_token', 'hf_token', 'use_auth_token', 'authorization', 'password', 'secret', 'auth_token'}
MODELS = ['Qwen3-8B', 'Qwen3-14B', 'Qwen2.5-Coder-7B-Instruct', 'gemma-2-9b', 'gemma-3-12b', 'codegemma-1.1-7b-it', 'Llama-3.1-8B-Instruct', 'CodeLlama-7b-Instruct-hf', 'gpt-oss-20b']
SAFE_SETTING_KEYS = {'csv', 'model', 'model_name', 'run_dir', 'max_new_tokens', 'fallback_abbr', 'max_tokens', 'max_yn_tokens', 'max_pick_tokens', 'max_input_tokens', 'temperature', 'top_p', 'top_k', 'do_sample', 'enable_thinking', 'reasoning_effort', 'seed', 'torch_dtype', 'dtype', 'device_map', 'max_length', 'repetition_penalty', 'eos_token_id', 'pad_token_id', 'num_beams', 'model_id', 'backend'}


def excluded(path):
    return bool(set(path.relative_to(DATA).parts) & EXCLUDE)


def model_name(value):
    value = str(value).lower()
    for name in MODELS:
        if name.lower() in value:
            return name
    for token, name in [('qwen2.5-coder-7b', MODELS[2]), ('codegemma-1.1-7b', MODELS[5]), ('llama-3.1-8b', MODELS[6]), ('codellama-7b', MODELS[7])]:
        if token in value:
            return name
    return None


def sanitize(value, key=''):
    if key.lower() in SENSITIVE:
        return '<REDACTED_AUTH_VALUE>'
    if isinstance(value, dict):
        return {str(k): sanitize(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    if isinstance(value, str):
        value = re.sub(r'\bhf_[A-Za-z0-9]{12,}\b', '<REDACTED_AUTH_VALUE>', value)
        value = re.sub(r'\bsk-[A-Za-z0-9_-]{12,}\b', '<REDACTED_AUTH_VALUE>', value)
    return value


def expression(node):
    if node is None:
        return None
    return sanitize(ast.unparse(node))


def write_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict, tuple)) else v for k, v in row.items()})


def inventory(path):
    source = path.read_text(encoding='utf-8-sig')
    result = {'script': str(path.relative_to(DATA)), 'absolute_path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'line_count': len(source.splitlines())}
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        result.update({'parse_status': 'syntax_error_prefix_parsed_not_runnable', 'parse_error_line': error.lineno, 'parse_error': error.msg})
        try:
            tree = ast.parse('\n'.join(source.splitlines()[:error.lineno - 1]))
            result['parsed_prefix_end_line'] = error.lineno - 1
        except SyntaxError:
            result['parse_status'] = 'syntax_error_not_parsed'
            return result
    else:
        result['parse_status'] = 'ok'
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    assignments = defaultdict(list)
    functions = defaultdict(list)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imported.append(expression(node))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions[node.name].append(node)
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    assignments[target.id].append((node.lineno, node.value))

    def resolve(node, line=10**9, visited=()):
        if isinstance(node, ast.Name) and node.id not in visited:
            options = [(n, value) for n, value in assignments.get(node.id, []) if n <= line]
            if options:
                ln, value = max(options, key=lambda x: x[0])
                return resolve(value, ln, visited + (node.id,))
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str) and len(node.value) > 300:
                return {'string_length': len(node.value), 'sha256': hashlib.sha256(node.value.encode()).hexdigest()}
            return sanitize(node.value)
        if isinstance(node, (ast.List, ast.Tuple)):
            return [resolve(x, line, visited) for x in node.elts]
        if isinstance(node, ast.Dict):
            return {str(resolve(k, line, visited)): sanitize(resolve(v, line, visited), str(resolve(k, line, visited))) for k, v in zip(node.keys, node.values) if k is not None}
        return {'expression': expression(node)}

    def scope_info(node):
        context, conditionals, scope_node = [], [], None
        while node in parents:
            node = parents[node]
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                context.append(node.name)
                if scope_node is None:
                    scope_node = node
            elif isinstance(node, ast.If):
                conditionals.append(expression(node.test))
        shadowed = scope_node is not None and functions[scope_node.name][-1] is not scope_node
        return {'function_scope': '.'.join(reversed(context)) or '<module>', 'inside_if_conditions': conditionals, 'inside_shadowed_function_definition': shadowed}

    def call_record(node):
        kws = {}
        for kw in node.keywords:
            name = kw.arg or '**'
            kws[name] = '<REDACTED_AUTH_VALUE>' if name.lower() in SENSITIVE else {'expression': expression(kw.value), 'resolved': resolve(kw.value, node.lineno)}
        return {'line': node.lineno, 'end_line': node.end_lineno, 'callee': expression(node.func), 'args': [resolve(x, node.lineno) for x in node.args[:3]], 'kwargs': kws, **scope_info(node)}

    pipeline_vars = set()
    for name, definitions in assignments.items():
        if any(isinstance(value, ast.Call) and expression(value.func).split('.')[-1] == 'pipeline' for _, value in definitions):
            pipeline_vars.add(name)
    categories = defaultdict(list)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = expression(node.func)
            terminal = func.split('.')[-1]
            cat = None
            if terminal in ('generate', 'chat', 'complete', 'completions') or terminal == 'create' and ('completion' in func or 'responses' in func):
                cat = 'generation_calls'
            elif terminal == 'pipeline' and (not isinstance(node.func, ast.Name) or node.func.id not in pipeline_vars or 'model' in {k.arg for k in node.keywords}):
                cat = 'pipeline_factory_or_invocation'
            elif isinstance(node.func, ast.Name) and node.func.id in pipeline_vars:
                cat = 'generation_calls'
            elif terminal in ('SamplingParams', 'GenerationConfig', 'make_sampler') or 'generation_config' in func and terminal in ('update', 'from_pretrained'):
                cat = 'sampler_or_generation_config_calls'
            elif terminal == 'apply_chat_template':
                cat = 'chat_template_calls'
            elif terminal in ('from_pretrained', 'LLM'):
                cat = 'model_and_tokenizer_load_calls'
            elif terminal in ('manual_seed', 'manual_seed_all', 'seed', 'set_seed'):
                cat = 'seed_calls'
            elif isinstance(node.func, ast.Name) and re.search(r'(tokenizer|processor)$', node.func.id):
                cat = 'tokenization_calls'
            elif terminal in ('get_response', 'greedy_generate_first_line', 'call_model', 'generate_first_line', 'generate_text', 'classify', 'generate_reason', '_gen'):
                cat = 'generation_helper_call_sites'
            elif terminal in functions:
                cat = 'local_function_call_sites'
            if cat:
                # Authentication-only utilities have no need for arbitrary
                # positional arguments in this audit.
                if path.name == 'download_model.py':
                    record = call_record(node)
                    record['args'] = [v for v in record['args'] if not isinstance(v, str) or 'REDACTED' not in v]
                else:
                    record = call_record(node)
                categories[cat].append(record)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                name = expression(target)
                if 'generation_config' in name or name.endswith(('.truncation_side', '.padding_side')):
                    categories['configuration_assignments'].append({'line': node.lineno, 'target': name, 'value': resolve(node.value, node.lineno), **scope_info(node)})
                if 'environ' in name and ('CUDA_VISIBLE_DEVICES' in name or 'VLLM' in name):
                    categories['device_environment'].append({'line': node.lineno, 'target': name, 'value': resolve(node.value, node.lineno)})
    result.update({key: sorted(value, key=lambda r: r['line']) for key, value in categories.items()})
    constants = {}
    for key, values in assignments.items():
        if key.isupper() and (any(x in key for x in ('TOKEN', 'MODEL', 'DEVICE', 'TEMP', 'TOP_', 'LENGTH', 'SEED', 'BATCH', 'PATH', 'DTYPE', 'REASON', 'INPUT'))):
            if key.lower() in SENSITIVE or key in ('PROMPT_HEAD', 'PROMPT_TAIL'):
                continue
            line, value = values[-1]
            constants[key] = {'line': line, 'value': resolve(value, line)}
    result['configuration_constants'] = constants
    result['imports'] = imported
    result['model_family'] = model_name(path.name) or model_name(json.dumps(result.get('model_and_tokenizer_load_calls', [])))
    if path.name == 'data_process.py':
        role = 'data_preprocessing_utility'
    elif path.name == 'download_model.py':
        role = 'model_download_utility_auth_values_redacted'
    elif path.parent == DATA:
        role = 'root_experimental_or_legacy_candidate_not_run_linked'
    elif path.parts[-2] == 'random_classify':
        role = 'random_baseline_result_producer'
    elif path.parts[-2] == 'battle':
        role = 'arbiter_experiment_not_in_active_manuscript_tables'
    else:
        role = 'method_family_result_producer_candidate'
    result['role'] = role
    result['backend_evidence'] = sorted(set('transformers' if 'transformers' in x else 'vllm' if 'vllm' in x else 'openai' if 'openai' in x else '' for x in imported) - {''})
    result['pipeline_variable_names'] = sorted(pipeline_vars)
    result['static_limitations'] = ['Syntactically active AST only; imported checkpoint defaults and actual execution are not inferred.', 'Current source hash is not linked to historical runs unless a source snapshot/hash was saved.', 'Resolved names are best-effort lexical assignments; dynamic kwargs and call arguments remain expressions.']
    for record in result.get('generation_calls', []):
        if 'do_sample' not in record['kwargs']:
            record['sampling_mode_observation'] = 'do_sample omitted at call; effective mode depends on passed/inherited GenerationConfig or backend defaults'
        else:
            record['sampling_mode_observation'] = 'do_sample explicitly supplied'
    return sanitize(result)


def read_table_rows(path):
    lines = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        # The manuscript does not use escaped percentage signs in result cells.
        line = re.split(r'(?<!\\)%', line)[0]
        lines.append((n, line))
    pending, start, rows = '', 1, []
    for n, line in lines:
        if not pending.strip():
            start = n
        pending += ' ' + line
        while '\\\\' in pending:
            part, pending = pending.split('\\\\', 1)
            rows.append((start, n, part))
            start = n
    return rows


def number_cells(row):
    cells = row.split('&')
    return [float(m.group(1)) if (m := re.search(r'(?<![\w.])(0\.\d+|1\.0+)(?!\d)', cell)) else None for cell in cells]


def table_records(path):
    source = path.read_text()
    label = re.search(r'\\label\{([^}]+)\}', source)
    label = label.group(1) if label else None
    active_main = re.sub(r'(?m)^\s*%.*$', '', (PAPER / 'main.tex').read_text())
    rel_no_ext = str(path.relative_to(PAPER).with_suffix(''))
    active = bool(re.search(r'\\input\{' + re.escape(rel_no_ext) + r'\}', active_main))
    rows = read_table_rows(path)
    output = []
    basename = path.name
    model_row_tables = {'zero-shot-resizebox-bold.tex': ['vanilla_zero', 'binary_zero', 'hierarchical_fine_zero'], 'zero-shot-resizebox.tex': ['vanilla_zero', 'binary_zero', 'hierarchical_fine_zero'], 'basic and bis.tex': ['vanilla_cot_zero', 'binary_cot_zero'], 'table11.tex': ['hierarchical_coarse_zero']}
    if basename in model_row_tables:
        for start, end, row in rows:
            name = model_name(row.split('&')[0])
            nums = [n for n in number_cells(row)[1:] if n is not None]
            if not name:
                continue
            for j, setting in enumerate(model_row_tables[basename]):
                if len(nums) >= 3 * (j + 1):
                    output.append({'model': name, 'setting': setting, 'shots': 0, 'precision': nums[3*j], 'recall': nums[3*j+1], 'f1': nums[3*j+2], 'table_line_start': start, 'table_line_end': end})
    elif basename == 'table10.tex':
        appendix_models = ['Qwen3-14B', 'gemma-2-9b', 'gpt-oss-20b']
        for start, end, row in rows:
            first_cell = row.split('&')[0].strip()
            match = re.search(r'\b(AWR|RDB|DCE|CFO|CDD|DTD|MDL|PRF|SCL|weighted avg)\s*$', first_cell)
            nums = [n for n in number_cells(row)[1:] if n is not None]
            if match and len(nums) == 9:
                for i, name in enumerate(appendix_models):
                    output.append({'model': name, 'setting': 'hierarchical_coarse_zero', 'shots': 0, 'report_label': match.group(1), 'precision': nums[3*i], 'recall': nums[3*i+1], 'f1': nums[3*i+2], 'table_line_start': start, 'table_line_end': end})
    elif basename == 'thinking vs no-thinking.tex':
        name, grouped = None, {}
        for start, end, row in rows:
            name = model_name(row.split('&')[0]) or name
            metric = next((m for m in ('precision', 'recall', 'f1') if re.search(r'&\s*' + m + r'\b', row)), None)
            nums = [n for n in number_cells(row) if n is not None]
            if name and metric and len(nums) == 4:
                for i, setting in enumerate(['vanilla_zero', 'vanilla_native_thinking', 'binary_zero', 'binary_native_thinking']):
                    key = (name, setting)
                    grouped.setdefault(key, {'model': name, 'setting': setting, 'shots': 0, 'table_line_start': start})[metric] = nums[i]
                    grouped[key]['table_line_end'] = end
        output.extend(grouped.values())
    elif basename in {'basic few shot.tex', 'binary few shot.tex', 'basic new cot.tex', 'bisection CoT results.tex', 'rq4.tex'}:
        grouped, shots = {}, None
        if basename == 'rq4.tex':
            headers = MODELS[:2] + [MODELS[-1]] + MODELS[:2] + [MODELS[-1]]
        else:
            header = next((row for _, _, row in rows if 'Qwen3-8B' in row and 'precision' not in row), '')
            headers = [model_name(cell) for cell in header.split('&') if model_name(cell)]
        for start, end, row in rows:
            shot_match = re.search(r'\\multirow\{3\}\{\*\}\{(\d+)\}', row)
            if shot_match:
                shots = int(shot_match.group(1))
            metric = next((m for m in ('precision', 'recall', 'f1') if re.search(r'&\s*' + m + r'\b', row)), None)
            nums = [n for n in number_cells(row) if n is not None]
            if shots is not None and metric and len(nums) == len(headers):
                for i, name in enumerate(headers):
                    setting = {'basic few shot.tex': 'vanilla_few', 'binary few shot.tex': 'binary_few', 'basic new cot.tex': 'vanilla_cot_few', 'bisection CoT results.tex': 'binary_cot_few'}.get(basename, 'vanilla_cot_few' if i < 3 else 'binary_cot_few')
                    if shots == 0:
                        setting = {'vanilla_few': 'vanilla_zero', 'binary_few': 'binary_zero', 'vanilla_cot_few': 'vanilla_cot_zero', 'binary_cot_few': 'binary_cot_zero'}.get(setting, setting)
                    key = (name, setting, shots, i)
                    grouped.setdefault(key, {'model': name, 'setting': setting, 'shots': shots, 'table_line_start': start})[metric] = nums[i]
                    grouped[key]['table_line_end'] = end
        output.extend(grouped.values())
    for row in output:
        row.update({'table': str(path.relative_to(PAPER)), 'table_absolute_path': str(path), 'table_label': label, 'active_input_in_main': active})
    return [row for row in output if all(k in row for k in ('precision', 'recall', 'f1'))]


def infer_families(setting):
    return {'vanilla_zero': ['basic_zero_shot'], 'binary_zero': ['bisection_zero_shot'], 'hierarchical_fine_zero': ['tree_zero_shot'], 'hierarchical_coarse_zero': ['tree_zero_shot'], 'vanilla_few': ['basic_few_shot'], 'binary_few': ['bisection_few_shot'], 'vanilla_native_thinking': ['basic_zero_shot_reason', 'basic_zero_shot_CoT_reason'], 'binary_native_thinking': ['bisection_zero_shot'], 'vanilla_cot_zero': ['basic_zero_shot_CoT'], 'binary_cot_zero': ['bisection_CoT'], 'vanilla_cot_few': ['basic_few_shot_CoT'], 'binary_cot_few': ['bisection_CoT']}[setting]


def scan_results():
    report_rows, run_records, report_errors = [], [], []
    for path in sorted(DATA.rglob('report*.csv')):
        if excluded(path):
            continue
        try:
            with path.open(encoding='utf-8-sig', newline='') as stream:
                reader = csv.DictReader(stream)
                rows = list(reader)
                if not reader.fieldnames:
                    report_errors.append({'report': str(path.relative_to(DATA)), 'error': 'empty_report_file'})
                    continue
                label_col = reader.fieldnames[0]
            for row in rows:
                if row.get(label_col, '').strip() in ('weighted avg', 'macro avg', 'accuracy', 'AWR', 'RDB', 'DCE', 'CFO', 'CDD', 'DTD', 'MDL', 'PRF', 'SCL'):
                    try:
                        report_rows.append({'report': str(path.relative_to(DATA)), 'report_absolute_path': str(path), 'model': model_name(str(path)), 'metric_average': row[label_col], 'precision': float(row['precision']), 'recall': float(row['recall']), 'f1': float(row['f1-score']), 'support': row.get('support')})
                    except (ValueError, KeyError):
                        pass
        except Exception as error:
            report_errors.append({'report': str(path.relative_to(DATA)), 'error': type(error).__name__})
    for path in sorted(DATA.rglob('report*.txt')):
        if excluded(path):
            continue
        for line in path.read_text(errors='replace').splitlines():
            match = re.match(r'\s*(weighted avg|macro avg|AWR|RDB|DCE|CFO|CDD|DTD|MDL|PRF|SCL)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)\s*$', line)
            if match:
                average, precision, recall, f1, support = match.groups()
                report_rows.append({'report': str(path.relative_to(DATA)), 'report_absolute_path': str(path), 'model': model_name(str(path)), 'metric_average': average, 'precision': float(precision), 'recall': float(recall), 'f1': float(f1), 'support': support, 'stored_precision': 'three_decimals_text_only'})
    run_dirs = sorted({p.parent for p in DATA.rglob('predictions.csv') if not excluded(p)} | {p.parent for p in DATA.rglob('settings.json') if not excluded(p)})
    for folder in run_dirs:
        record = {'run_dir': str(folder.relative_to(DATA))}
        settings = folder / 'settings.json'
        if settings.exists():
            try:
                values = json.loads(settings.read_text())
                record['settings'] = sanitize(values)
                record['settings_keys'] = [k for k in values if k.lower() not in SENSITIVE]
                record['settings_missing_generation_keys'] = sorted({'do_sample', 'temperature', 'top_p', 'top_k', 'seed', 'enable_thinking', 'reasoning_effort'} - set(values))
            except Exception as error:
                record['settings_error'] = type(error).__name__
        else:
            record['settings'] = None
        pred = folder / 'predictions.csv'
        if pred.exists():
            try:
                with pred.open(encoding='utf-8-sig', newline='') as stream:
                    reader = csv.DictReader(stream)
                    rows = list(reader)
                    fields = reader.fieldnames
                rawcols = [k for k in fields if any(x in k.lower() for x in ('raw', 'reason', 'output'))]
                stats = {}
                for col in rawcols:
                    strings = [str(r.get(col) or '') for r in rows]
                    stats[col] = {'nonempty': sum(bool(s.strip()) for s in strings), 'contains_open_think': sum('<think>' in s for s in strings), 'contains_close_think': sum('</think>' in s for s in strings), 'starts_analysis_channel_marker': sum(bool(re.match(r'\s*(?:analysis|<\|channel\|>analysis)', s)) for s in strings), 'max_characters': max(map(len, strings), default=0)}
                record.update({'prediction_rows': len(rows), 'prediction_fields': fields, 'raw_field_summary': stats})
            except Exception as error:
                record['prediction_error'] = type(error).__name__
        report_infos = []
        for rp in folder.glob('report*.csv'):
            try:
                with rp.open(encoding='utf-8-sig', newline='') as stream:
                    rs = list(csv.DictReader(stream))
                report_infos.extend(r.get('info', '') for r in rs if r.get('info'))
            except Exception:
                pass
        record['report_info'] = report_infos
        run_records.append(record)
    return report_rows, run_records, report_errors


def main():
    csv.field_size_limit(50_000_000)
    scripts = [inventory(path) for path in sorted(DATA.rglob('*.py')) if not excluded(path)]
    (OUT / 'all_scripts_inventory.json').write_text(json.dumps(scripts, ensure_ascii=False, indent=2) + '\n')
    flat = []
    for row in scripts:
        calls = row.get('generation_calls', []) + row.get('pipeline_factory_or_invocation', [])
        flat.append({'script': row['script'], 'model_family': row.get('model_family'), 'role': row.get('role'), 'parse_status': row['parse_status'], 'sha256': row['sha256'], 'backend_evidence': row.get('backend_evidence'), 'generation_and_pipeline_call_lines': [r['line'] for r in calls], 'generation_and_pipeline_calls': calls, 'chat_template_calls': row.get('chat_template_calls'), 'tokenization_calls': row.get('tokenization_calls'), 'sampler_or_generation_config_calls': row.get('sampler_or_generation_config_calls'), 'configuration_assignments': row.get('configuration_assignments'), 'configuration_constants': row.get('configuration_constants'), 'seed_calls': row.get('seed_calls'), 'model_and_tokenizer_load_calls': row.get('model_and_tokenizer_load_calls')})
    write_csv(OUT / 'all_scripts_inventory.csv', flat)
    reports, runs, report_errors = scan_results()
    (OUT / 'run_evidence_inventory.json').write_text(json.dumps(runs, ensure_ascii=False, indent=2) + '\n')
    write_csv(OUT / 'run_evidence_inventory.csv', runs)
    (OUT / 'report_metrics_inventory.json').write_text(json.dumps(reports, ensure_ascii=False, indent=2) + '\n')
    table_files = [PAPER / 'zero-shot-resizebox-bold.tex', PAPER / 'zero-shot-resizebox.tex'] + sorted((PAPER / 'table').glob('*.tex'))
    tables = [row for path in table_files for row in table_records(path)]
    mappings = []
    for row in tables:
        triple = tuple(f'{row[k]:.3f}' for k in ('precision', 'recall', 'f1'))
        matches = [r for r in reports if r['model'] == row['model'] and r['metric_average'] == row.get('report_label', 'weighted avg') and tuple(f'{r[k]:.3f}' for k in ('precision', 'recall', 'f1')) == triple]
        expected = infer_families(row['setting'])
        result_aliases = list(expected)
        if row['setting'] == 'vanilla_few':
            result_aliases += {1: ['one_results'], 2: ['two-results'], 3: [], 5: ['five-results'], 10: ['ten-results']}.get(row['shots'], [])
        if row['setting'] == 'vanilla_native_thinking':
            expected = ['basic_zero_shot_CoT']
            result_aliases += ['basic_zero_shot_CoT']
        if row['setting'] == 'binary_native_thinking':
            expected += ['test1.py', 'test3.py']
            result_aliases += ['results']
        likely = [r for r in matches if r['report'].split('/')[0] in result_aliases]
        script_candidates = [s['script'] for s in scripts if s.get('model_family') == row['model'] and s['script'].split('/')[0] in expected]
        mappings.append({**row, 'rounded_weighted_prf': triple, 'expected_script_families': expected, 'expected_result_directory_aliases': result_aliases, 'current_script_candidates': script_candidates, 'report_matches_all_families': [r['report'] for r in matches], 'report_matches_expected_family': [r['report'] for r in likely], 'match_status': 'unique_expected_family_metric_match' if len(likely) == 1 else 'multiple_expected_family_metric_matches' if likely else 'metric_match_outside_expected_family' if matches else 'no_exact_rounded_weighted_prf_match', 'provenance_caveat': 'Matching aggregate metrics identifies candidate result records; it does not prove which current script version generated the historical run.'})
    (OUT / 'table_runmapping.json').write_text(json.dumps(mappings, ensure_ascii=False, indent=2) + '\n')
    write_csv(OUT / 'table_runmapping.csv', mappings)
    for script in scripts:
        associated = [r for r in mappings if r['active_input_in_main'] and script['script'] in r['current_script_candidates']]
        script['active_manuscript_candidate_triples'] = len(associated)
        script['active_manuscript_candidate_tables'] = sorted({r['table'] for r in associated})
        script['active_table_mapping_status'] = 'current_script_family_candidate_only_not_historical_proof' if associated else 'no_direct_active_table_candidate_mapping'
    (OUT / 'all_scripts_inventory.json').write_text(json.dumps(scripts, ensure_ascii=False, indent=2) + '\n')
    for row, script in zip(flat, scripts):
        row['active_manuscript_candidate_triples'] = script['active_manuscript_candidate_triples']
        row['active_manuscript_candidate_tables'] = script['active_manuscript_candidate_tables']
        row['active_table_mapping_status'] = script['active_table_mapping_status']
    write_csv(OUT / 'all_scripts_inventory.csv', flat)
    active_main = re.sub(r'(?m)^\s*%.*$', '', (PAPER / 'main.tex').read_text())
    coverage = {'active_inputs': [], 'external_workflows': [{'name': 'GPT-5 synthetic example generation', 'source': str(PAPER / 'main.tex'), 'line_start': 779, 'line_end': 788, 'status': 'external_interactive_workflow_no_generation_script_or_machine_readable_generation_settings_found', 'evidence': 'Manuscript states GPT-5 deep research generated examples; current classification scripts embed resulting examples as literals. GPT-5 reasoning/sampling/template settings cannot be audited from these scripts.'}]}
    for match in re.finditer(r'\\input\{([^}]+)\}', active_main):
        rel = match.group(1) + '.tex'
        count = sum(r['table'] == rel and r['active_input_in_main'] for r in mappings)
        coverage['active_inputs'].append({'file': rel, 'matched_result_prf_triples': count, 'kind': 'quantitative_result_table' if count else 'label_count_definition_or_non_result_table'})
    (OUT / 'workflow_coverage.json').write_text(json.dumps(coverage, ensure_ascii=False, indent=2) + '\n')
    summary = {'script_count': len(scripts), 'parse_status_counts': dict(Counter(r['parse_status'] for r in scripts)), 'generation_calls': sum(len(r.get('generation_calls', [])) for r in scripts), 'pipeline_factory_or_invocation_calls': sum(len(r.get('pipeline_factory_or_invocation', [])) for r in scripts), 'script_role_counts': dict(Counter(r.get('role') for r in scripts)), 'report_metric_rows': len(reports), 'run_records': len(runs), 'runs_missing_settings': sum(r.get('settings') is None for r in runs), 'table_rows_total_including_inactive': len(mappings), 'table_rows_active': sum(r['active_input_in_main'] for r in mappings), 'active_mapping_status_counts': dict(Counter(r['match_status'] for r in mappings if r['active_input_in_main'])), 'report_errors': report_errors, 'original_files_read_only': True, 'auth_values_redacted': True, 'limitations': ['No original script was executed or imported.', 'No effective sampling mode is inferred solely from a temperature argument when do_sample is absent.', 'No historical script hashes or full prompts are available for most original runs.', 'Observed final-only raw output cannot establish that internal thinking did not occur.', 'Table comparison uses weighted P/R/F1 rounded to three decimals, never folder accuracy suffixes.']}
    (OUT / 'inventory_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
