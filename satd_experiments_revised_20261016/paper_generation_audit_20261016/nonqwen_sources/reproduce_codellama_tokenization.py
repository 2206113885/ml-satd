"""Read-only tokenizer reproduction; never loads model weights."""
import ast
import hashlib
import json
import platform
from pathlib import Path

import transformers
from transformers import AutoTokenizer

ROOT = Path('/Users/yizho/Desktop/satd_data')
OUT = ROOT / 'paper_generation_audit_20261016/nonqwen_sources'
SOURCE = ROOT / 'basic_zero_shot/CodeLlama-7b_zero_shot.py'
text = SOURCE.read_text()
tree = ast.parse(text)
scope = {}
for node in tree.body:
    if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'LABEL_ABBR' for target in node.targets):
        scope['ABBR_SET'] = set(ast.literal_eval(node.value).values())
for node in tree.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in {'PROMPT_HEAD', 'PROMPT_TAIL'}:
                # These two reviewed expressions contain strings, strip, sorted,
                # and join only; no model loading or module-level code executes.
                scope[target.id] = eval(compile(ast.Expression(node.value), str(SOURCE), 'eval'), {'__builtins__': {'sorted': sorted}}, scope)
    if isinstance(node, ast.FunctionDef) and node.name == 'build_prompt':
        function = node
exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), 'exec'), scope)
content = scope['build_prompt']('TODO: optimize data loading.')
messages = [{'role': 'user', 'content': content}]
tokenizer = AutoTokenizer.from_pretrained(OUT / 'codellama_tokenizer_only', local_files_only=True)
rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
original = tokenizer(rendered, truncation=True, max_length=4096)['input_ids']
direct = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
separate_no_special = tokenizer(rendered, add_special_tokens=False, truncation=True, max_length=4096)['input_ids']

def describe(ids):
    leading = 0
    for token in ids:
        if token != tokenizer.bos_token_id:
            break
        leading += 1
    return {
        'length': len(ids),
        'first_16_ids': ids[:16],
        'first_16_tokens': tokenizer.convert_ids_to_tokens(ids[:16]),
        'leading_bos_count': leading,
        'total_bos_count': ids.count(tokenizer.bos_token_id),
    }

result = {
    'purpose': 'Tokenizer-only reproduction of the exact active basic script formatting/tokenization sequence.',
    'python': platform.python_version(),
    'transformers': transformers.__version__,
    'tokenizer_class': type(tokenizer).__name__,
    'official_repo': 'codellama/CodeLlama-7b-Instruct-hf',
    'official_revision': '22cb240e0292b0b5ab4c17ccd97aa3a2f799cbed',
    'source': str(SOURCE),
    'source_sha256': hashlib.sha256(text.encode()).hexdigest(),
    'source_render_lines': [138, 142],
    'source_encode_lines': [143, 144],
    'add_bos_token': tokenizer.add_bos_token,
    'bos_token_id': tokenizer.bos_token_id,
    'rendered_prefix': rendered[:70],
    'source_pipeline': describe(original),
    'direct_apply_template_tokenize_true': describe(direct),
    'separate_encoding_add_special_tokens_false': describe(separate_no_special),
    'original_equals_one_extra_bos_plus_direct': original == [tokenizer.bos_token_id] + direct,
    'direct_equals_separate_no_special': direct == separate_no_special,
    'historical_limit': 'Original server tokenizer revision and transformers version are unavailable; the reproduction confirms current official tokenizer behavior, not the historical server runtime.',
    'weights_loaded': False,
}
(OUT / 'codellama_bos_reproduction.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps(result, ensure_ascii=False, indent=2))
