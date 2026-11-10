# [R14/R30/R35 原代码保留] 历史入口整体归档；以下原文不参与执行。
# import pandas as pd
# 
# 
# # df_file = pd.read_csv('/home/feizhiwei/ML_SATD/dataset/OBrien_789_v2.csv')
# 
# df_file = pd.read_csv('/home/feizhiwei/ML_SATD/dataset/sampled_label_dataset_normalized.csv')
# 
# df_processed_file = pd.read_csv('/home/feizhiwei/ML_SATD/dataset/sampled_label_dataset_normalized.csv')
# 
# 
# def statistic_info(df_file):
#     ML_label = []
#     for _,row in df_file.iterrows():
#         ML_label.append(row['ML TD Type'])
#     print(set(ML_label))
# 
# def statistic_info1(df_file):
#     ml_types = df_file['ML TD Type'].dropna().map(lambda x: x.strip()).unique()
#     ml_types = sorted(ml_types)
#     print("ML TD Type种类数量：", len(ml_types))
#     print("ML TD Type所有种类如下：")
#     for t in ml_types:
#         print(t)
# 
# def find_column_for_value(df, value):
#     flag=0
#     for col in df.columns:
#         if value in df[col].astype(str).values:
#             print(f"'{value}' 出现在列: {col}")
#             #把整行打印出来，不要有省略号
#             print(df[df[col].astype(str).str.contains(value)])
#             flag=1
#     if flag == 0:
#         print(f"'{value}' 未在任何列中找到")
# def count_type1_and_type2(df_file):
#     result = df_file[df_file['ML TD Type'].notna() & df_file['ML TD Type 2'].notna()]
#     print("ML TD Type和ML TD Type 2都有值的行数：", len(result))
#     return result
# def find_unexpected_types(df_file, valid_types_dict):
#     valid_types = set(valid_types_dict.keys())
#     # type1
#     type1_values = set(df_file['ML TD Type'].dropna().map(lambda x: x.strip()))
#     # type2
#     type2_values = set(df_file['ML TD Type 2'].dropna().map(lambda x: x.strip()))
#     # 查找不在字典里的值
#     unexpected_type1 = type1_values - valid_types
#     unexpected_type2 = type2_values - valid_types
#     print("ML TD Type中未在字典里的值：", unexpected_type1)
#     print("ML TD Type 2中未在字典里的值：", unexpected_type2)
# valid_types_dict = {
#     'MLKnow':'MachineLearning Knowledge',
#     'MLInter':'Model Interpretability',
#     'MCC':'Model Code Comprehension',
#     'DMC':'Duplicate Model Code',
#     'DFEC':'Duplicate Feature Extraction Code',
#     'Weight Conf':'Weight Configuration',
#     'Layer Conf':'Layer Configuration',
# 
#     'HP Conf':'Hyper-parameter Configuration',
#     'HPConf':'Hyper-parameter Configuration',
# 
#     'MLDepend':'Machine Learning Dependency',
#     'MDepend':'Machine Learning Dependency',
# 
#     'GC':'Glue Code',
#     'CDT':'Custom Data Type',
#     'ML':'Multiple Languages',
#     'UMC':'Unnecessary Model Code',
#     'Data Conf':'Data Processing Configuration',
#     'PODT':'Plain Old Data Type',
#     
#     'DS Configuration':'Data Storage Configuration',
#     'DS Conf':'Data Storage Configuration',
# 
#     'Abs':'Abstraction',
#     'BE':'Boundary Erosion',
#     'MCM':'Model Code Modifiability',   
#     'MCR':'Model Code Reusability',
#     'PQ':'Prediction Quality',
# 
#     'MLRelia': 'Machine Learning Reliability',
#     'MRelia': 'Machine Learning Reliability',
#       
#     'Proto':'Prototype',       
# }
# def normalize_types(df_file, main_type_map):
#     # type1标准化
#     df_file['ML TD Type'] = df_file['ML TD Type'].map(lambda x: main_type_map.get(x.strip(), x.strip()) if pd.notna(x) else x)
#     # type2标准化
#     df_file['ML TD Type 2'] = df_file['ML TD Type 2'].map(lambda x: main_type_map.get(x.strip(), x.strip()) if pd.notna(x) else x)
#     return df_file
# 
# # 构建主标签映射（如遇到'MRelia'就变成'MLRelia'）
# main_type_map = {
#     'MLRelia': 'MLRelia',
#     'MRelia': 'MLRelia',
#     'DS Configuration': 'DS Conf',
#     'DS Conf': 'DS Conf',
#     'MLDepend': 'MLDepend',
#     'MDepend': 'MLDepend',
#     'HPConf': 'HP Conf',
#     'HP Conf': 'HP Conf',
# }
# def normalize_types(df, main_type_map):
#     df = df.copy()
#     df['ML TD Type'] = df['ML TD Type'].map(lambda x: main_type_map.get(str(x).strip(), str(x).strip()) if pd.notna(x) else x)
#     df['ML TD Type 2'] = df['ML TD Type 2'].map(lambda x: main_type_map.get(str(x).strip(), str(x).strip()) if pd.notna(x) else x)
#     return df
# 
# def filter_rows(df):
#     # type和type2都空，舍弃
#     df = df[~(df['ML TD Type'].isna() & df['ML TD Type 2'].isna())]
#     # type和type2都存在，舍弃
#     df = df[~(df['ML TD Type'].notna() & df['ML TD Type 2'].notna())]
#     return df
# 
# def count_each_type(df):
#     # 合并两列，统计所有出现过的标签
#     all_types = pd.concat([df['ML TD Type'], df['ML TD Type 2']]).dropna().map(lambda x: str(x).strip())
#     type_counts = all_types.value_counts()
#     print("每个类（合并type和type2）出现的条数：")
#     print(type_counts)
#     return type_counts
# 
# if __name__ == '__main__':
#     # statistic_info(df_processed_file)
#     statistic_info(df_file)
#     statistic_info1(df_file)
#     find_column_for_value(df_file, 'DFEC')
#     # count_type1_and_type2(df_file)
#     # find_unexpected_types(df_file, valid_types_dict)
#     # df_file = normalize_types(df_file, main_type_map)
#     # new_df = normalize_types(df_file, main_type_map)
#     # new_df = filter_rows(new_df)
#     # new_df.to_csv('/home/feizhiwei/ML_SATD/dataset/sampled_label_dataset_normalized.csv', index=False)
#     # count_each_type("/home/feizhiwei/ML_SATD/dataset/sampled_label_dataset_normalized_rmvtodo.csv")
#     df_stat = pd.read_csv("/home/feizhiwei/ML_SATD/dataset/sampled_label_dataset_normalized_rmvtodo.csv")
#     count_each_type(df_stat)

# [R14/R30/R35 修订] 独立、显式且不覆写源CSV的数据流水线。
"""Create a NEW normalized/deduplicated dataset and row-level provenance ledger.

This implements a prospective protocol. It does not reconstruct the undocumented
historical version chain or claim to reproduce the 393-row paper dataset.
"""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from auxiliary_protocol import file_sha256, new_output_dir, normalize_label, normalize_text, read_csv, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-csv', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--source-version-note', default='Unresolved historical source/version; hash identifies only this supplied file')
    parser.add_argument('--strip-leading-debt-markers', action='store_true',
                        help='Optional NEW protocol; remove leading TODO/FIXME/XXX only, never infer an old preprocessing step')
    args = parser.parse_args()
    source = args.input_csv.resolve()
    input_hash = file_sha256(source)
    fields, rows = read_csv(source)
    required = ['Dataset ID', 'Repo Name', 'Filename', 'Comment-Introducing Revision',
                'Comment Text', 'ML TD Type', 'ML TD Type 2']
    if any(key not in fields for key in required):
        raise ValueError('Missing input columns: ' + ', '.join(key for key in required if key not in fields))
    entries = []
    for index, row in enumerate(rows, 2):
        item = dict(row)
        entry = {'source_row_1based': index, 'sample_id': row['Dataset ID'].strip(),
                 'status': 'candidate', 'reasons': [], 'secondary_promoted': False,
                 'raw_primary': row['ML TD Type'], 'raw_secondary': row['ML TD Type 2']}
        try:
            first, second = normalize_label(row['ML TD Type']), normalize_label(row['ML TD Type 2'])
        except ValueError:
            first = second = ''
            entry['reasons'].append('unknown_label')
        if not entry['reasons']:
            if first and second:
                entry['reasons'].append('two_nonempty_label_fields')
            elif not first and not second:
                entry['reasons'].append('no_label')
            else:
                item['ML TD Type'] = first or second
                item['ML TD Type 2'] = ''
                entry['secondary_promoted'] = bool(second and not first)
        if not entry['sample_id']:
            entry['reasons'].append('missing_dataset_id')
        if not row['Repo Name'].strip() or not row['Filename'].strip():
            entry['reasons'].append('missing_project_or_filename')
        comment = row['Comment Text']
        if args.strip_leading_debt_markers:
            comment = re.sub(r'^\s*(?:(?:TODO|FIXME|XXX)\b\s*[:;\-]?\s*)+', '', comment, flags=re.I)
        item['Comment Text'] = comment
        if not comment.strip():
            entry['reasons'].append('empty_comment')
        entry['gold_label'] = item.get('ML TD Type', '') if not entry['reasons'] else None
        entry['normalized_comment_sha256'] = hashlib.sha256(normalize_text(comment).encode()).hexdigest()
        entry['identity'] = [entry['sample_id'], row['Repo Name'].strip(), row['Filename'].strip(),
                             row['Comment-Introducing Revision'].strip(), entry['normalized_comment_sha256']]
        entry['row'] = item
        entries.append(entry)
    # Same-ID identity/label conflicts are excluded as whole groups, never resolved by first-row order.
    by_id = defaultdict(list)
    for entry in entries:
        if not entry['reasons']:
            by_id[entry['sample_id']].append(entry)
    for group in by_id.values():
        identities = {(tuple(e['identity']), e['gold_label']) for e in group}
        if len(identities) > 1:
            for e in group:
                e['reasons'].append('conflicting_duplicate_id_group')
        else:
            for e in group[1:]:
                e['reasons'].append('duplicate_identity_same_gold')
    # This new protocol removes same-project identical normalized comments; cross-project copies are flagged and retained.
    by_project_text = defaultdict(list)
    for entry in entries:
        if not entry['reasons']:
            by_project_text[(entry['row']['Repo Name'].strip(), entry['normalized_comment_sha256'])].append(entry)
    for group in by_project_text.values():
        if len({e['gold_label'] for e in group}) > 1:
            for e in group:
                e['reasons'].append('conflicting_project_text_gold_group')
        else:
            for e in group[1:]:
                e['reasons'].append('same_project_duplicate_text')
    cross_project = defaultdict(list)
    for e in entries:
        if not e['reasons']:
            cross_project[e['normalized_comment_sha256']].append(e)
    for group in cross_project.values():
        if len({e['row']['Repo Name'].strip() for e in group}) > 1:
            for e in group:
                e['cross_project_same_text_ids'] = [x['sample_id'] for x in group]
    selected = []
    for e in entries:
        e['status'] = 'excluded' if e['reasons'] else 'kept'
        if e['status'] == 'kept':
            if not e['gold_label']:
                raise ValueError('Internal protocol violation: empty retained gold')
            selected.append(e['row'])
    if not selected:
        raise ValueError('No rows remain; no output dataset created')
    if file_sha256(source) != input_hash:
        raise ValueError('Source CSV changed during preparation')
    output = new_output_dir(args.output_dir)
    destination = output / 'processed.csv'
    with destination.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
    ledger = output / 'row_ledger.jsonl'
    with ledger.open('w', encoding='utf-8') as handle:
        for e in entries:
            handle.write(json.dumps({k: v for k, v in e.items() if k != 'row'}, ensure_ascii=False) + '\n')
    write_json(output / 'manifest.json', {
        'protocol': 'R14/R30/R35 prospective preprocessing v1',
        'input': {'path': str(source), 'sha256': input_hash, 'rows': len(rows)},
        'source_version_note_unverified': args.source_version_note,
        'output': {'path': str(destination), 'sha256': file_sha256(destination), 'rows': len(selected)},
        'ledger_sha256': file_sha256(ledger), 'script_sha256': file_sha256(Path(__file__)),
        'helper_sha256': file_sha256(Path(__file__).with_name('auxiliary_protocol.py')),
        'excluded_reasons': dict(Counter(reason for e in entries for reason in e['reasons'])),
        'secondary_only_rows_promoted': sum(e['secondary_promoted'] and not e['reasons'] for e in entries),
        'label_counts': dict(Counter(row['ML TD Type'] for row in selected)),
        'dedup_policy': 'same ID/identity and same-project normalized text; conflicting groups excluded; cross-project same text retained and flagged',
        'strip_leading_debt_markers': args.strip_leading_debt_markers,
        'historical_393_reconstruction': 'UNRESOLVED: no missing historical raw version or undocumented steps are inferred',
        'historical_scores_or_csvs_changed': False,
    })


if __name__ == '__main__':
    main()
