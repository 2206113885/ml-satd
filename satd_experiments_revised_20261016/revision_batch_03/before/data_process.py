import pandas as pd


# df_file = pd.read_csv('/home/feizhiwei/ML_SATD/dataset/OBrien_789_v2.csv')

df_file = pd.read_csv('/home/feizhiwei/ML_SATD/dataset/sampled_label_dataset_normalized.csv')

df_processed_file = pd.read_csv('/home/feizhiwei/ML_SATD/dataset/sampled_label_dataset_normalized.csv')


def statistic_info(df_file):
    ML_label = []
    for _,row in df_file.iterrows():
        ML_label.append(row['ML TD Type'])
    print(set(ML_label))

def statistic_info1(df_file):
    ml_types = df_file['ML TD Type'].dropna().map(lambda x: x.strip()).unique()
    ml_types = sorted(ml_types)
    print("ML TD Type种类数量：", len(ml_types))
    print("ML TD Type所有种类如下：")
    for t in ml_types:
        print(t)

def find_column_for_value(df, value):
    flag=0
    for col in df.columns:
        if value in df[col].astype(str).values:
            print(f"'{value}' 出现在列: {col}")
            #把整行打印出来，不要有省略号
            print(df[df[col].astype(str).str.contains(value)])
            flag=1
    if flag == 0:
        print(f"'{value}' 未在任何列中找到")
def count_type1_and_type2(df_file):
    result = df_file[df_file['ML TD Type'].notna() & df_file['ML TD Type 2'].notna()]
    print("ML TD Type和ML TD Type 2都有值的行数：", len(result))
    return result
def find_unexpected_types(df_file, valid_types_dict):
    valid_types = set(valid_types_dict.keys())
    # type1
    type1_values = set(df_file['ML TD Type'].dropna().map(lambda x: x.strip()))
    # type2
    type2_values = set(df_file['ML TD Type 2'].dropna().map(lambda x: x.strip()))
    # 查找不在字典里的值
    unexpected_type1 = type1_values - valid_types
    unexpected_type2 = type2_values - valid_types
    print("ML TD Type中未在字典里的值：", unexpected_type1)
    print("ML TD Type 2中未在字典里的值：", unexpected_type2)
valid_types_dict = {
    'MLKnow':'MachineLearning Knowledge',
    'MLInter':'Model Interpretability',
    'MCC':'Model Code Comprehension',
    'DMC':'Duplicate Model Code',
    'DFEC':'Duplicate Feature Extraction Code',
    'Weight Conf':'Weight Configuration',
    'Layer Conf':'Layer Configuration',

    'HP Conf':'Hyper-parameter Configuration',
    'HPConf':'Hyper-parameter Configuration',

    'MLDepend':'Machine Learning Dependency',
    'MDepend':'Machine Learning Dependency',

    'GC':'Glue Code',
    'CDT':'Custom Data Type',
    'ML':'Multiple Languages',
    'UMC':'Unnecessary Model Code',
    'Data Conf':'Data Processing Configuration',
    'PODT':'Plain Old Data Type',
    
    'DS Configuration':'Data Storage Configuration',
    'DS Conf':'Data Storage Configuration',

    'Abs':'Abstraction',
    'BE':'Boundary Erosion',
    'MCM':'Model Code Modifiability',   
    'MCR':'Model Code Reusability',
    'PQ':'Prediction Quality',

    'MLRelia': 'Machine Learning Reliability',
    'MRelia': 'Machine Learning Reliability',
      
    'Proto':'Prototype',       
}
def normalize_types(df_file, main_type_map):
    # type1标准化
    df_file['ML TD Type'] = df_file['ML TD Type'].map(lambda x: main_type_map.get(x.strip(), x.strip()) if pd.notna(x) else x)
    # type2标准化
    df_file['ML TD Type 2'] = df_file['ML TD Type 2'].map(lambda x: main_type_map.get(x.strip(), x.strip()) if pd.notna(x) else x)
    return df_file

# 构建主标签映射（如遇到'MRelia'就变成'MLRelia'）
main_type_map = {
    'MLRelia': 'MLRelia',
    'MRelia': 'MLRelia',
    'DS Configuration': 'DS Conf',
    'DS Conf': 'DS Conf',
    'MLDepend': 'MLDepend',
    'MDepend': 'MLDepend',
    'HPConf': 'HP Conf',
    'HP Conf': 'HP Conf',
}
def normalize_types(df, main_type_map):
    df = df.copy()
    df['ML TD Type'] = df['ML TD Type'].map(lambda x: main_type_map.get(str(x).strip(), str(x).strip()) if pd.notna(x) else x)
    df['ML TD Type 2'] = df['ML TD Type 2'].map(lambda x: main_type_map.get(str(x).strip(), str(x).strip()) if pd.notna(x) else x)
    return df

def filter_rows(df):
    # type和type2都空，舍弃
    df = df[~(df['ML TD Type'].isna() & df['ML TD Type 2'].isna())]
    # type和type2都存在，舍弃
    df = df[~(df['ML TD Type'].notna() & df['ML TD Type 2'].notna())]
    return df

def count_each_type(df):
    # 合并两列，统计所有出现过的标签
    all_types = pd.concat([df['ML TD Type'], df['ML TD Type 2']]).dropna().map(lambda x: str(x).strip())
    type_counts = all_types.value_counts()
    print("每个类（合并type和type2）出现的条数：")
    print(type_counts)
    return type_counts

if __name__ == '__main__':
    # statistic_info(df_processed_file)
    statistic_info(df_file)
    statistic_info1(df_file)
    find_column_for_value(df_file, 'DFEC')
    # count_type1_and_type2(df_file)
    # find_unexpected_types(df_file, valid_types_dict)
    # df_file = normalize_types(df_file, main_type_map)
    # new_df = normalize_types(df_file, main_type_map)
    # new_df = filter_rows(new_df)
    # new_df.to_csv('/home/feizhiwei/ML_SATD/dataset/sampled_label_dataset_normalized.csv', index=False)
    # count_each_type("/home/feizhiwei/ML_SATD/dataset/sampled_label_dataset_normalized_rmvtodo.csv")
    df_stat = pd.read_csv("/home/feizhiwei/ML_SATD/dataset/sampled_label_dataset_normalized_rmvtodo.csv")
    count_each_type(df_stat)