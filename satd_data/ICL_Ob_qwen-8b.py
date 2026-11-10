from datasets import load_dataset,Dataset,DatasetDict
from transformers import DataCollatorWithPadding,AutoModelForSequenceClassification, Trainer, TrainingArguments,AutoTokenizer,AutoModel,AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput
from sentence_transformers import SentenceTransformer
import sentence_transformers
import torch
import torch.nn as nn
import pandas as pd
import gc
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import random
import re
import os
from vllm import LLM, SamplingParams
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# add context and prompt_context colums to the dataset
def add_context(df, inputs):
    context = [] # to be used in SentenceTransform
    prompt_context = [] # to be used in prompt generation
    for _,row in df.iterrows():
        if inputs == 'ct':
            context.append(row['comment_text'])
            prompt_context.append('### Technical debt comment: """ ' + row['comment_text'] + ' """')
        elif inputs == 'fp+ct':
            context.append(row['file_path'] + '\n' + row['comment_text'])
            prompt_context.append('### file path: ' + row['file_path'] + '\n' +
                       '### Technical debt comment: """ ' + row['comment_text'] + ' """')
        elif inputs == 'fp+cms+ct':
            context.append(row['file_path'] + '\n' + str(row['containing_method_signature']) + '\n' + row['comment_text'])
            prompt_context.append('### file path: ' + row['file_path'] + '\n' +
                       '### Containing method signature: """ ' + str(row['containing_method_signature']) + ' """\n' +
                       '### Technical debt comment: """ ' + row['comment_text'] + ' """')
        elif inputs == 'fp+ct+cmb':
            context.append(row['file_path'] + '\n' + row['comment_text'] + '\n' + str(row['containing_method']))
            prompt_context.append('### file path: ' + row['file_path'] + '\n' +
                       '### Technical debt comment: """ ' + row['comment_text'] + ' """\n' +
                       '### Containing method: """ ' + str(row['containing_method']).replace('"""',"'''") + ' """')
        elif inputs == 'cm':
            context.append(row['commit_message'])
            prompt_context.append('### Commit message: """ ' + row['commit_message'] + ' """')
        elif inputs == 'ct+cm':
            context.append(row['comment_text'] + '\n' + row['commit_message'])
            prompt_context.append('### Technical debt comment: """ ' + row['comment_text'] + ' """' + '\n' +
                                '### Commit message: """ ' + row['commit_message'] + ' """')
        elif inputs == 'fp+ct+cm':
            context.append(row['file_path'] + '\n' + row['comment_text'] + '\n' + row['commit_message'])
            prompt_context.append('### file path: ' + row['file_path'] + '\n' +
                       '### Technical debt comment: """ ' + row['comment_text'] + ' """\n' + 
                       '### Commit message: """ ' + row['commit_message'] + ' """')     
        else:
            print('ERROR!')

    df['context'] = context
    df['prompt_context'] = prompt_context
    return df

# read and prepare the OBrien dataset (SATD classification)

# df = pd.read_csv('Dataset/23_Shades/OBrien_789.csv') # this version doesn't have the containing_method and containing_method_signature columns
df = pd.read_csv('/home/feizhiwei/ML_SATD/dataset/OBrien_789_v2.csv')
print(len(df))
print(df.columns)

df = df.rename(columns={"filename": "file_path"})

DATASET = 'OBrien'

INPUT = 'ct'               # only comment text
# INPUT = 'fp+ct'            # file path + comment text
# INPUT = 'fp+cms+ct'        # file path + containing method signature + comment text
# INPUT = 'fp+ct+cmb'        # file path + comment text + containing method body
# INPUT = 'cm'                 # commit message
# INPUT = 'ct+cm'                 # comment text + commit message
# INPUT = 'fp+ct+cm'                 # file path + comment text + commit message

df = add_context(df, INPUT)

print('\n-------------- An example of input data - context ---------------\n')
print(df.context[2]) # 151
print('\n-------------- An example of input data - prompt_context ---------------\n')
print(df.prompt_context[2]) # 151
print('-------------------------------------------------------\n')

df = df[['context','prompt_context','satd_type','fold']]

# for each project, split data to train and test and save it in a dataset
dataset = {}
for test_fold in sorted(set(df['fold'])):
    test_df  = df[df['fold'] == test_fold]
    train_df = df[df['fold'] != test_fold]
    train_df = train_df.sample(frac=1, random_state=42) # shuffle train

    data = DatasetDict({"train": Dataset.from_pandas(train_df), "test": Dataset.from_pandas(test_df)})
    data=data.rename_column("satd_type","label")
    data=data.remove_columns(['fold','__index_level_0__'])
    dataset[test_fold] = data
    
METRIC = 'accuracy'


from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
import torch
import time
import pandas as pd
import numpy as np

checkpoint = "/home/feizhiwei/ML_SATD/Qwen3-8b"


# model = LLM(model="/home/feizhiwei/ML_SATD/Qwen3-8b",gpu_memory_utilization=0.99)
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载生成任务的模型
model = AutoModelForCausalLM.from_pretrained("/home/feizhiwei/ML_SATD/Qwen3-8b", device_map="auto", trust_remote_code=True)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained("/home/feizhiwei/ML_SATD/Qwen3-8b", trust_remote_code=True)



# 将模型移动到所选设备（第一个 GPU 作为主 GPU）

# tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# generation_config = GenerationConfig(max_new_tokens=5, do_sample=True, temperature=0.01) # when I set max_new_tokens=3, it generates 'Require' rather 'Requirement'

# Qwen 2.5 
# generation_config = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=4096)
from transformers import GenerationConfig

generation_config = GenerationConfig(
    temperature=0.7, 
    top_p=0.8, 
    repetition_penalty=1.05, 
    max_length=4096  # Hugging Face 使用 max_length 而非 max_tokens
)

def generate_prompt(instruction: str, input_ctxt: str = None) -> str:
    if input_ctxt:
        return f"""
### Instruction:
{instruction}

### Input:
{input_ctxt}

### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""



def get_response(model, tokenizer, prompt):
    # Tokenize input，限制最大长度，防止溢出
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)

    # 删除 token_type_ids，防止传入不支持参数
    if 'token_type_ids' in inputs:
        del inputs['token_type_ids']

    # 生成
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        repetition_penalty=1.05
        # 采样参数 temperature, top_p 可以删掉，因为您的模型不支持
    )

    # 取生成内容
    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    return generated_text


def get_confmat_str(real, pred, labels):
    output = ''
    cm = confusion_matrix(real, pred, labels=labels)
    max_label_length = max([len(label) for label in labels] + [5])
    output = " " * max_label_length + " " + " ".join(label.ljust(max_label_length) for label in labels) + "\n"
    for i, label in enumerate(labels):
        row = " ".join([str(cm[i][j]).ljust(max_label_length) for j in range(len(labels))])
        output += label.ljust(max_label_length) + " " + row + "\n"
    return output

def split_to_tokens(text):
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    return tokens

def get_the_most_relevant_items_for_an_item(item_embed, items_embed, n):
    cos_sim = sentence_transformers.util.cos_sim(item_embed, items_embed) # Note: make sure to pass ndarray not list due to performance
    itemId_similarity = dict(zip(range(len(items_embed)),cos_sim.tolist()[0]))
    itemId_similarity = dict(sorted(itemId_similarity.items(), key=lambda item: item[1], reverse=True)) # sort
    itemId_similarity = [(k,itemId_similarity[k]) for k in list(itemId_similarity)[:n]] # take top n
    return itemId_similarity

def get_the_most_relevant_items_for_an_item_given_cos_sim(item_indx, cos_sim, n):
    cos_sim_row = cos_sim[item_indx]
    itemId_similarity = dict(zip(range(len(cos_sim_row)),cos_sim_row.tolist()))
    itemId_similarity = dict(sorted(itemId_similarity.items(), key=lambda item: item[1], reverse=True)) # sort
    itemId_similarity = [(k,itemId_similarity[k]) for k in list(itemId_similarity)[:n]] # take top n
    return itemId_similarity

init_prompt_for_OBrien = \
'''There are 23 types of Machine Learning (ML) SATD (Self-Admitted Technical Debt):\n\n

MACHINE LEARNING KNOWLEDGE: Machine learning software carries plenty of unique challenges. Uneducated solutions by unaware developers may have to be revisited.\nExample: FIXME: can I backprop error through both\n
MODEL INTERPRETABILITY: Machine learning models are a black box, causing poor understanding of model's functionality. This can lead to unknown behavior.\nExample: TODO: Model Precision\n
MODEL CODE COMPREHENSION: Model code carries extra legibility concerns that do not occur in traditional software. (i.e., poorly named temporary matrix variables).\nExample: TODO refactor second part of if statement when implementing live model prediction\n
DUPLICATE MODEL CODE: Code duplication frequently occurs in model code.\nExample: TODO: Basically identical to 'test_intra_cv_target_transform' except for repeated KFold\n
DUPLICATE FEATURE EXTRACTION CODE: Code duplication frequently occurs in feature extraction code.\nExample: XXX de-duplicate this with code from Montage somewhere?\n
WEIGHT CONFIGURATION: Editing code that involves the weights of a ML model, or configuring a ML model's weights directly.\nExample: FIXME: non-uniform sample weights not yet supported\n
LAYER CONFIGURATION: Editing code that deals with ML model's layers, or configuring a model's layers itself.\nExample: TODO: experiment with more layers\n
HYPER-PARAMETER CONFIGURATION: Configuring hyper-parameters of a model, or editing the default values of off-the-shelf model.\nExample: TODO: convert this to x/y/params\n
CODE DEPENDENCY: When a needed change in ML software is blocked by an external dependency on an external library or other piece of the ML software system. Usually indicates a condition that is waiting to be met before removal.\nExample: TODO add test for keypoints once their handling was improved in Convolve\n
GLUE CODE: Supporting code written to interface with other code, inhibiting improvements due to peculiarities of dependent code.\nExample: FIXME XXX: Implement by rewriting functions above, copied from autoresolve.py\n
CUSTOM DATA TYPE: Using data types provided by general-purpose packages can cause extensive inter-operating with external libraries.\nExample: TODO: Repeat_elements and tf.split doesn't support dynamic splits.\n
MULTIPLE LANGUAGES: Components written in other languages may introduce difficulties in ML development.\nExample: TODO(sonots): Implement in C++\n
UNNECESSARY MODEL CODE: Model code that either bottlenecks execution, is unreachable or deprecated, or is unnecessary and should be removed.\nExample: DEPRICATE? I don't think this is needed anymore\n
DATA PROCESSING CONFIGURATION: Configuring the way that data is processed either by editing the data directly, or by adding in new processing steps.\nExample: TODO: normalize true states\n
PLAIN OLD DATA TYPE: Using raw data types in ML causes confusion when interpreting processes.\nExample: TODO: handle record value which are lists; at least error\n
DATA STORAGE CONFIGURATION: Configuring how data is represented within the source code (data structure) or how data is stored externally (database).\nExample: TODO json?\n
ABSTRACTION: Lack of abstractions in ML systems and subsystems cause cascading changes when changes are introduced to one component.\nExample: TODO Split into separate functions\n
BOUNDARY EROSION: Lack of boundaries between subsystems, creating difficulties when maintaining software and isolating changes made in ML software.\nExample: @todo nrows is for testing only!\n
MODEL CODE MODIFIABILITY: Model code should be implemented in ways that enable easy maintenance and future modifications.\nExample: TODO init this somewhere else in a more principled way\n
MODEL CODE REUSABILITY: Model code should be generalized to be able to be reused in varying situations.\nExample: TODO: At the moment LHUC is RNN specific.\n
PREDICTION QUALITY: Previous work in evaluating ML workflows shows that changes may affect performance.\nExample: TODO: compare the performance!\n
MACHINE LEARNING RELIABILITY: Machine learning models' functionalities are determined by the quality of their data, measures should be in place to ensure robustness.\nExample: TODO: extract an eval func more robustly\n
PROTOTYPE: Small-scale prototypes being deployed into full systems can be dangerous.\nExample: TODO: This matching process is slow. Make it faster; avoid loops where possible.\n\nHere are some examples:\n\n'''


# init_prompt_for_OBrien = \
# '''<|im_start|>system
# You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
# <|im_start|>user
# There are 23 types of Machine Learning (ML) SATD (Self-Admitted Technical Debt):\n\n

# MACHINE LEARNING KNOWLEDGE: Machine learning software carries plenty of unique challenges. Uneducated solutions by unaware developers may have to be revisited.\nExample: FIXME: can I backprop error through both\n
# MODEL INTERPRETABILITY: Machine learning models are a black box, causing poor understanding of model's functionality. This can lead to unknown behavior.\nExample: TODO: Model Precision\n
# MODEL CODE COMPREHENSION: Model code carries extra legibility concerns that do not occur in traditional software. (i.e., poorly named temporary matrix variables).\nExample: TODO refactor second part of if statement when implementing live model prediction\n
# DUPLICATE MODEL CODE: Code duplication frequently occurs in model code.\nExample: TODO: Basically identical to 'test_intra_cv_target_transform' except for repeated KFold\n
# DUPLICATE FEATURE EXTRACTION CODE: Code duplication frequently occurs in feature extraction code.\nExample: XXX de-duplicate this with code from Montage somewhere?\n
# WEIGHT CONFIGURATION: Editing code that involves the weights of a ML model, or configuring a ML model's weights directly.\nExample: FIXME: non-uniform sample weights not yet supported\n
# LAYER CONFIGURATION: Editing code that deals with ML model's layers, or configuring a model's layers itself.\nExample: TODO: experiment with more layers\n
# HYPER-PARAMETER CONFIGURATION: Configuring hyper-parameters of a model, or editing the default values of off-the-shelf model.\nExample: TODO: convert this to x/y/params\n
# CODE DEPENDENCY: When a needed change in ML software is blocked by an external dependency on an external library or other piece of the ML software system. Usually indicates a condition that is waiting to be met before removal.\nExample: TODO add test for keypoints once their handling was improved in Convolve\n
# GLUE CODE: Supporting code written to interface with other code, inhibiting improvements due to peculiarities of dependent code.\nExample: FIXME XXX: Implement by rewriting functions above, copied from autoresolve.py\n
# CUSTOM DATA TYPE: Using data types provided by general-purpose packages can cause extensive inter-operating with external libraries.\nExample: TODO: Repeat_elements and tf.split doesn't support dynamic splits.\n
# MULTIPLE LANGUAGES: Components written in other languages may introduce difficulties in ML development.\nExample: TODO(sonots): Implement in C++\n
# UNNECESSARY MODEL CODE: Model code that either bottlenecks execution, is unreachable or deprecated, or is unnecessary and should be removed.\nExample: DEPRICATE? I don't think this is needed anymore\n
# DATA PROCESSING CONFIGURATION: Configuring the way that data is processed either by editing the data directly, or by adding in new processing steps.\nExample: TODO: normalize true states\n
# PLAIN OLD DATA TYPE: Using raw data types in ML causes confusion when interpreting processes.\nExample: TODO: handle record value which are lists; at least error\n
# DATA STORAGE CONFIGURATION: Configuring how data is represented within the source code (data structure) or how data is stored externally (database).\nExample: TODO json?\n
# ABSTRACTION: Lack of abstractions in ML systems and subsystems cause cascading changes when changes are introduced to one component.\nExample: TODO Split into separate functions\n
# BOUNDARY EROSION: Lack of boundaries between subsystems, creating difficulties when maintaining software and isolating changes made in ML software.\nExample: @todo nrows is for testing only!\n
# MODEL CODE MODIFIABILITY: Model code should be implemented in ways that enable easy maintenance and future modifications.\nExample: TODO init this somewhere else in a more principled way\n
# MODEL CODE REUSABILITY: Model code should be generalized to be able to be reused in varying situations.\nExample: TODO: At the moment LHUC is RNN specific.\n
# PREDICTION QUALITY: Previous work in evaluating ML workflows shows that changes may affect performance.\nExample: TODO: compare the performance!\n
# MACHINE LEARNING RELIABILITY: Machine learning models' functionalities are determined by the quality of their data, measures should be in place to ensure robustness.\nExample: TODO: extract an eval func more robustly\n
# PROTOTYPE: Small-scale prototypes being deployed into full systems can be dangerous.\nExample: TODO: This matching process is slow. Make it faster; avoid loops where possible.\n\nHere are some examples:\n\n'''


def generate_prompt_without_adding_dynamic_examples(init_prompt, test_context):
    prompt = init_prompt
    prompt += test_context + '\n'
    prompt += '### Label: '
    return prompt

def generate_prompt_by_top_n_items(init_prompt, test_context, top_n_items, data):
    prompt = init_prompt
    for indx,similarity in top_n_items:
        if len(split_to_tokens(prompt+data['prompt_context'][indx]+test_context))<4096:
            prompt += data['prompt_context'][indx] + '\n'
            prompt += '### Label: ' + data['label'][indx].upper() + '\n\n'
    prompt += 'Give the type of following technical debt:\n'
    prompt += test_context + '\n'
    prompt += 'You only need give the type.'
    # prompt += 'You only need give the type.<|im_end|>\n<|im_start|>assistant\n'
    return prompt

def generate_prompt_by_random_n_items(init_prompt, test_context, num_rand, data):
    random_n_items = random.sample(range(len(data)), num_rand)
    prompt = init_prompt
    for indx in random_n_items:
        if len(split_to_tokens(prompt+data['prompt_context'][indx]+test_context))<4096:
            prompt += data['prompt_context'][indx] + '\n'
            prompt += '### Label: ' + data['label'][indx] + '\n\n'
    prompt += test_context + '\n'
    prompt += '### Label: '
    return prompt

from tqdm.auto import tqdm
from torch.utils.data import DataLoader
# from transformers import AdamW, get_scheduler

from torch.optim import AdamW  # 从PyTorch导入AdamW
from transformers import get_scheduler



from datasets import load_metric

# ICL_METHOD = 'task-level' # use the same prompt (zero-shot or the same demonstration examples) for all test data
ICL_METHOD = 'instance-level-nearest' # use different prompts (selects different demonstration examples) for different test samples by nearest examples selection
# ICL_METHOD = 'instance-level-random' # use different prompts (selects different demonstration examples) for different test samples by random selection

# if DATASET == 'OBrien':
#     project_name = 9
#     indx = 11
#     init_prompt = init_prompt_for_OBrien
# else:
    # project_name = 'apache-ant-1.7.0'
    # indx = 2
#     init_prompt = init_prompt_for_Maldonado62k_MAT

if DATASET == 'OBrien':
    project_name = 9
    # indx = 11
    INIT_PROMPT = init_prompt_for_OBrien
    # INIT_PROMPT = "" # provide no description for the task (i.e., just provide some examples)
elif DATASET == 'Maldonado62k':
    # INIT_PROMPT = init_prompt_for_Maldonado62k # include no keywords
    INIT_PROMPT = init_prompt_for_Maldonado62k_MAT # include MAT keywords
    # INIT_PROMPT = init_prompt_for_Maldonado62k_Easy # include Easy keywords
    # INIT_PROMPT = init_prompt_for_Maldonado62k_GPT4 # include GPT4 keywords
else:
    print("ERROR! Unknown dataset")

random.seed(42)

# from sentence_transformers import SentenceTransformer

# 从本地目录加载模型
# st_model = SentenceTransformer('/home/feizhiwei/ML_SATD/all-MiniLM-L6-v2')
# st_model = SentenceTransformer('/home/feizhiwei/ML_SATD/all-MiniLM-L6-v2', local_files_only=True)
# from sentence_transformers import SentenceTransformer

# # st_model = SentenceTransformer('file:///home/feizhiwei/ML_SATD/all-MiniLM-L6-v2')
# from sentence_transformers import SentenceTransformer

# st_model = SentenceTransformer(
#     '/home/feizhiwei/ML_SATD/all-MiniLM-L6-v2',
#     local_files_only=True,
#     device='cuda'  # 若要同时指定 device
# )
from transformers import AutoTokenizer, AutoModel
import torch

model_path = '/home/feizhiwei/ML_SATD/all-MiniLM-L6-v2'

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
transformer_model = AutoModel.from_pretrained(model_path, local_files_only=True).to('cuda')
transformer_model.eval()

def encode_sentences(sentences, tokenizer, model):
    embeddings = []
    batch_size = 32  # 可调节
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu()
        embeddings.append(batch_embeddings)
    return torch.cat(embeddings, dim=0)
# st_model = SentenceTransformer('all-MiniLM-L6-v2') # model size: 80MB

if 'instance' in ICL_METHOD:
    if len(INIT_PROMPT)>0:
        num_instances = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
        # num_instances = [24,25]
    else:
        num_instances = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
        # num_instances = [24,25]
else:
    num_instances = [0]

for NUM_EXAMPLES_IN_PROMPT in num_instances:
    print('Run the experiments for num_instances=',num_instances)

    if ICL_METHOD == 'task-level':
        icl_name = '_ICL-task'
    elif ICL_METHOD == 'instance-level-nearest':
        icl_name = '_ICL-nearest-' + str(NUM_EXAMPLES_IN_PROMPT).zfill(2)
    elif ICL_METHOD == 'instance-level-random':
        icl_name = '_ICL-random-' + str(NUM_EXAMPLES_IN_PROMPT).zfill(2)
    else:
        icl_name = '_ICL-error'

    print('Run the experiments for: ' + DATASET + '_Input-' + INPUT + '_' + checkpoint.split('/')[-1] + icl_name)
    file_name = 'Results/' + DATASET + '_Input-' + INPUT + '_' + checkpoint.split('/')[-1] + icl_name

    test_results = {}
    projects_real = {}
    projects_pred = {}
    all_real = []
    all_pred = []
    all_context = []
    all_project = []
    labels = [
        "Machine Learning Knowledge",
        "Model Interpretability",
        "Model Code Comprehension",
        "Duplicate Model Code",
        "Duplicate Feature Extraction Code",
        "Weight Configuration",
        "Layer Configuration",
        "Hyper-parameter Configuration",
        "Code Dependency",
        "Glue Code",
        "Custom Data Type",
        "Multiple Languages",
        "Unnecessary Model Code",
        "Data Processing Configuration",
        "Plain Old Data Type",
        "Data Storage Configuration",
        "Abstraction",
        "Boundary Erosion",
        "Model Code Modifiability",
        "Model Code Reusability",
        "Prediction Quality",
        "Machine Learning Reliability",
        "Prototype"
    ]

    unrecognized_pred = 0 # don't move it to outer loop
    with open(file_name+'_confmat.txt', "w") as output_file:
        for project_name, data in dataset.items():
            print('---------- '+str(project_name)+' ----------')
            output_file.write('\n---------- '+str(project_name)+' ----------\n')
            torch.cuda.empty_cache()
            gc.collect()
            if True: # ICL_METHOD == 'instance-level-nearest': 
                # train_data_embed = st_model.encode(data['train']['context'], show_progress_bar=False)
                # test_data_embed = st_model.encode(data['test']['context'], show_progress_bar=False)
                train_data_embed = encode_sentences(data['train']['context'], tokenizer, transformer_model)
                test_data_embed = encode_sentences(data['test']['context'], tokenizer, transformer_model)
                # cos_sim = sentence_transformers.util.cos_sim(test_data_embed, train_data_embed)
                cos_sim = torch.matmul(test_data_embed, train_data_embed.T) / (
                    torch.norm(test_data_embed, dim=1, keepdim=True) * torch.norm(train_data_embed, dim=1)
                    )
                cos_sim = cos_sim.cpu().numpy()
            test_results[project_name] = []
            projects_real[project_name] = []
            projects_pred[project_name] = []
            for indx, row, row_embed in zip(range(len(data['test'])), data['test'], test_data_embed):
                if ICL_METHOD == 'task-level':
                    # prompt = generate_prompt(INIT_PROMPT, row['prompt_context'])
                    prompt = generate_prompt_without_adding_dynamic_examples(INIT_PROMPT, row['prompt_context'])
                elif ICL_METHOD == 'instance-level-nearest':
                    top_n_items = get_the_most_relevant_items_for_an_item_given_cos_sim(indx, cos_sim, NUM_EXAMPLES_IN_PROMPT)
                    prompt = generate_prompt_by_top_n_items(INIT_PROMPT, row['prompt_context'], top_n_items, data['train'])
                elif ICL_METHOD == 'instance-level-random':
                    prompt = generate_prompt_by_random_n_items(INIT_PROMPT, row['prompt_context'], NUM_EXAMPLES_IN_PROMPT, data['train'])
                else:
                    print('ERROR!')
                # print(len(split_to_tokens(prompt)))
                if len(split_to_tokens(prompt))<4608:
                    # print(prompt)
                    pred = get_response(model, tokenizer, prompt)
                    print(pred)
                    for label in labels:
                        if label.lower() in pred.lower():
                            pred = label
                            break
                else:
                    pred = ''
                for label in labels:
                    if len(pred)>0 and pred.lower() == label.lower():
                        pred = label
                if pred not in labels:
                    print('---------------------------------------')
                    print(pred)
                    print('---------------------------------------')
                    # time.sleep(3)
                    pred = labels[0]  # 默认归为第一个标签“Machine Learning Knowledge”
                    unrecognized_pred += 1
                if pred=='SATD' and row['label']=='Not-SATD' and False:
                    print(prompt)
                    print('--------------------------')
                projects_real[project_name].append(row['label'])               
                projects_pred[project_name].append(pred)
                all_context.append(row['prompt_context'])
                all_project.append(project_name)
            all_real += projects_real[project_name]    
            all_pred += projects_pred[project_name]
            # print precision recall and F1 for this project
            print(classification_report(projects_real[project_name], projects_pred[project_name], zero_division=0, digits=3))
            output_file.write(classification_report(projects_real[project_name], projects_pred[project_name], zero_division=0, digits=3)+"\n")
            # print confusion matrix for this project
            confmat_str = get_confmat_str(projects_real[project_name], projects_pred[project_name], labels=labels)
            print(confmat_str)
            output_file.write(confmat_str)
        print('=========== Overall ==========')
        output_file.write('\n=========== Overall ==========\n')
        # print precision recall and F1 for all data
        print(classification_report(all_real, all_pred, zero_division=0, digits=3))
        output_file.write(classification_report(all_real, all_pred, zero_division=0, digits=3)+"\n")
        # print confusion matrix for all data
        confmat_str = get_confmat_str(all_real, all_pred, labels=labels)
        print(confmat_str)
        output_file.write(confmat_str)
        print('\nNumber of unrecognized predictions:', unrecognized_pred, '\nWe considered them as the majority class.')
        output_file.write('\nNumber of unrecognized predictions: '+str(unrecognized_pred)+'\nWe considered them as the majority class.\n')

    test_result_df = pd.DataFrame({'project': all_project, 'context':all_context, 'real': all_real, 'pred': all_pred})
    test_result_df.to_csv(file_name+'_pred.csv', index=False)de