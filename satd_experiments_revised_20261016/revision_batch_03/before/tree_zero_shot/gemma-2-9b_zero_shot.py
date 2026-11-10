import re, os, gc, json
from pathlib import Path
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_float32_matmul_precision('high')

# ========== 0. 基础路径（仅此处改） ==========
os.environ["CUDA_VISIBLE_DEVICES"] = "1"        # GPU 号
CSV_PATH   = "/data/yangzhifei/pro/ML_SATD/dataset/sampled_label_dataset_normalized_rmvtodo.csv"  # ← 修改
MODEL_PATH = "/data/yangzhifei/pro/ML_SATD/models/gemma-2-9b-it"                                        # ← 修改

MODEL_NAME = Path(MODEL_PATH).name
RUN_DIR    = Path("results") / f"run-{datetime.now():%Y%m%d-%H%M%S}-{MODEL_NAME}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

# ========== 1. 其它参数 ==========
MAX_NEW_TOKENS = 2048
COARSE_MAX_LEN = 2048
FINE_MAX_LEN = 2048
PRINT_UNPARSED = True


# ========== 2. 细粒度标签（23 类） ==========
FINE_LABEL_ABBR = {
    "Machine Learning Knowledge":"MLKnow","Model Interpretability":"MLInter",
    "Model Code Comprehension":"MCC","Duplicate Model Code":"DMC",
    "Duplicate Feature Extraction Code":"DFEC","Weight Configuration":"Weight Conf",
    "Layer Configuration":"Layer Conf","Hyper-parameter Configuration":"HP Conf",
    "Code Dependency":"MLDepend","Glue Code":"GC","Custom Data Type":"CDT",
    "Multiple Languages":"ML","Unnecessary Model Code":"UMC",
    "Data Processing Configuration":"Data Conf","Plain Old Data Type":"PODT",
    "Data Storage Configuration":"DS Conf","Abstraction":"Abs","Boundary Erosion":"BE",
    "Model Code Modifiability":"MCM","Model Code Reusability":"MCR",
    "Prediction Quality":"PQ","Machine Learning Reliability":"MLRelia","Prototype":"Proto",
}
FINE_ABBR2FULL  = {v:k for k,v in FINE_LABEL_ABBR.items()}
FINE_LABELS     = list(FINE_LABEL_ABBR.keys())
FINE_ABBR_SET   = set(FINE_ABBR2FULL.keys())

# ========== 3. 粗粒度（9 类）与细↔粗映射 ==========
COARSE_LABELS = ["AWR","RDB","DCE","CFO","CDD","DTD","MDL","PRF","SCL"]
COARSE_ABBR_SET = set(COARSE_LABELS)
COARSE_ABBR_LOWER = {a.lower():a for a in COARSE_LABELS}

# 细(缩写) → 粗
FINE2COARSE_ABBR = {
    "MLKnow":"AWR","MLInter":"AWR",
    "MCC":"RDB",
    "DMC":"DCE","DFEC":"DCE",
    "Weight Conf":"CFO","Layer Conf":"CFO","HP Conf":"CFO",
    "MLDepend":"CDD","GC":"CDD","CDT":"CDD","ML":"CDD","UMC":"CDD",
    "Data Conf":"DTD","PODT":"DTD","DS Conf":"DTD",
    "Abs":"MDL","BE":"MDL","MCM":"MDL","MCR":"MDL",
    "PQ":"PRF","MLRelia":"PRF",
    "Proto":"SCL",
}

# 粗 → 细(缩写) 候选集合
COARSE2FINE_ABBR = {
    "AWR":  ["MLKnow","MLInter"],
    "RDB":  ["MCC"],
    "DCE":  ["DMC","DFEC"],
    "CFO":  ["Weight Conf","Layer Conf","HP Conf"],
    "CDD":  ["MLDepend","GC","CDT","ML","UMC"],
    "DTD":  ["Data Conf","PODT","DS Conf"],
    "MDL":  ["Abs","BE","MCM","MCR"],
    "PRF":  ["PQ","MLRelia"],
    "SCL":  ["Proto"],
}

# 细(全称) → 粗（用于评估映射）
FINE2COARSE_FULL = {
    "Machine Learning Knowledge":"AWR",
    "Model Interpretability":"AWR",
    "Model Code Comprehension":"RDB",
    "Duplicate Model Code":"DCE",
    "Duplicate Feature Extraction Code":"DCE",
    "Weight Configuration":"CFO",
    "Layer Configuration":"CFO",
    "Hyper-parameter Configuration":"CFO",
    "Code Dependency":"CDD",
    "Glue Code":"CDD",
    "Custom Data Type":"CDD",
    "Multiple Languages":"CDD",
    "Unnecessary Model Code":"CDD",
    "Data Processing Configuration":"DTD",
    "Plain Old Data Type":"DTD",
    "Data Storage Configuration":"DTD",
    "Abstraction":"MDL",
    "Boundary Erosion":"MDL",
    "Model Code Modifiability":"MDL",
    "Model Code Reusability":"MDL",
    "Prediction Quality":"PRF",
    "Machine Learning Reliability":"PRF",
    "Prototype":"SCL",
}

# 各粗类下细类的回退（细类解析失败时使用）
FALLBACK_FINE_BY_COARSE = {
    "AWR":"MLKnow",
    "RDB":"MCC",
    "DCE":"DMC",
    "CFO":"HP Conf",
    "CDD":"MLDepend",
    "DTD":"Data Conf",
    "MDL":"MCM",
    "PRF":"PQ",
    "SCL":"Proto",
}

# ========== 4. Prompt（Coarse 9 类） ==========
COARSE_PROMPT_HEAD = r"""
There are 9 coarse-grained types of Machine Learning (ML) Self-Admitted Technical Debt (SATD):

Awareness (AWR): Lack of developer understanding or certainty about ML algorithms, APIs, data, or model behavior. Often arises due to “black box” nature of ML models.
Example Symptoms:Doubts on correctness, API usage uncertainty, unclear design decisions, unknown error causes, lack of domain knowledge.

Readability (RDB):Model code has poor legibility or unclear logic, especially in ML contexts (e.g., poorly named variables, dense math code).
Example Symptoms:Refactoring for clarity, simplifying complex statements, improving naming.

Duplicate Code Elimination (DCE): Identical or near-identical ML model or preprocessing logic appears in multiple places.
Example Symptoms:Repeated feature extraction, repeated model evaluation loops.

Configurable Options (CFO): Missing or poorly implemented configuration points for weights, network layers, or model parameters.
Example Symptoms:Hard-coded parameters, unimplemented config flags, missing flexibility for experiments.

Code Dependency (CDD): Code depends on other code or libraries in a way that blocks maintenance, requires workarounds, or is outdated.
Example Symptoms:Waiting for library updates, integration hacks, dependency-induced bugs, deprecated model code.

Data Dependency (DTD): Issues in how data is processed, represented, or stored that hinder maintainability or correctness.
Example Symptoms:Missing preprocessing steps, poor choice of data formats, unoptimized storage.

Modularity (MDL): Poor modular boundaries or design make the system harder to evolve, reuse, or maintain.
Example Symptoms:Missing abstractions, test parameters used in production, model-specific code that should be generalized.

Performance (PRF): Performance or robustness is insufficient, requiring evaluation, optimization, or reliability improvements.
Example Symptoms:Model accuracy concerns, robustness checks, performance comparisons.

Scalability (SCL): Prototype-level code is used in production without scaling optimizations.
Example Symptoms:Inefficient loops, naive algorithms, lack of batching/parallelism.
""".strip()

def build_prompt_coarse(comment: str) -> str:
    tail = "Please classify the following Code Comment into exactly ONE of these coarse abbreviations:\n" \
           + ", ".join(COARSE_LABELS) + \
           "\nReturn the abbreviation only (e.g., AWR)."
    return f"{COARSE_PROMPT_HEAD}\n\n{tail}\n\nCode Comment:\n\"\"\"\n{comment}\n\"\"\"\n\nAnswer:"

# ========== 5. Prompt（Fine 23 类；按粗类限制候选） ==========
FINE_PROMPT_HEAD = r"""
There are 23 types of Machine Learning (ML) Self-Admitted Technical Debt (SATD):

MACHINE LEARNING KNOWLEDGE (MLKnow): Machine-learning software carries plenty of unique challenges. Uneducated solutions by unaware developers may have to be revisited.

MODEL INTERPRETABILITY (MLInter): Machine-learning models are a black box, causing poor understanding of model functionality; this can lead to unknown behaviour.

MODEL CODE COMPREHENSION (MCC): Model code carries extra legibility concerns not present in traditional software (e.g., poorly-named temporary matrix variables).

DUPLICATE MODEL CODE (DMC): Code duplication frequently occurs in model code.

DUPLICATE FEATURE EXTRACTION CODE (DFEC): Code duplication frequently occurs in feature-extraction code.

WEIGHT CONFIGURATION (Weight Conf): Editing code that involves the weights of a model, or configuring a model's weights directly.

LAYER CONFIGURATION (Layer Conf): Editing code that deals with model layers, or configuring layers themselves.

HYPER-PARAMETER CONFIGURATION (HP Conf): Configuring hyper-parameters of a model, or editing default values of off-the-shelf models.

CODE DEPENDENCY (MLDepend): A needed change is blocked by an external dependency (library / subsystem).

GLUE CODE (GC): Supporting code written to interface with other code, inhibiting improvements due to dependent peculiarities.

CUSTOM DATA TYPE (CDT): Using data types provided by general-purpose packages can cause extensive inter-operability burden.

MULTIPLE LANGUAGES (ML): Components written in other languages may introduce difficulties in ML development.

UNNECESSARY MODEL CODE (UMC): Model code that bottlenecks execution, is unreachable / deprecated, or is unnecessary and should be removed.

DATA PROCESSING CONFIGURATION (Data Conf): Configuring how data is processed, by editing the data directly or adding new processing steps.

PLAIN OLD DATA TYPE (PODT): Using raw data types in ML causes confusion when interpreting processes.

DATA STORAGE CONFIGURATION (DS Conf): Configuring how data is represented internally or stored externally.

ABSTRACTION (Abs): Lack of abstractions in ML systems causes cascading changes when a component is modified.

BOUNDARY EROSION (BE): Lack of boundaries between subsystems complicates maintenance and isolates changes poorly.

MODEL CODE MODIFIABILITY (MCM): Model code should enable easy maintenance and future modifications.

MODEL CODE REUSABILITY (MCR): Model code should be generalised for reuse in varying situations.

PREDICTION QUALITY (PQ): Workflow changes may affect performance.

MACHINE LEARNING RELIABILITY (MLRelia): Model functionality depends on data quality; robustness measures should be in place.

PROTOTYPE (Proto): Deploying small-scale prototypes into full systems can be dangerous.
""".strip()

def build_prompt_fine(comment: str, coarse_abbr: str) -> str:
    allowed = COARSE2FINE_ABBR.get(coarse_abbr, sorted(FINE_ABBR_SET))
    tail = "Please determine the fine-grained ML SATD type. You MUST answer with only ONE abbreviation from:\n" \
           + ", ".join(allowed) + "\nReturn the abbreviation only (e.g., HP Conf)."
    return f"{FINE_PROMPT_HEAD}\n\n{tail}\n\nCode Comment:\n\"\"\"\n{comment}\n\"\"\"\n\nAnswer:"

# ========== 6. 解析函数（健壮版） ==========
COARSE_REGEX = re.compile(r'\b(AWR|RDB|DCE|CFO|CDD|DTD|MDL|PRF|SCL)\b', re.I)

def extract_coarse(raw: str) -> str:
    """从任意首行文本里提取粗类缩写（容忍 'Abbreviation: **AWR** (Awareness)' 等格式）"""
    if not isinstance(raw, str):
        return "<UNK>"

    # 1) 直接检索允许缩写
    m = COARSE_REGEX.search(raw)
    if m:
        return m.group(1).upper()

    # 2) markdown **AWR**
    m = re.search(r'\*\*\s*(AWR|RDB|DCE|CFO|CDD|DTD|MDL|PRF|SCL)\s*\*\*', raw, flags=re.I)
    if m:
        return m.group(1).upper()

    # 3) 兜底（去非字母后包含匹配，如 abbreviationawrawareness）
    letters = re.sub(r'[^A-Za-z]', '', raw).lower()
    for abbr in COARSE_LABELS:
        if abbr.lower() in letters:
            return abbr
    return "<UNK>"

def extract_fine(raw: str, allowed_abbr_list) -> str:
    """从 raw 中提取细类缩写，仅允许 allowed_abbr_list 内的候选。兼容空格差异与 markdown。"""
    # [R04 原代码保留] 原入口没有处理空字符串或纯空白输出。
    # if not isinstance(raw, str):
    #     return "<UNK>"
    # [R04 修订] 在读取首行前返回原有失败标记，保持后续回退策略。
    if not isinstance(raw, str) or not raw.strip():
        return "<UNK>"
    s = raw.splitlines()[0].strip()

    # 1) 动态构建模式：支持 'HP Conf' 中的空格写法
    pat_choices = [re.escape(x).replace(r'\ ', r'\s*') for x in allowed_abbr_list]
    pat = r'\b(' + '|'.join(pat_choices) + r')\b'
    m = re.search(pat, s, flags=re.I)
    if m:
        hit = re.sub(r'\s+', ' ', m.group(1)).strip().lower()
        for a in allowed_abbr_list:
            if re.sub(r'\s+', ' ', a).strip().lower() == hit:
                return a

    # 2) markdown **HP Conf**
    m = re.search(r'\*\*\s*(' + '|'.join(pat_choices) + r')\s*\*\*', s, flags=re.I)
    if m:
        hit = re.sub(r'\s+', ' ', m.group(1)).strip().lower()
        for a in allowed_abbr_list:
            if re.sub(r'\s+', ' ', a).strip().lower() == hit:
                return a

    # 3) 兜底：letters-only 包含匹配
    letters = re.sub(r'[^A-Za-z]', '', s).lower()
    for a in allowed_abbr_list:
        if re.sub(r'[^A-Za-z]', '', a).lower() in letters:
            return a
    return "<UNK>"

# ========== 7. 模型加载 ==========
print("[INFO] Loading model …")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)





@torch.inference_mode()
def generate_once(prompt: str, max_len: int):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # [R02 原代码保留] 原调用丢弃模板text，且没有使用max_len。
    # inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    # [R02 修订] 编码完整聊天模板；模板已包含特殊token，避免二次添加。
    inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt", truncation=False)
    input_tokens = inputs["input_ids"].shape[1]
    if not isinstance(max_len, int) or max_len <= 0:
        raise ValueError("[R02] max_len必须是正整数输入预算。")
    input_budget = max_len
    context_limit = getattr(model.config, "max_position_embeddings", None)
    if isinstance(context_limit, int) and 0 < context_limit <= 1_000_000:
        input_budget = min(input_budget, context_limit - MAX_NEW_TOKENS)
    if input_tokens > input_budget:
        raise ValueError(
            f"[R02] 完整模板输入共{input_tokens} tokens，超过阶段输入预算{input_budget}；"
            "请核对max_len及模型上下文，禁止静默截断目标评论。"
        )
    inputs = inputs.to(model.device)

    outputs = model.generate(
        # [R02 原代码保留]
        # input_ids=inputs.to(model.device),
        # [R02 修订] 同时传入模板编码所得input_ids与attention_mask。
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,

    )
    # [R02 原代码保留] 原处理包含输入prompt，按字符串切分可能混入提示内容。
    # raw = tokenizer.decode(outputs[0])
    # print(raw.split("Answer:")[-1].split("<end_of_turn>")[0].strip())
    # return raw.split("Answer:")[-1].split("<end_of_turn>")[0].strip()
    # [R02 修订] 仅解码新生成token，首行处理与其他层次脚本保持一致。
    raw = tokenizer.decode(outputs[0][input_tokens:], skip_special_tokens=True).strip()
    return re.split(r"[\r\n]", raw, maxsplit=1)[0]

# ========== 8. 两阶段推理 ==========
@torch.inference_mode()
def classify_two_stage(comment: str):
    # Stage-1: coarse
    coarse_prompt = build_prompt_coarse(comment)
    coarse_raw = generate_once(coarse_prompt, max_len=COARSE_MAX_LEN)
    # print(coarse_raw)
    coarse_abbr = extract_coarse(coarse_raw)
    # print(coarse_abbr)

    # Stage-2: fine
    if coarse_abbr == "<UNK>":
        allowed = sorted(FINE_ABBR_SET)  # 粗类不明 -> 全局允许
    else:
        allowed = COARSE2FINE_ABBR.get(coarse_abbr, sorted(FINE_ABBR_SET))

    fine_prompt = build_prompt_fine(comment, coarse_abbr)
    fine_raw = generate_once(fine_prompt, max_len=FINE_MAX_LEN)
    fine_abbr = extract_fine(fine_raw, allowed)

    if fine_abbr == "<UNK>":
        # 若细类不在允许集合，回退到该粗类默认
        fine_abbr = FALLBACK_FINE_BY_COARSE.get(coarse_abbr, "Data Conf")

    return coarse_abbr, coarse_raw, fine_abbr, fine_raw

# ========== 9. 主流程 ==========
df = pd.read_csv(CSV_PATH)
rows = []
unk_coarse_cnt = 0
unk_fine_cnt   = 0

print("[INFO] Classifying (two-stage, fixed parser) …")
for cmt, gold in tqdm(zip(df["Comment Text"], df["ML TD Type"]),
                      total=len(df), ncols=80):
    coarse_abbr, coarse_raw, fine_abbr, fine_raw = classify_two_stage(cmt)

    if coarse_abbr == "<UNK>":
        unk_coarse_cnt += 1
    if fine_abbr not in FINE_ABBR_SET:
        unk_fine_cnt += 1

    rows.append({
        "comment": cmt,
        "real": FINE_ABBR2FULL.get(gold, gold),     # 细类全称（真实）
        "pred": FINE_ABBR2FULL.get(fine_abbr, fine_abbr),  # 细类全称（预测）
        # —— Stage-1 —— #
        "coarse_pred": coarse_abbr,
        "coarse_raw_output": coarse_raw,
        # —— Stage-2 —— #
        "fine_abbr": fine_abbr,
        "raw_output": fine_raw,
    })

pred_df = pd.DataFrame(rows)

# ========== 10. 评估 ==========
# 细（23）
fine_report_text = classification_report(pred_df["real"], pred_df["pred"],
                                         labels=FINE_LABELS, zero_division=0, digits=3)
print("\n[INFO] Fine-grained metrics (23):\n", fine_report_text)

# 粗（9）
pred_df["real_coarse"] = pred_df["real"].map(FINE2COARSE_FULL)
pred_df["pred_coarse"] = pred_df["pred"].map(FINE2COARSE_FULL)
coarse_report_text = classification_report(pred_df["real_coarse"], pred_df["pred_coarse"],
                                           labels=COARSE_LABELS, zero_division=0, digits=3)
print("\n[INFO] Coarse-grained metrics (9):\n", coarse_report_text)

# ========== 11. 保存 ==========
pred_df.to_csv(RUN_DIR/"predictions.csv", index=False)

pd.DataFrame(
    confusion_matrix(pred_df["real"], pred_df["pred"], labels=FINE_LABELS),
    index=FINE_LABELS, columns=FINE_LABELS
).to_csv(RUN_DIR/"confusion_matrix_fine.csv")

pd.DataFrame(
    confusion_matrix(pred_df["real_coarse"], pred_df["pred_coarse"], labels=COARSE_LABELS),
    index=COARSE_LABELS, columns=COARSE_LABELS
).to_csv(RUN_DIR/"confusion_matrix_coarse.csv")

(Path(RUN_DIR/"report_fine.txt")).write_text(fine_report_text, encoding="utf-8")
(Path(RUN_DIR/"report_coarse.txt")).write_text(coarse_report_text, encoding="utf-8")

(Path(RUN_DIR/"settings.json")).write_text(json.dumps({
    "csv": CSV_PATH,
    "model": MODEL_PATH,
    "run_dir": str(RUN_DIR),
    "max_new_tokens": MAX_NEW_TOKENS,
    # [R02 修订] 记录已执行的阶段输入预算与模板处理策略。
    "coarse_max_input_tokens": COARSE_MAX_LEN,
    "fine_max_input_tokens": FINE_MAX_LEN,
    "input_length_policy": "full chat template; over-budget inputs raise an error; no truncation",
    "two_stage": True,
    "unknown_coarse": unk_coarse_cnt,
    "unknown_fine": unk_fine_cnt
}, indent=2))

print(f"\n[INFO] Files saved in {RUN_DIR}")
print(f"[INFO] Unknown coarse predictions: {unk_coarse_cnt}")
print(f"[INFO] Unknown fine predictions:   {unk_fine_cnt}")

gc.collect(); torch.cuda.empty_cache()
