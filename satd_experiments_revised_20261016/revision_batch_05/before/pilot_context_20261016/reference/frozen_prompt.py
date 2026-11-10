import re
LABEL_ABBR = {'Machine Learning Knowledge': 'MLKnow', 'Model Interpretability': 'MLInter', 'Model Code Comprehension': 'MCC', 'Duplicate Model Code': 'DMC', 'Duplicate Feature Extraction Code': 'DFEC', 'Weight Configuration': 'Weight Conf', 'Layer Configuration': 'Layer Conf', 'Hyper-parameter Configuration': 'HP Conf', 'Code Dependency': 'MLDepend', 'Glue Code': 'GC', 'Custom Data Type': 'CDT', 'Multiple Languages': 'ML', 'Unnecessary Model Code': 'UMC', 'Data Processing Configuration': 'Data Conf', 'Plain Old Data Type': 'PODT', 'Data Storage Configuration': 'DS Conf', 'Abstraction': 'Abs', 'Boundary Erosion': 'BE', 'Model Code Modifiability': 'MCM', 'Model Code Reusability': 'MCR', 'Prediction Quality': 'PQ', 'Machine Learning Reliability': 'MLRelia', 'Prototype': 'Proto'}
ABBR2FULL = {v: k for k, v in LABEL_ABBR.items()}
LABELS = list(LABEL_ABBR.keys())
ABBR_SET = set(ABBR2FULL.keys())
ABBR_LOWER = {a.lower(): a for a in ABBR_SET}
PROMPT_HEAD = "\nThere are 23 types of Machine Learning (ML) SATD (Self-Admitted Technical Debt):\n\nMACHINE LEARNING KNOWLEDGE (MLKnow): Machine-learning software carries plenty of unique challenges. Uneducated solutions by unaware developers may have to be revisited.\n\nMODEL INTERPRETABILITY (MLInter): Machine-learning models are a black box, causing poor understanding of model functionality; this can lead to unknown behaviour.\n\nMODEL CODE COMPREHENSION (MCC): Model code carries extra legibility concerns not present in traditional software (e.g., poorly-named temporary matrix variables).\n\nDUPLICATE MODEL CODE (DMC): Code duplication frequently occurs in model code.\n\nDUPLICATE FEATURE EXTRACTION CODE (DFEC): Code duplication frequently occurs in feature-extraction code.\n\nWEIGHT CONFIGURATION (Weight Conf): Editing code that involves the weights of a model, or configuring a model's weights directly.\n\nLAYER CONFIGURATION (Layer Conf): Editing code that deals with model layers, or configuring layers themselves.\n\nHYPER-PARAMETER CONFIGURATION (HP Conf): Configuring hyper-parameters of a model, or editing default values of off-the-shelf models.\n\nCODE DEPENDENCY (MLDepend): A needed change is blocked by an external dependency (library / subsystem).\n\nGLUE CODE (GC): Supporting code written to interface with other code, inhibiting improvements due to dependent peculiarities.\n\nCUSTOM DATA TYPE (CDT): Using data types provided by general-purpose packages can cause extensive inter-operability burden.\n\nMULTIPLE LANGUAGES (ML): Components written in other languages may introduce difficulties in ML development.\n\nUNNECESSARY MODEL CODE (UMC): Model code that bottlenecks execution, is unreachable / deprecated, or is unnecessary and should be removed.\n\nDATA PROCESSING CONFIGURATION (Data Conf): Configuring how data is processed, by editing the data directly or adding new processing steps.\n\nPLAIN OLD DATA TYPE (PODT): Using raw data types in ML causes confusion when interpreting processes.\n\nDATA STORAGE CONFIGURATION (DS Conf): Configuring how data is represented internally or stored externally.\n\nABSTRACTION (Abs): Lack of abstractions in ML systems causes cascading changes when a component is modified.\n\nBOUNDARY EROSION (BE): Lack of boundaries between subsystems complicates maintenance and isolates changes poorly.\n\nMODEL CODE MODIFIABILITY (MCM): Model code should enable easy maintenance and future modifications.\n\nMODEL CODE REUSABILITY (MCR): Model code should be generalised for reuse in varying situations.\n\nPREDICTION QUALITY (PQ): Workflow changes may affect performance.\n\nMACHINE LEARNING RELIABILITY (MLRelia): Model functionality depends on data quality; robustness measures should be in place.\n\nPROTOTYPE (Proto): Deploying small-scale prototypes into full systems can be dangerous.\n".strip()
PROMPT_TAIL = f"Please determine which type of ML Self-Admitted Technical Debt (SATD) the following Code Comment belongs to. \nYou MUST answer with **only ONE** abbreviation from:\n{', '.join(sorted(ABBR_SET))}\n".strip()

def build_prompt(comment: str) -> str:
    return f'{PROMPT_HEAD}\n\n{PROMPT_TAIL}\n\nCode Comment:{comment}\n"""\n\nAbbreviation:'

def simple_extract(raw: str) -> str:
    first = re.split('[\\r\\n]', raw.strip())[0]
    token = re.sub('[^a-z]', '', first.lower())
    if token in ABBR_LOWER:
        return ABBR_LOWER[token]
    raw_strip = first.strip()
    if raw_strip in ABBR2FULL:
        return raw_strip
    join = raw_strip.replace(' ', '')
    if join.lower() in ABBR_LOWER:
        return ABBR_LOWER[join.lower()]
    return '<UNK>'
