#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_satd_qwen3_yn_polling.py
----------------------------
模型：/data/yangzhifei/pro/ML_SATD/models/codegemma-1.1-7b-it
方法：对每条评论，对 23 标签逐一做 Yes/No，汇总 Yes 候选；如有多个，再让模型在候选中只选 1 个缩写；
      若没有 Yes，则做一次全标签强制单标签输出；仍解析失败则回退到 FALLBACK_ABBR。
输出：results/run-YYYYmmdd-HHMMSS-模型名 / predictions.csv, confusion_matrix.csv, report.csv, settings.json
"""

import re, os, gc, json
from pathlib import Path
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
os.environ["TORCHDYNAMO_DISABLE"] = "1"  # 禁用 torch dynamo
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_float32_matmul_precision('high')
# -------- 0. 基础路径（仅此处改） --------
os.environ["CUDA_VISIBLE_DEVICES"] = "1"        # GPU 号
CSV_PATH   = "/data/yangzhifei/pro/ML_SATD/dataset/sampled_label_dataset_normalized_rmvtodo.csv"  # ← 修改
MODEL_PATH = "/data/yangzhifei/pro/ML_SATD/models/codegemma-1.1-7b-it"                                       # ← 修改

# 目录名包含时间戳 + 模型名
MODEL_NAME = Path(MODEL_PATH).name
RUN_DIR    = Path("results") / f"run-{datetime.now():%Y%m%d-%H%M%S}-{MODEL_NAME}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

# -------- 1. 其它参数 --------
PRINT_UNPARSED = True
FALLBACK_ABBR  = "Data Conf"
MAX_YN_TOKENS  = 8     # Yes/No 判定长度很短
MAX_PICK_TOKENS= 64     # 候选中选择 1 个缩写
MAX_FALLBACK_TOKENS = 32

# ----------------- 标签映射 -----------------
LABEL_ABBR = {
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
ABBR2FULL  = {v:k for k,v in LABEL_ABBR.items()}
FULL2ABBR  = {k:v for k,v in LABEL_ABBR.items()}
LABELS     = list(LABEL_ABBR.keys())
ABBR_SET   = set(ABBR2FULL.keys())
ABBR_LOWER = {a.lower():a for a in ABBR_SET}
ABBR_REGEX = re.compile(r"\b(" + "|".join(re.escape(a) for a in ABBR_SET) + r")\b", re.I)

# ============ 2. Prompt 头（仅描述，无示例） ============
# PROMPT_HEAD = r"""
# There are 23 types of Machine Learning (ML) SATD (Self-Admitted Technical Debt):

# MACHINE LEARNING KNOWLEDGE (MLKnow): Machine-learning software carries plenty of unique challenges. Uneducated solutions by unaware developers may have to be revisited.
# examples:
# 1. `TODO: Confirm whether layer-norm before attention actually boosts convergence for small-batch training on this dataset`
# 2. `TODO: Figure out if label smoothing distorts precision-recall when classes are highly imbalanced`
# 3. `XXX: Revisit our choice of cosine annealing—does it still make sense once we switch to AdamW?`

# MODEL INTERPRETABILITY (MLInter): Machine-learning models are a black box, causing poor understanding of model functionality; this can lead to unknown behaviour.
# examples:
# 1. `TODO: Generate SHAP summary plot so users can see feature impact`
# 2. `TODO: Log attention weights for each prediction to aid debugging`
# 3. `XXX: Provide saliency-map overlay in evaluation notebook for misclassified images`

# MODEL CODE COMPREHENSION (MCC): Model code carries extra legibility concerns not present in traditional software (e.g., poorly-named temporary matrix variables).
# examples:
# 1. `TODO: break up monolithic function train_eval_predict() into clearer, single-purpose helpers`
# 2. `FIXME: variable “h” is reused for both hidden state and Hessian—pick distinct, descriptive names`
# 3. `XXX: replace magic constant 512 (actually EMBED_SIZE) with a config parameter or enum`

# DUPLICATE MODEL CODE (DMC): Code duplication frequently occurs in model code.
# examples:
# 1. `TODO: remove copy-pasted ConvBlock definition—reuse common_layers.ConvBlock instead`
# 2. `FIXME: forward() in MyResNet duplicates logic from BaseResNet; refactor to call super()`
# 3. `XXX: data_augmentation_v2() is 95 % identical to data_augmentation(); consolidate into one helper`

# DUPLICATE FEATURE EXTRACTION CODE (DFEC): Code duplication frequently occurs in feature-extraction code.
# examples:
# 1. `TODO: merge extract_mfcc() and extract_mfcc_fast(); only differ in hop_length`
# 2. `FIXME: pixel_stats_v2() repeats almost all logic from pixel_stats(); refactor into shared util`
# 3. `XXX: text_features_v1 and text_features_v2 are copy-pasted—create one configurable pipeline`

# WEIGHT CONFIGURATION (Weight Conf): Editing code that involves the weights of a model, or configuring a model's weights directly.
# examples:
# 1. `TODO: expose weight_decay as configurable argument instead of hard-coding 0.01`
# 2. `FIXME: pretrained checkpoint loads fp16 weights—add auto-cast to ensure dtype consistency`
# 3. `XXX: add toggle to freeze first two Transformer layers during fine-tune via gradient mask`

# LAYER CONFIGURATION (Layer Conf): Editing code that deals with model layers, or configuring layers themselves.
# examples:
# 1. `TODO: swap BatchNorm for LayerNorm when depth > 24`
# 2. `FIXME: insert positional-embedding layer before first Transformer block`
# 3. `XXX: make dropout rate per-layer configurable instead of using global 0.1`

# HYPER-PARAMETER CONFIGURATION (HP Conf): Configuring hyper-parameters of a model, or editing default values of off-the-shelf models.
# examples:
# 1. `TODO: make learning_rate schedule (cosine vs. linear) selectable via CLI flag`
# 2. `FIXME: batch_size is hard-coded to 32; derive from GPU memory or user input`
# 3. `XXX: expose early_stopping_patience in config.yaml instead of magic value 10`

# CODE DEPENDENCY (MLDepend): A needed change is blocked by an external dependency (library / subsystem).
# examples:
# 1. `TODO: bump scikit-learn to ≥1.6 once conda forge wheels for arm64 become available`
# 2. `FIXME: delete custom tokenizer patch after huggingface/transformers merges PR #42812`
# 3. `XXX: keep legacy CUDA 11 build until upstream faiss releases official 12.x binaries`

# GLUE CODE (GC): Supporting code written to interface with other code, inhibiting improvements due to dependent peculiarities.
# examples:
# 1. `TODO: remove pandas-to-torch bridge once DataLoader accepts DataFrame directly`
# 2. `FIXME: adapter.py only exists to map old Predictor API to new one—rewrite consumers instead`
# 3. `XXX: consolidate fetch_from_s3_and_cache(); purely glue between boto3 and our local cache manager`

# CUSTOM DATA TYPE (CDT): Using data types provided by general-purpose packages can cause extensive inter-operability burden.
# examples:
# 1. `TODO: replace custom SparseTensorDict with standard torch.sparse_coo_tensor before exporting`
# 2. `FIXME: graph_utils.Node stores metadata in AttrDict; breaks ONNX serialization—use plain dict`
# 3. `XXX: pipeline returns dask DataFrame, but Keras expects NumPy array—add type-conversion layer`

# MULTIPLE LANGUAGES (ML): Components written in other languages may introduce difficulties in ML development.
# examples:
# 1. `TODO: port custom attention kernel from CUDA C++ to Metal for macOS deployment`
# 2. `FIXME: keep tokenizer logic in sync between Rust CLI and Python training script—generate bindings instead`
# 3. `XXX: consolidate duplicate data_cleaning.{scala,py}; two implementations drift out of spec easily`

# UNNECESSARY MODEL CODE (UMC): Model code that bottlenecks execution, is unreachable / deprecated, or is unnecessary and should be removed.
# examples:
# 1. `TODO: delete legacy compute_grad_stats(); superseded by torch.autograd.profiler`
# 2. `FIXME: remove flag use_old_preprocess once new pipeline is stable`
# 3. `XXX: drop debug print statements inside training loop before release`

# DATA PROCESSING CONFIGURATION (Data Conf): Configuring how data is processed, by editing the data directly or adding new processing steps.
# examples:
# 1. `TODO: add configurable pipeline step to down-sample majority class before batching`
# 2. `FIXME: current JSON parser drops NaNs—update to preserve missing values for later imputation`
# 3. `XXX: expose window_size and stride in config.yaml so sliding-window generator isn’t hard-coded`

# PLAIN OLD DATA TYPE (PODT): Using raw data types in ML causes confusion when interpreting processes.
# examples:
# 1. `TODO: replace raw list of logits with a NamedTuple so callers know the order`
# 2. `FIXME: function returns plain dict; define a TypedConfig dataclass instead for clarity`
# 3. `XXX: passing labels as bare NumPy array—wrap in LabelsTensor to preserve shape metadata`

# DATA STORAGE CONFIGURATION (DS Conf): Configuring how data is represented internally or stored externally.
# examples:
# 1. `TODO: migrate checkpoint format from .pth to safetensors for safer serialization`
# 2. `FIXME: results_writer currently appends CSV rows one by one—buffer and batch-flush to reduce I/O`
# 3. `XXX: store intermediate feature maps in mmap file to avoid exceeding GPU memory`

# ABSTRACTION (Abs): Lack of abstractions in ML systems causes cascading changes when a component is modified.
# examples:
# 1. `TODO: extract common scoring logic into BaseMetric so each model wrapper isn’t re-implementing it`
# 2. `FIXME: decouple data_loader from Trainer—should accept any iterable, not hard-coded DataLoader`
# 3. `XXX: replace direct calls to torch.cuda.* with a HardwareBackend interface to support TPU`

# BOUNDARY EROSION (BE): Lack of boundaries between subsystems complicates maintenance and isolates changes poorly.
# examples:
# 1. `TODO: migrate training-time logging out of Trainer class; should live in metrics.logger`
# 2. `FIXME: data_loader currently triggers augmentations inline—move to separate preprocess pipeline`
# 3. `XXX: remove direct DB writes from evaluate.py; expose results through results_api instead`

# MODEL CODE MODIFIABILITY (MCM): Model code should enable easy maintenance and future modifications.
# examples:
# 1. `TODO: move hyper-parameter defaults into config file so changing them doesn’t require code edits`
# 2. `FIXME: hard-coded embedding_dim (256) sprinkled across modules—define once in constants.py`
# 3. `XXX: split run_experiment.py into cli entrypoint and reusable Experiment class for easier reuse`

# MODEL CODE REUSABILITY (MCR): Model code should be generalised for reuse in varying situations.
# examples:
# 1. `TODO: abstract make_encoder() so VisionTransformer and ResNet builders share the same backbone factory`
# 2. `FIXME: refactor evaluate_batch() into mix-in class so both Trainer and HyperTune reuse identical logic`
# 3. `XXX: remove hard-coded Softmax loss; accept any callable loss_fn to support regression and ranking tasks`

# PREDICTION QUALITY (PQ): Workflow changes may affect performance.
# examples:
# 1. `TODO: rerun validation after enabling mixed-precision; confirm accuracy isn’t degraded`
# 2. `FIXME: measure F1 change when we swap tokenizer to WordPiece—may hurt rare-word recall`
# 3. `XXX: compare BLEU before/after removing dropout layer to ensure translation quality holds`

# MACHINE LEARNING RELIABILITY (MLRelia): Model functionality depends on data quality; robustness measures should be in place.
# examples:
# 1. `TODO: add sanity-check to reject examples with >20 % missing features before prediction`
# 2. `FIXME: wrap model call in try/except and fall back to last-known good output when decoder crashes`
# 3. `XXX: create robustness test suite with Gaussian noise and FGSM perturbations to monitor accuracy drift`

# PROTOTYPE (Proto): Deploying small-scale prototypes into full systems can be dangerous.
# examples:
# 1. `TODO: replace the single-threaded demo server with a production-ready Gunicorn setup`
# 2. `FIXME: temp JSON files used for state—migrate to proper Redis cache before rollout`
# 3. `XXX: remove hard-coded API key left from prototype; load securely from env vars instead`
# """.strip()

PROMPT_HEAD = r"""
There are 23 types of Machine Learning (ML) SATD (Self-Admitted Technical Debt):

MACHINE LEARNING KNOWLEDGE (MLKnow): Machine-learning software carries plenty of unique challenges. Uneducated solutions by unaware developers may have to be revisited.
examples:
1. TODO: confirm normalization scheme (per-channel mean/std from train only); code currently computes on full dataset.
2. TODO: for time series, disable random shuffle; current DataLoader breaks temporal order.
3. TODO: compute AUROC/AUPRC in addition to accuracy; metrics must reflect class imbalance.
4. TODO: split by user/session to prevent cross-contamination; random split inflates scores.
5. TODO: add warmup steps before cosine decay; initial loss spikes suggest under-warming.
6. TODO: handle unseen categories at inference; map to OOV instead of failing.
7. TODO: freeze backbone for first N epochs; full fine-tune on small data overfits.
8. TODO: calibrate probabilities (temperature scaling/Platt); thresholds unreliable now.
9. TODO: make runs deterministic (seed CUDA + workers + hashing); results still vary.
10. XXX: AMP with gradient accumulation — verify unscale/clip/update order; may be incorrect.

MODEL INTERPRETABILITY (MLInter): Machine-learning models are a black box, causing poor understanding of model functionality; this can lead to unknown behaviour.
examples:
1. TODO: add SHAP feature attributions to eval report; predictions are opaque right now.
2. FIXME: log top contributing features per instance; need human-readable rationale.
3. TODO: publish a model card (data summary, metrics, caveats, intended use); none exists.
4. XXX: train a surrogate decision tree to approximate the black-box and sanity-check rules.
5. TODO: run counterfactual tests (swap/remove sensitive tokens) to expose spurious correlations.
6. TODO: generate Partial Dependence/ICE plots for key features to visualize effects.
7. FIXME: attention weights ≠ explanations; validate importance via feature ablation.
8. TODO: add monotonicity checks (e.g., risk_score should not increase with higher income); alert on violations.
9. XXX: compute per-slice metrics (by cohort/device/locale) to reveal hidden failure modes.
10. TODO: persist prediction traces (inputs, transforms, logits, threshold) for auditability and error analysis.

MODEL CODE COMPREHENSION (MCC): Model code carries extra legibility concerns not present in traditional software (e.g., poorly-named temporary matrix variables).
examples:
1. TODO: replace single-letter tensors (u,v,w) with descriptive names; current shapes are unclear.
2. FIXME: magic strides [1,2,1,1] hardcoded in conv stack; extract to named config for readability.
3. TODO: forward() both mutates self.cache and returns logits; split side effects to clarify flow.
4. TODO: rename "ff"→"feedforward", "mh"→"multihead_attn"; abbreviations obscure intent.
5. TODO: break 200-line train_step into smaller helpers (compute_loss/log_metrics/apply_grad); too hard to scan.
6. XXX: flag "sc" toggles sigmoid vs softmax but name is cryptic; use "use_softmax".
7. TODO: add shape annotations in docstrings: x [B,T,C], mask [B,1,T,T], logits [B,T,V].
8. FIXME: inconsistent dim order (NCHW vs NHWC) across modules; standardize or comment conversions.
9. TODO: replace nested list comps for attention mask construction with clear loops + comments.
10. TODO: name loss weights (alpha_ce=0.9, beta_l2=0.01) instead of inline 0.9*CE + 0.01*L2.

DUPLICATE MODEL CODE (DMC): Code duplication frequently occurs in model code.
examples:
1. TODO: factor out identical Conv+BN+ReLU blocks from Encoder and Decoder; avoid copy-paste.
2. FIXME: train_step() duplicated in BaseTrainer and CustomTrainer; extract shared mixin.
3. TODO: merge two implementations of positional_encoding() (transformer.py vs utils/pe.py).
4. XXX: attention mask builder copy-pasted in bert.py and gpt.py; centralize in masking.py.
5. TODO: dedupe LayerNorm init (bias=False, eps=1e-5) repeated across four modules.
6. TODO: consolidate predict() and infer() paths — 90% identical besides logging.
7. FIXME: label_smoothing_ce defined in losses.py and heads/classifier.py; keep one.
8. TODO: unify ResBlockV1 and ResBlockV2; only activation differs (ReLU vs GELU).
9. TODO: F1 computation reimplemented in evaluator.py; import metrics.f1 instead.
10. TODO: dropout scheduling logic duplicated in three schedulers; extract helper.

DUPLICATE FEATURE EXTRACTION CODE (DFEC): Code duplication frequently occurs in feature-extraction code.
examples:
1. TODO: dedupe TF-IDF vectorizer setup (ngram/range/min_df) repeated in baseline.py and train.py.
2. FIXME: compute_mel_spectrogram() duplicated in audio/extract.py and preprocess/audio.py; keep single source.
3. TODO: consolidate image preprocess (resize→center_crop→normalize) copy-pasted in dataset_a.py and dataset_b.py into transforms.py.
4. XXX: tokenization + truncation + attention_mask building duplicated in text_dataset.py and infer.py; extract build_inputs().
5. TODO: standardize numerical features (fit scaler on train only) duplicated in tabular/prep.py and model/train.py; centralize.
6. FIXME: identical categorical encoding (OneHot with rare-bucket) exists in preprocess.py and feature_engineering.py; merge.
7. TODO: factor out sliding-window/overlap logic for time-series in signals/extract.py and sensors/pipeline.py.
8. XXX: graph feature construction (k-hop neighbors, degree, centrality) duplicated in gnn_utils.py and graph_features.py; unify.
9. TODO: handcrafted temporal features (rolling mean/lag diff/exp smoothing) scattered across notebooks; move to features/temporal.py.
10. XXX: dimensionality reduction (PCA→UMAP) blocks reimplemented in three pipelines; create features/dimensionality.py.

WEIGHT CONFIGURATION (Weight Conf): Editing code that involves the weights of a model, or configuring a model's weights directly.
examples:
1. TODO: tie decoder lm_head weights to token embedding (weight tying) to reduce params.
2. FIXME: reinitialize classifier head weights/bias after loading pretrained backbone; current head retains stale weights.
3. TODO: set per-parameter weight_decay; exclude LayerNorm/BatchNorm/bias from decay.
4. XXX: apply Kaiming/Xavier init for Conv/Linear; document fan_in/fan_out to avoid silent misinit.
5. TODO: add freeze_backbone option (requires setting requires_grad=False for all backbone weights for first N epochs).
6. FIXME: AMP training should update FP32 master weights via GradScaler; verify unscale/step order.
7. TODO: load partial state_dict with strict=False + key mapping; missing keys should trigger explicit warnings.
8. TODO: implement Exponential Moving Average (EMA) of model weights for more stable evaluation.
9. TODO: add Stochastic Weight Averaging (SWA) checkpointing/restore; keep separate averaged weight vector.
10. XXX: implement per-channel int8 weight quantization for inference; store scale/zero_point with state_dict.

LAYER CONFIGURATION (Layer Conf): Editing code that deals with model layers, or configuring layers themselves.
examples:
1. TODO: replace BatchNorm2d with GroupNorm(num_groups=32); tiny batch sizes make BN stats noisy.
2. FIXME: Transformer blocks should be pre-norm, not post-norm; switch LayerNorm placement for stability.
3. TODO: make Conv stem stride/padding configurable; fixed stride=2 breaks small input resolutions.
4. XXX: set bias=False on conv/linear when followed by norm; redundant parameters otherwise.
5. TODO: add residual path around decoder cross-attention to mirror encoder design.
6. TODO: set head_dim=64 and adjust num_heads accordingly to keep hidden_size constant.
7. FIXME: FFN activation should be GELU (approx) instead of ReLU; update all blocks.
8. TODO: replace 3×3 conv with depthwise-separable conv (dw+pw) to reduce params while keeping RF.
9. TODO: move dropout after residual add (a.k.a. DropPath style), not before MLP/attention.
10. XXX: Embedding layer missing padding_idx; pad tokens should be masked from training updates.

HYPER-PARAMETER CONFIGURATION (HP Conf): Configuring hyper-parameters of a model, or editing default values of off-the-shelf models.
examples:
1. TODO: expose learning_rate via CLI/config; remove hardcoded 3e-4.
2. TODO: choose one of warmup_steps or warmup_ratio; current code mixes both.
3. FIXME: scheduler stepping at epoch vs step — verify for cosine schedule with warmup.
4. TODO: keep global_batch_size fixed via per_device_batch_size × grad_accum_steps.
5. XXX: tune dropout rate; set default 0.3 and log to config.
6. TODO: add weight_decay sweep (0, 1e-4, 5e-4, 1e-3) for regularization.
7. TODO: expose label_smoothing (ε) for CE loss; default 0.1.
8. TODO: make max_seq_length configurable (128/256/512); affects memory/perf.
9. FIXME: early_stopping monitor/patience mismatch; monitor 'val_f1' instead of 'val_loss'.
10. TODO: add gradient_clipping max_norm as hyperparam; default 1.0.

CODE DEPENDENCY (MLDepend): A needed change is blocked by an external dependency (library / subsystem).
examples:
1. TODO: drop custom attention kernel once flash-attn is available for our PyTorch/CUDA combo on the cluster.
2. FIXME: keep transformers pinned (<4.44); generation API regression upstream — unpin after fix lands.
3. TODO: replace local NMS with torchvision.ops.nms when CUDA wheels are published for our driver version.
4. XXX: ONNX export lacks dynamic-shape support for this op; keep static workaround until exporter adds it.
5. TODO: upgrade to cuDNN 9 to enable bf16 conv; blocked by CI GPU driver compatibility.
6. TODO: move to Hydra Structured Configs after base image updates to Hydra ≥1.3; current runtime too old.
7. FIXME: can't adopt Python 3.11 yet — internal toolkit only supports 3.10; revert to typing_extensions for now.
8. TODO: switch to sklearn’s new CalibratedClassifier when multi-label is supported; retain custom cal step until then.
9. XXX: use triton-optimized softmax once Windows wheels are released; fallback to torch implementation for now.
10. TODO: remove matplotlib import side-effect once headless backend is default on the cluster nodes.

GLUE CODE (GC): Supporting code written to interface with other code, inhibiting improvements due to dependent peculiarities.
examples:
1. TODO: replace hand-rolled JSON→dict→dataclass marshaling with a proper schema; current glue hides real validation errors.
2. FIXME: adapter translating legacy predict(X)->y to new predict({"inputs": X}) is brittle; refactor once clients migrate.
3. TODO: remove pandas→numpy→torch hop; implement Dataset.from_pandas to avoid double copies in the glue layer.
4. XXX: temporary BGR↔RGB shim for cv2 vs PIL expectations; delete after image stack is unified.
5. TODO: wrap third-party logger to our Trainer callback interface; stop forwarding **kwargs blindly.
6. FIXME: REST client builds query strings by string concat; switch to official SDK when auth issue is resolved.
7. TODO: normalize time units (ms vs s) across services; current glue multiplies/divides by magic constants.
8. XXX: JSON↔Protobuf bridge for model serving; remove once server accepts Protobuf natively.
9. TODO: monkey-patch library to expose logits; drop patch after upstream PR merges.
10. TODO: move IO packing/unpacking out of pipeline.forward(); keep only model logic and place adapters under io/bridge.py.

CUSTOM DATA TYPE (CDT): Using data types provided by general-purpose packages can cause extensive inter-operability burden.
examples:
1. TODO: stop passing pandas.DataFrame across model boundaries; define FeatureBatch (X: np.ndarray[float32], y: np.ndarray) and convert once.
2. FIXME: torch.float32 ↔ sklearn.float64 mismatch; unify dtype at the ingress to avoid silent casts everywhere.
3. TODO: replace numpy.recarray/structured dtypes with flat 2D arrays; downstream libs don't handle fielded dtypes well.
4. XXX: scipy.sparse.csr not supported by our torch modules; convert to coo or dense at a single boundary.
5. TODO: avoid pandas.Timestamp/Timedelta in features; convert to POSIX int64 seconds early.
6. FIXME: returning dict[str, Any] of tensors is fragile; use a TypedDict/dataclass schema to make types explicit.
7. TODO: standardize categorical representation (pandas.Categorical vs string) before vectorization; current mix breaks encoders.
8. XXX: np.bool_ vs torch.bool inconsistencies cause mask errors; cast masks to a single canonical bool type.
9. TODO: move from DataFrame→Arrow→NumPy roundtrips to a single pyarrow.Table contract; cut conversion glue.
10. FIXME: numpy types (np.int64/np.float32) not JSON-serializable; sanitize via .item()/.tolist() at the API boundary.

MULTIPLE LANGUAGES (ML): Components written in other languages may introduce difficulties in ML development.
examples:
1. TODO: move hot path to C++ via pybind11; keep Python fallback until manylinux wheels are ready.
2. FIXME: custom CUDA op only built for sm_80; add sm_70/sm_75 or guard at runtime.
3. TODO: rewrite tokenizer in Rust (pyo3) to replace slow Python regex; FFI boundary is messy for now.
4. XXX: TorchScript op implemented in Python — need C++ extension for mobile; interpreter import fails.
5. TODO: add JNI bridge for Java inference server; current py4j workaround crashes under multithreading.
6. TODO: integrate R feature pipeline with Python trainer via Arrow IPC; CSV roundtrip loses dtypes.
7. FIXME: Scala Spark UDF returns Map<String,Double> but Python expects dict[str,float]; define explicit schema.
8. XXX: CoreML export needs Swift pre/post wrapper; Python preprocessing duplicated — unify in Swift.
9. TODO: implement ONNX Runtime custom op in C++; Python simulation too slow for production.
10. FIXME: Cython memoryviews incur implicit copies due to stride mismatch; enforce contiguous layout or zero-copy API.

UNNECESSARY MODEL CODE (UMC): Model code that bottlenecks execution, is unreachable / deprecated, or is unnecessary and should be removed.
examples:
1. TODO: remove legacy fast_path(); unreachable since v2 API landed.
2. FIXME: drop manual softmax implementation; use torch.nn.functional.softmax instead.
3. TODO: delete debug print block in forward(); floods logs and slows training.
4. XXX: old 'use_fp16' flag duplicates AMP; remove to avoid confusion.
5. TODO: prune dead code in _build_optimizer(); SGD branch never used.
6. TODO: deprecate custom DataParallel wrapper; DistributedDataParallel supersedes it.
7. FIXME: vectorize IoU loop; pure-Python nested loops are a hotspot with no added value.
8. TODO: remove fallback for PyTorch<1.10; min supported version is 2.x.
9. XXX: eliminate redundant to(device) calls inside model; handled by Trainer already.
10. TODO: delete serialization shim (pickle_v4); torch.save covers this now.

DATA PROCESSING CONFIGURATION (Data Conf): Configuring how data is processed, by editing the data directly or adding new processing steps.
examples:
1. TODO: rebuild train/val/test with GroupKFold(user_id) + stratify(y); current random split leaks sessions.
2. FIXME: image normalize with ImageNet mean/std (0.485,0.456,0.406 / 0.229,0.224,0.225); stop min–max scaling to [0,1].
3. TODO: standardize numeric features via z-score fit on train only; persist stats for val/test/infer.
4. XXX: impute missing — numeric→median by site_id; categorical→"__UNK__"; avoid dropping rows.
5. TODO: deduplicate samples by content hash after lowercasing and whitespace normalization; duplicates inflate metrics.
6. TODO: text cleaning pipeline: Unicode NFKC, lowercase, strip URLs/@handles, collapse spaces; keep emojis.
7. FIXME: convert all timestamps to UTC and sort before windowing; DST made windows misaligned.
8. TODO: handle class imbalance — use WeightedRandomSampler or per-sample weights in DataLoader.
9. TODO: enable audio augs (time-shift, gaussian-noise, SpecAugment) via config flags; seed for reproducibility.
10. XXX: cap outliers at p99 and apply log1p to skewed features ['amount','duration']; document in config.

PLAIN OLD DATA TYPE (PODT): Using raw data types in ML causes confusion when interpreting processes.
examples:
1. TODO: replace nested Python lists with np.ndarray(float32); current code assumes rectangular shape but gets ragged.
2. FIXME: passing raw dict of features → key order non-deterministic; use dataclass/NamedTuple with explicit fields.
3. TODO: avoid pandas 'object' dtype for mixed columns; cast to categorical/int64/float32 explicitly.
4. XXX: timestamps stored as strings "YYYY-MM-DD" — parse to int64 epoch or pandas.DatetimeIndex before processing.
5. TODO: convert label strings {"yes","no"} to numeric {1,0}; metrics expect numeric types.
6. FIXME: using None to mark missing numerics; prefer np.nan with proper mask to avoid type upcasting.
7. TODO: standardize booleans to true bools; dataset mixes {0,1,"true","false"} causing silent coercions.
8. XXX: bytes vs str confusion — decode incoming bytes to UTF-8 str at the data boundary.
9. TODO: stop storing vectors as comma-separated strings; use fixed-length arrays (np.ndarray) per row.
10. TODO: replace tuple boxes (x,y,w,h) with typed Box(width,height,center) struct; tuple indexing is error-prone.

DATA STORAGE CONFIGURATION (DS Conf): Configuring how data is represented internally or stored externally.
examples:
1. TODO: move dataset from CSV to Parquet/Arrow with explicit schema; current CSV loses dtypes and is slow.
2. FIXME: features saved as float64 by default; cast to float32/int32 to reduce memory footprint.
3. TODO: represent sparse features as CSR/COO instead of dense ndarrays; update loaders accordingly.
4. TODO: introduce primary key sample_id and add index; current joins on non-keys create duplicate rows.
5. XXX: shard training data by dt=YYYY-MM-DD partitions with zstd compression; single monolithic file is too large.
6. TODO: replace directory-of-images with WebDataset (.tar shards) or LMDB for random/random-access IO.
7. FIXME: feature store mixes snake_case and camelCase; standardize column naming and add migration.
8. TODO: persist schema_version + preprocessing stats (means/stds/categories) alongside tensors (metadata.json).
9. XXX: enable memory-mapped arrays (np.memmap) for large embeddings; avoid full RAM load.
10. TODO: standardize time columns to UTC int64 epoch; store tz/units in dataset metadata for clarity.

ABSTRACTION (Abs): Lack of abstractions in ML systems causes cascading changes when a component is modified.
examples:
1. TODO: introduce a Predictor interface; changing model.forward(...) currently forces edits in trainer/evaluator/serving.
2. FIXME: data loading is hand-wired per script; add a DataSource abstraction so swapping CSV→Parquet doesn’t break pipelines.
3. TODO: replace if/elif loss selection scattered across modules with a registry/factory (name→loss); reduces ripple edits.
4. XXX: metrics computed ad-hoc in loops; define a Metrics API so adding macro-F1 doesn’t touch every trainer.
5. TODO: unify transforms via a Compose/Transform API; augmentations duplicated per dataset cause cascading changes.
6. FIXME: LR scheduler logic is baked into Trainer; extract SchedulerStrategy to switch cosine/step/warmup without refactors.
7. TODO: create a CheckpointIO interface; save/load code copy-pasted in 6 places breaks when we add EMA weights.
8. XXX: move preprocessing constants into a typed Config schema; constants spread across modules cause drift on updates.
9. TODO: define a FeaturePipeline abstraction; adding a new feature currently requires edits in featurizer/dataloader/model heads.
10. FIXME: serving backends each implement their own pre/post; introduce an InferenceAdapter so outputs stay stable across runtimes.

BOUNDARY EROSION (BE): Lack of boundaries between subsystems complicates maintenance and isolates changes poorly.
examples:
1. TODO: model.forward() writes checkpoints to disk; move IO to trainer — side effects bleed across module boundaries.
2. FIXME: preprocessing mutates global config used by trainer; break the hidden coupling with explicit params.
3. TODO: serving imports training-only augmentations; split runtime packages to avoid cross-layer dependencies.
4. XXX: metrics module queries the production DB directly; consume logged events via a clean interface instead.
5. TODO: optimizer reads Dataset.num_steps; pass schedule from trainer instead of reaching across layers.
6. FIXME: dataset applies label smoothing during __getitem__; this is model concern — restore read-only samples.
7. TODO: callbacks reach into model._private buffers; expose events/hooks API instead of poking internals.
8. XXX: CLI parses flags inside library modules; parse in entrypoint and pass a typed Config object.
9. TODO: feature engineering runs inside model.forward(); move to Preprocess pipeline to keep model pure.
10. FIXME: shared singleton registry used by train and serve causes hidden state leakage; isolate scopes per subsystem.

MODEL CODE MODIFIABILITY (MCM): Model code should enable easy maintenance and future modifications.
examples:
1. TODO: remove hard-coded dims (hidden_size=768, num_classes=23); read from config to ease future model variants.
2. FIXME: add type hints + dataclass config for __init__ args; validate to prevent silent shape mismatches.
3. TODO: split monolithic Model into Encoder/Head/SchedulerStrategy; favor composition to enable drop-in swaps.
4. XXX: introduce HeadFactory (name→nn.Module) so adding a new task head doesn’t touch trainer/evaluator code.
5. TODO: expose model hyperparams via YAML/CLI (learning_rate, dropout, activation); avoid editing source for tweaks.
6. TODO: add forward-hooks/callback API (on_before_forward/on_after_forward) to support feature dumping without patching code.
7. FIXME: stabilize public methods (fit/predict/predict_proba/save/load) and deprecate legacy names with warnings.
8. TODO: write unit tests for tensor shape contracts (B,T,C) and serialization round-trips to catch regressions early.
9. TODO: move device/precision management (cpu/cuda/amp) into a utility; model code should be device-agnostic.
10. XXX: document extension points in README (custom loss/optimizer/scheduler) and provide minimal templates for quick mods.

MODEL CODE REUSABILITY (MCR): Model code should be generalised for reuse in varying situations.
examples:
1. TODO: remove fixed image size (224); use adaptive pooling to accept arbitrary H×W inputs.
2. FIXME: first conv assumes in_channels=3; parameterize in_channels to support grayscale/multispectral.
3. TODO: decouple backbone from head — expose Encoder/Head registry so CNN/Transformer backbones can be swapped.
4. XXX: loss function hardwired to CrossEntropy; allow pluggable losses (CE/BCE/Focal/CTC) via config.
5. TODO: generalize metric pipeline; support classification/segmentation/regression by registering metric sets.
6. TODO: make normalization layer selectable (BatchNorm/GroupNorm/LayerNorm) with a flag instead of hardcoding BN.
7. FIXME: tokenizer coupled to a single model; accept any HF tokenizer through an interface for reuse across NLP tasks.
8. TODO: export adapters for TorchScript/ONNX; keep forward signatures stable for deployment reuse.
9. TODO: implement shape-agnostic heads using nn.AdaptiveAvgPool and global pooling to reduce assumptions.
10. XXX: unify batch and streaming inference paths behind a Predictor API; reuse the same model in offline/online settings.

PREDICTION QUALITY (PQ): Workflow changes may affect performance.
examples:
1. TODO: re-benchmark macro-F1 after new text cleaning; scores dipped post v2 pipeline.
2. FIXME: AMP enabled caused AUPRC drop; run ablation (fp32 vs amp) to confirm regression.
3. TODO: retune lr/weight_decay after adding strong augs; prior hyperparams likely suboptimal.
4. XXX: high run-to-run variance; report mean±std over 5 seeds before claiming gains.
5. TODO: recalibrate thresholds (ROC/PR sweep) after label smoothing; precision@k shifted.
6. FIXME: new GroupKFold split not comparable to old random split; recompute all baselines.
7. TODO: check distribution shift in 2025-09 data snapshot; refresh normalization stats.
8. XXX: verify int8 quantization accuracy delta <1%; log pre/post top-1 and latency.
9. TODO: compare tokenizer v2 vs v1 (truncation rate, OOV, F1); suspected recall loss.
10. TODO: ablate early stopping removal; monitor overfitting (train↑, val↓) and final test perf.

MACHINE LEARNING RELIABILITY (MLRelia): Model functionality depends on data quality; robustness measures should be in place.
examples:
1. TODO: validate input schema (required columns/types/ranges) and reject bad batches; pipeline currently trusts data.
2. FIXME: guard against NaN/Inf in features/labels; add clamp/impute to stop silent propagation.
3. TODO: implement OOD detection (max-softmax/Mahalanobis) and abstain when confidence < τ.
4. XXX: add adversarial robustness tests (FGSM/PGD/synonym attacks) in CI; report worst-case metrics.
5. TODO: monitor data drift via PSI/KS per feature and trigger alerts when drift > threshold.
6. FIXME: calibrate probabilities (ECE/MCE) and watch calibration drift in production dashboards.
7. TODO: add invariance tests (feature permutation, unit scaling, token order) to catch brittle behavior.
8. TODO: define safe ranges for sensor/text length; clip/sanitize outliers and drop impossible timestamps.
9. XXX: implement low-confidence fallback (human-in-the-loop or rule baseline) to avoid risky auto-actions.
10. TODO: shadow deploy + canary (5% traffic) and auto-rollback on error/latency/quality regressions.

PROTOTYPE (Proto): Deploying small-scale prototypes into full systems can be dangerous.
examples:
1. TODO: PoC uses in-memory dict as cache; replace with Redis/TTL before enabling multi-worker.
2. FIXME: demo ETL shell script glues CSV→model; reimplement as idempotent Airflow DAG with retries.
3. TODO: API runs on Flask dev server; move to gunicorn/uvicorn with timeouts, health checks, and graceful shutdown.
4. XXX: prototype loads model on every request; add global model pool + cold-start warmup.
5. TODO: hardcoded secrets in notebook; switch to env vars + secret manager before deploy.
6. FIXME: “quick” preprocessing path differs from training; unify pipeline to avoid train/serve skew.
7. TODO: using SQLite for demo metrics; migrate to Postgres with schema + migrations before scale.
8. XXX: batch inference is single-threaded; parallelize (Ray/Dask) and add backpressure/queue limits.
9. TODO: no input validation; add JSON Schema + size/type/range checks to reject bad payloads.
10. FIXME: logging = print(); integrate structured logs + tracing + alerting for prod incidents.
""".strip()

# ---- Prompt 片段构造 ----
def build_yn_prompt(comment: str, full_label: str, abbr: str) -> str:
    """构造“是否属于某标签”的 Yes/No prompt。"""
    return (
        f"{PROMPT_HEAD}\n\n"
        "We are classifying the following code comment into one of the SATD categories.\n\n"
        "Code Comment:\n\"\"\"\n"
        f"{comment}\n\"\"\"\n\n"
        f"Question: Does this comment indicate the category \"{full_label} ({abbr})\"?\n"
        "Answer strictly with Yes or No."
    )

def build_pick_from_candidates_prompt(comment: str, candidates_abbr) -> str:
    """在候选缩写中只选 1 个的 prompt。"""
    cand_str = ", ".join(candidates_abbr)
    return (
        f"{PROMPT_HEAD}\n\n"
        "Pick the single best matching category for the comment below.\n"
        f"Candidates (abbreviations only): {cand_str}\n\n"
        "Comment:\n\"\"\"\n"
        f"{comment}\n\"\"\"\n\n"
        "Return exactly ONE abbreviation on a single line, with no extra text."
    )

def build_force_pick_any_prompt(comment: str) -> str:
    """从 23 个缩写中强制挑一个（兜底）。"""
    cand_str = ", ".join(sorted(ABBR_SET))
    return (
        f"{PROMPT_HEAD}\n\n"
        "Please determine which type of ML SATD the following Code Comment belongs to.\n"
        "Return exactly ONE of these abbreviations on a single line, with no extra text:\n"
        f"{cand_str}\n\n"
        "Comment:\n\"\"\"\n"
        f"{comment}\n\"\"\"\n\n"
        "Abbreviation:"
    )

# -------- 3. 解析函数 --------
# [R08 原代码保留]
# def extract_yes_no(raw: str) -> str:
#     """解析 Yes/No；返回 'yes'/'no'/'unk'。"""
#     first = re.split(r"[\r\n]", raw.strip(), maxsplit=1)[0].strip()
#     first = re.sub(r"(^[*_`]+|[*_`]+$)", "", first)  # 去修饰
#     t = first.lower()
#     t = re.sub(r"[^a-z\u4e00-\u9fa5]", "", t)  # 只留字母与中文（兼容“是/否”）
#     if t.startswith("yes") or t.startswith("y") or t.startswith("是"):
#         return "yes"
#     if t.startswith("no") or t.startswith("n") or t.startswith("否") or t.startswith("不是"):
#         return "no"
#     return "unk"
# [R08 修订]
def extract_yes_no(raw: str) -> str:
    if not isinstance(raw, str):
        return 'unk'
    answer = raw.strip()
    answer = re.sub('(^[*_`]+|[*_`]+$)', '', answer).strip()
    match = re.fullmatch('(yes|no|y|n|是|否|不是)[.!。！]?', answer, flags=re.I)
    if match is None:
        return 'unk'
    token = match.group(1).lower()
    if token in ('yes', 'y', '是'):
        return 'yes'
    return 'no'

def extract_abbr(raw: str) -> str:
    """从一行文本中提取我们 23 种缩写之一；失败返 '<UNK>'。"""
    first = re.split(r"[\r\n]", raw.strip(), maxsplit=1)[0].strip()
    first = re.sub(r"(^[*_`]+|[*_`]+$)", "", first)
    m = ABBR_REGEX.search(first)
    if m:
        return ABBR_LOWER[m.group(1).lower()]
    token = re.sub(r"[^a-z]", "", first.lower())
    if token in ABBR_LOWER:
        return ABBR_LOWER[token]
    # 尝试从全称首词猜测（很弱的兜底）
    # [R03 原代码保留]
    # for full, ab in LABEL_ABBR.items():
    #     if full.split()[0].lower() in token:
    #         return ab
    # [R03 修订]
    full_name_matches = [
        ab for full, ab in LABEL_ABBR.items()
        if re.sub(r"[^a-z]", "", full.lower()) == token
    ]
    if len(full_name_matches) == 1:
        return full_name_matches[0]
    return "<UNK>"

# -------- 4. 加载模型 --------
print("[INFO] Loading model …")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# -------- 5. 一个小封装：给 messages 生成（贪婪） --------
@torch.inference_mode()
# [R08 原代码保留]
# def greedy_generate_first_line(messages, max_new_tokens: int) -> str:
# [R08 修订]
def greedy_generate_first_line(messages, max_new_tokens: int, return_full_text: bool=False) -> str:
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, max_length=65536).to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        # do_sample=False,
        # eos_token_id=tokenizer.eos_token_id,
        # pad_token_id=tokenizer.eos_token_id,
        # temperature=0.7,
        # top_p=0.8,
        # top_k=20,
    )
    raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                           skip_special_tokens=True).strip()
    # [R08 原代码保留]
    # return raw.split("\n", 1)[0].strip()
    # [R08 修订]
    if return_full_text:
        return raw.strip()
    return raw.split("\n", 1)[0].strip()

# -------- 6. classify（轮询 Yes/No → 从候选选 1 个） --------
@torch.inference_mode()
# [R08 原代码保留]
# def classify_by_polling(comment: str):
#     yes_candidates = []
#
#     # 6.1 逐标签 Yes/No
#     for full in LABELS:
#         abbr = FULL2ABBR[full]
#         prompt = build_yn_prompt(comment, full, abbr)
#         messages = [{"role": "user", "content": prompt}]
#         raw_yn = greedy_generate_first_line(messages, MAX_YN_TOKENS)
#         yn = extract_yes_no(raw_yn)
#         if yn == "yes":
#             yes_candidates.append(abbr)
#
#     # 6.2 根据候选决定最终结果
#     if len(yes_candidates) == 1:
#         final_abbr = yes_candidates[0]
#         raw_final = final_abbr  # 直接用候选作为“原始输出”体现
#     elif len(yes_candidates) >= 2:
#         pick_prompt = build_pick_from_candidates_prompt(comment, yes_candidates)
#         raw_pick = greedy_generate_first_line([{"role":"user","content":pick_prompt}],
#                                               MAX_PICK_TOKENS)
#         pick_abbr = extract_abbr(raw_pick)
#         if pick_abbr in yes_candidates:
#             final_abbr, raw_final = pick_abbr, raw_pick
#         else:
#             # 如果解析不到候选之一，就退回到候选列表的第一个
#             final_abbr, raw_final = yes_candidates[0], raw_pick
#     else:
#         # 没有任何 Yes：走全量强制单标签输出
#         fb_prompt = build_force_pick_any_prompt(comment)
#         raw_fb = greedy_generate_first_line([{"role":"user","content":fb_prompt}],
#                                             MAX_FALLBACK_TOKENS)
#         fb_abbr = extract_abbr(raw_fb)
#         if fb_abbr in ABBR_SET:
#             final_abbr, raw_final = fb_abbr, raw_fb
#         else:
#             final_abbr, raw_final = FALLBACK_ABBR, raw_fb
#
#     return final_abbr, raw_final, yes_candidates
# [R08 修订]
def classify_by_polling(comment: str, diagnostics=None):
    yes_candidates = []
    unknown_labels = []
    fallback_reason = ''
    for full in LABELS:
        abbr = FULL2ABBR[full]
        prompt = build_yn_prompt(comment, full, abbr)
        messages = [{'role': 'user', 'content': prompt}]
        # [R08 修订] Y/N使用完整响应，以便拒绝后续行中的冲突答案。
        raw_yn = greedy_generate_first_line(messages, MAX_YN_TOKENS, return_full_text=True)
        yn = extract_yes_no(raw_yn)
        if yn == 'yes':
            yes_candidates.append(abbr)
        elif yn not in ('yes', 'no'):
            unknown_labels.append(abbr)
    if not unknown_labels and len(yes_candidates) == 1:
        final_abbr = yes_candidates[0]
        raw_final = final_abbr
    elif not unknown_labels and len(yes_candidates) >= 2:
        pick_prompt = build_pick_from_candidates_prompt(comment, yes_candidates)
        raw_pick = greedy_generate_first_line([{'role': 'user', 'content': pick_prompt}], MAX_PICK_TOKENS)
        pick_abbr = extract_abbr(raw_pick)
        if pick_abbr in yes_candidates:
            (final_abbr, raw_final) = (pick_abbr, raw_pick)
        else:
            (final_abbr, raw_final) = (FALLBACK_ABBR, raw_pick)
            fallback_reason = 'candidate_arbitration_invalid'
    else:
        fallback_reason = 'unresolved_votes_full_label' if unknown_labels else 'no_positive_votes_full_label'
        fb_prompt = build_force_pick_any_prompt(comment)
        raw_fb = greedy_generate_first_line([{'role': 'user', 'content': fb_prompt}], MAX_FALLBACK_TOKENS)
        fb_abbr = extract_abbr(raw_fb)
        if fb_abbr in ABBR_SET:
            (final_abbr, raw_final) = (fb_abbr, raw_fb)
        else:
            (final_abbr, raw_final) = (FALLBACK_ABBR, raw_fb)
            fallback_reason += ';full_label_parse_failed'
    if diagnostics is not None:
        diagnostics.update({
            'unknown_labels': list(unknown_labels),
            'fallback_reason': fallback_reason,
        })
    return (final_abbr, raw_final, yes_candidates)

# -------- 7. 主流程 --------
df = pd.read_csv(CSV_PATH)
rows, unk_cnt = [], 0

print("[INFO] Classifying …")
# [R08 原代码保留]
# for cmt, gold in tqdm(zip(df["Comment Text"], df["ML TD Type"]),
#                       total=len(df), ncols=80):
#     pred_abbr, raw_choice, yes_list = classify_by_polling(cmt)
#     if pred_abbr not in ABBR_SET:
#         unk_cnt += 1
#         if PRINT_UNPARSED:
#             with open(RUN_DIR/"unparsed.txt", "a", encoding="utf-8") as f:
#                 f.write(str(raw_choice) + "\n")
#         pred_abbr = FALLBACK_ABBR
#
#     rows.append({
#         "comment": cmt,
#         "real": ABBR2FULL.get(gold, gold),
#         "pred": ABBR2FULL[pred_abbr],
#         "raw_output": raw_choice,                 # 最终决策那一步的原始输出
#         "yes_candidates": ";".join(yes_list),     # 该条评论获得 Yes 的缩写集合
#     })
# [R08 修订]
for (cmt, gold) in tqdm(zip(df['Comment Text'], df['ML TD Type']), total=len(df), ncols=80):
    poll_diagnostics = {}
    (pred_abbr, raw_choice, yes_list) = classify_by_polling(cmt, diagnostics=poll_diagnostics)
    if pred_abbr not in ABBR_SET:
        unk_cnt += 1
        if PRINT_UNPARSED:
            with open(RUN_DIR / 'unparsed.txt', 'a', encoding='utf-8') as f:
                f.write(str(raw_choice) + '\n')
        pred_abbr = FALLBACK_ABBR
    rows.append({
        'comment': cmt,
        'real': ABBR2FULL.get(gold, gold),
        'pred': ABBR2FULL[pred_abbr],
        'raw_output': raw_choice,
        'yes_candidates': ';'.join(yes_list),
        'unknown_labels': ';'.join(poll_diagnostics.get('unknown_labels', [])),
        'fallback_reason': poll_diagnostics.get('fallback_reason', ''),
    })

pred_df = pd.DataFrame(rows)

# 终端打印（文本版）
report_text = classification_report(pred_df["real"], pred_df["pred"],
                                    labels=LABELS, zero_division=0, digits=3)
print("\n[INFO] Overall metrics:\n", report_text)

# -------- 8. 保存结果 --------
# 8.1 逐样本预测
pred_df.to_csv(RUN_DIR/"predictions.csv", index=False)

# 8.2 混淆矩阵
pd.DataFrame(
    confusion_matrix(pred_df["real"], pred_df["pred"], labels=LABELS),
    index=LABELS, columns=LABELS
).to_csv(RUN_DIR/"confusion_matrix.csv")

# 8.3 classification_report 存为 CSV，并追加 INFO 行
report_dict = classification_report(pred_df["real"], pred_df["pred"],
                                    labels=LABELS, zero_division=0, output_dict=True)
rep_df = pd.DataFrame(report_dict).transpose().reset_index().rename(columns={"index":"label"})
rep_df["info"] = ""
info_line = f"Unrecognized mapped to '{FALLBACK_ABBR}': {unk_cnt}/{len(df)}"
rep_df.loc[len(rep_df)] = {
    "label": "[INFO]", "precision": "", "recall": "", "f1-score": "", "support": "", "info": info_line
}
rep_df.to_csv(RUN_DIR/"report.csv", index=False)

# 8.4 保存设置
(Path(RUN_DIR/"settings.json")).write_text(json.dumps({
    "csv": CSV_PATH,
    "model": MODEL_PATH,
    "model_name": MODEL_NAME,
    "run_dir": str(RUN_DIR),
    "max_yn_tokens": MAX_YN_TOKENS,
    "max_pick_tokens": MAX_PICK_TOKENS,
    "max_fallback_tokens": MAX_FALLBACK_TOKENS,
    "fallback_abbr": FALLBACK_ABBR,
    "method": "YN polling -> pick best",
}, indent=2))

print(f"\n[INFO] Files saved in {RUN_DIR}")
print(f"[INFO] Unrecognized mapped to '{FALLBACK_ABBR}': {unk_cnt}/{len(df)}")
gc.collect(); torch.cuda.empty_cache()