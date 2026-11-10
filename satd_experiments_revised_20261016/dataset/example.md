**MACHINE LEARNING KNOWLEDGE (MLKnow) – 新增示例**

1. `TODO: Confirm whether layer-norm before attention actually boosts convergence for small-batch training on this dataset`
2. `TODO: Figure out if label smoothing distorts precision-recall when classes are highly imbalanced`
3. `XXX: Revisit our choice of cosine annealing—does it still make sense once we switch to AdamW?`

**MODEL INTERPRETABILITY (MLInter) – 新增示例**

1. `TODO: Generate SHAP summary plot so users can see feature impact`
2. `TODO: Log attention weights for each prediction to aid debugging`
3. `XXX: Provide saliency-map overlay in evaluation notebook for misclassified images`

**MODEL CODE COMPREHENSION (MCC) – 新增示例**

1. `TODO: break up monolithic function train_eval_predict() into clearer, single-purpose helpers`
2. `FIXME: variable “h” is reused for both hidden state and Hessian—pick distinct, descriptive names`
3. `XXX: replace magic constant 512 (actually EMBED_SIZE) with a config parameter or enum`

**DUPLICATE MODEL CODE (DMC) – 新增示例**

1. `TODO: remove copy-pasted ConvBlock definition—reuse common_layers.ConvBlock instead`
2. `FIXME: forward() in MyResNet duplicates logic from BaseResNet; refactor to call super()`
3. `XXX: data_augmentation_v2() is 95 % identical to data_augmentation(); consolidate into one helper`

**DUPLICATE FEATURE EXTRACTION CODE (DFEC) – 新增示例**

1. `TODO: merge extract_mfcc() and extract_mfcc_fast(); only differ in hop_length`
2. `FIXME: pixel_stats_v2() repeats almost all logic from pixel_stats(); refactor into shared util`
3. `XXX: text_features_v1 and text_features_v2 are copy-pasted—create one configurable pipeline`

**WEIGHT CONFIGURATION (Weight Conf) – 新增示例**

1. `TODO: expose weight_decay as configurable argument instead of hard-coding 0.01`
2. `FIXME: pretrained checkpoint loads fp16 weights—add auto-cast to ensure dtype consistency`
3. `XXX: add toggle to freeze first two Transformer layers during fine-tune via gradient mask`

**LAYER CONFIGURATION (Layer Conf) – 新增示例**

1. `TODO: swap BatchNorm for LayerNorm when depth > 24`
2. `FIXME: insert positional-embedding layer before first Transformer block`
3. `XXX: make dropout rate per-layer configurable instead of using global 0.1`

**HYPER-PARAMETER CONFIGURATION (HP Conf) – 新增示例**

1. `TODO: make learning_rate schedule (cosine vs. linear) selectable via CLI flag`
2. `FIXME: batch_size is hard-coded to 32; derive from GPU memory or user input`
3. `XXX: expose early_stopping_patience in config.yaml instead of magic value 10`

**CODE DEPENDENCY (MLDepend) – 新增示例**

1. `TODO: bump scikit-learn to ≥1.6 once conda forge wheels for arm64 become available`
2. `FIXME: delete custom tokenizer patch after huggingface/transformers merges PR #42812`
3. `XXX: keep legacy CUDA 11 build until upstream faiss releases official 12.x binaries`

**GLUE CODE (GC) – 新增示例**

1. `TODO: remove pandas-to-torch bridge once DataLoader accepts DataFrame directly`
2. `FIXME: adapter.py only exists to map old Predictor API to new one—rewrite consumers instead`
3. `XXX: consolidate fetch_from_s3_and_cache(); purely glue between boto3 and our local cache manager`

**CUSTOM DATA TYPE (CDT) – 新增示例**

1. `TODO: replace custom SparseTensorDict with standard torch.sparse_coo_tensor before exporting`
2. `FIXME: graph_utils.Node stores metadata in AttrDict; breaks ONNX serialization—use plain dict`
3. `XXX: pipeline returns dask DataFrame, but Keras expects NumPy array—add type-conversion layer`

**MULTIPLE LANGUAGES (ML) – 新增示例**

1. `TODO: port custom attention kernel from CUDA C++ to Metal for macOS deployment`
2. `FIXME: keep tokenizer logic in sync between Rust CLI and Python training script—generate bindings instead`
3. `XXX: consolidate duplicate data_cleaning.{scala,py}; two implementations drift out of spec easily`

**UNNECESSARY MODEL CODE (UMC) – 新增示例**

1. `TODO: delete legacy compute_grad_stats(); superseded by torch.autograd.profiler`
2. `FIXME: remove flag use_old_preprocess once new pipeline is stable`
3. `XXX: drop debug print statements inside training loop before release`

**DATA PROCESSING CONFIGURATION (Data Conf) – 新增示例**

1. `TODO: add configurable pipeline step to down-sample majority class before batching`
2. `FIXME: current JSON parser drops NaNs—update to preserve missing values for later imputation`
3. `XXX: expose window_size and stride in config.yaml so sliding-window generator isn’t hard-coded`

**PLAIN OLD DATA TYPE (PODT) – 新增示例**

1. `TODO: replace raw list of logits with a NamedTuple so callers know the order`
2. `FIXME: function returns plain dict; define a TypedConfig dataclass instead for clarity`
3. `XXX: passing labels as bare NumPy array—wrap in LabelsTensor to preserve shape metadata`

**DATA STORAGE CONFIGURATION (DS Conf) – 新增示例**

1. `TODO: migrate checkpoint format from .pth to safetensors for safer serialization`
2. `FIXME: results_writer currently appends CSV rows one by one—buffer and batch-flush to reduce I/O`
3. `XXX: store intermediate feature maps in mmap file to avoid exceeding GPU memory`

**ABSTRACTION (Abs) – 新增示例**

1. `TODO: extract common scoring logic into BaseMetric so each model wrapper isn’t re-implementing it`
2. `FIXME: decouple data_loader from Trainer—should accept any iterable, not hard-coded DataLoader`
3. `XXX: replace direct calls to torch.cuda.* with a HardwareBackend interface to support TPU`

**BOUNDARY EROSION (BE) – 新增示例**

1. `TODO: migrate training-time logging out of Trainer class; should live in metrics.logger`
2. `FIXME: data_loader currently triggers augmentations inline—move to separate preprocess pipeline`
3. `XXX: remove direct DB writes from evaluate.py; expose results through results_api instead`

**MODEL CODE MODIFIABILITY (MCM) – 新增示例**

1. `TODO: move hyper-parameter defaults into config file so changing them doesn’t require code edits`
2. `FIXME: hard-coded embedding_dim (256) sprinkled across modules—define once in constants.py`
3. `XXX: split run_experiment.py into cli entrypoint and reusable Experiment class for easier reuse`

**MODEL CODE REUSABILITY (MCR) – 新增示例**

1. `TODO: abstract make_encoder() so VisionTransformer and ResNet builders share the same backbone factory`
2. `FIXME: refactor evaluate_batch() into mix-in class so both Trainer and HyperTune reuse identical logic`
3. `XXX: remove hard-coded Softmax loss; accept any callable loss_fn to support regression and ranking tasks`

**PREDICTION QUALITY (PQ) – 新增示例**

1. `TODO: rerun validation after enabling mixed-precision; confirm accuracy isn’t degraded`
2. `FIXME: measure F1 change when we swap tokenizer to WordPiece—may hurt rare-word recall`
3. `XXX: compare BLEU before/after removing dropout layer to ensure translation quality holds`

**MACHINE LEARNING RELIABILITY (MLRelia) – 新增示例**

1. `TODO: add sanity-check to reject examples with >20 % missing features before prediction`
2. `FIXME: wrap model call in try/except and fall back to last-known good output when decoder crashes`
3. `XXX: create robustness test suite with Gaussian noise and FGSM perturbations to monitor accuracy drift`

**PROTOTYPE (Proto) – 新增示例**

1. `TODO: replace the single-threaded demo server with a production-ready Gunicorn setup`
2. `FIXME: temp JSON files used for state—migrate to proper Redis cache before rollout`
3. `XXX: remove hard-coded API key left from prototype; load securely from env vars instead`



##### 

##### 