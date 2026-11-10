# ML-SATD 代码上下文先导实验

本目录包含一次100条样本、三个输入条件的配对实验。原始数据、论文和既有脚本未修改。

## 固定设计

- 模型：官方 `Qwen/Qwen3-8B`，原始BF16权重，无量化；revision见 `model_manifest.json`。
- 运行：Apple M3 Max / MLX；精确软件版本见 `requirements.lock.txt` 与运行设置。
- 提示：从 `../basic_zero_shot/Qwen3-8B_zero_shot_re.py` 通过AST提取类别定义、vanilla分类指令、提示构建器及解析器。原脚本副本、hash和提取结果在 `reference/`，不执行其CUDA加载副作用。
- 原提示为零示例、非思考vanilla；用户明确选择先用非思考模式尽快完成对照。本次不复现论文最佳三示例+CoT配置。中止的思考试跑保存在 reference_thinking_stopped/、runs/thinking_stopped_* 与 protocol_thinking_stopped.json，完全排除于本轮统计。
- 保留 `enable_thinking=False`、temperature=0.7、top_p=0.8、top_k=20、输入上限4096、输出上限20。实际token范围见 runs/main/settings.json，无截断。
- 显式采用采样并固定每样本种子；三臂共享该种子。原脚本未显式指定 `do_sample`，不能据源文件保证历史运行解码策略相同。
- 每个样本三臂：`comment_only`（原提示）、`window`（增加目标行前后各10行）、`function`（增加AST提取的完整最内层函数）。两种上下文臂添加相同的目标定位指令；不输入金标、仓库名、pipeline人工标签、提交消息或移除后的修复。
- 共享提示token前缀仅缓存一次；每次推理后把KV offset退回前缀，杜绝上一条样本内容进入下一条。
- 每样本内三臂的执行顺序确定性打乱；保存完整提示、token hash、种子、原始输出、完成原因、token数与耗时。
- 保留原解析器；不把解析失败回填为多数类。旧回填值仅作为诊断字段保存。非思考模式直接解析模型输出首行。

## 样本与历史来源

`prepare_pilot.py` 从原393条数据中建立225条满足源码对齐、去重和函数提取的候选池，按固定SHA排序和类别轮转选100条，来自74仓库、21类。未使用任何模型预测进行抽样。DMC和DFEC唯一样本不满足上下文条件，故本次不评估它们。

`data/cohort.jsonl` 是冻结队列；`data/audit.json` 记录类别数、排除规则和源CSV的hash；`data/validation.json` 记录确定性与对齐检查。分层样本不代表原393条的类别分布，不能直接与论文全量分数比较。

历史核验独立保存于 `data/historical_verification/`：96/100条的完整文件SHA与引入提交原文件完全相同；另外4条链接404，保持未验证，不用移除提交或当前HEAD替代。冻结队列中的 `snapshot_revision_verified:false` 是核验前的状态；历史manifest是核验结果的依据。

## 复现

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.lock.txt
.venv/bin/python prepare_pilot.py
.venv/bin/python download_model.py
.venv/bin/python -m unittest test_runner.py test_analyze_pilot.py
.venv/bin/python run_pilot.py --check-only
.venv/bin/python wait_and_run.py
```

用户要求保留另一个Ollama任务。因此 `wait_and_run.py` 等待模型下载校验完成和Ollama无驻留模型后才启动本次推理，绝不卸载或终止其他模型。状态见 `batch_status.json`、`logs/batch.log` 和 `logs/inference.log`。

如已确认内存可用，也可直接运行 `.venv/bin/python run_pilot.py`。每条完成预测同步写入 `runs/main/predictions.jsonl`，原协议和输入hash一致时可恢复。禁止同时启动两个runner。

完成后分析：

```sh
.venv/bin/python analyze_pilot.py --predictions runs/main/predictions.jsonl --output analysis --require-complete
```

主分析为相同100条的三臂配对结果：accuracy、weighted-F1、21类macro-F1，附固定23类macro-F1、各类结果、解析失败率、成本和2000次按金标类别分层的配对bootstrap区间。输出在 `analysis/`。区间不涵盖仓库内相关性或多解码种子的差异。人工盲审表只是待填写材料；本次模型推理不等同于已经完成双人标注。
