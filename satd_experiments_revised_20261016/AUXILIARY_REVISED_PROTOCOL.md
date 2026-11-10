# 第三批辅助实验代码修订说明

本批只修改代码和说明，并仅做 AST/文本检查；未执行模型、分词器、实验、数据流水线、函数测试或评估。没有新实验结果。

## 历史资料边界

- dataset 中的所有 CSV、random_classify 的旧 results、pilot 的 reference/reference_thinking_stopped、data、runs、analysis 和 model_manifest 均保持历史证据身份，不在本批重写。
- 旧 pilot 写入入口的 main 已禁用；其历史源码保留在本批 before 备份及修订文件注释中。新入口以 `_revised.py` 结尾，必须显式提供输入与新输出目录。
- 三份 ICL_Ob_qwen-8b 旧实现全部注释且禁用。语法、缺失名称、错误 gold、混用 tokenizer、旧 fold 和固定示例的问题没有被伪装成旧结果可用；新实验必须走 icl_revised.py 的独立协议。
- original_COT.py 的旧原型全部原位注释；其中未定义的阶段构造器也保留为历史。活跃部分现在有完整的两阶段实现。
- 只有认证明文不遵循原文恢复要求：download_model.py 及本批 before 对应文件已遮盖凭据，任何注释、manifest、diff 都不得复制凭据。此次安全清理不能字节恢复凭据。

## 新输出和数据处理

所有新流水线或实验输出只允许写入本副本 `revised_outputs/` 的一个新子目录。除 pilot 同协议断点续跑外，已有目录直接拒绝，任何输入 CSV 均不会被覆写。没有自动改变主论文脚本的数据路径。

data_process.py 必须显式传入 `--input-csv` 与 `--output-dir`。它规范化23类及已知别名，把仅 secondary 有值的合法单标签提升到主标签；两列都有值、非法/缺失 gold、缺失身份均逐行记录排除原因。先按 Dataset ID/项目/路径/引入提交/规范化注释核对身份，再移除同身份副本与同项目相同规范化文本；冲突组整体排除，跨项目同文本保留并标记。`row_ledger.jsonl` 保留每个源行、ID、提升/排除/重复原因；manifest 保存输入/输出/脚本/辅助模块 SHA256。

默认保留注释原文。`--strip-leading-debt-markers` 是单独申明的新处理选择，不能据此声称找回历史 rmvtodo 规则。R30 的原始版本与393条历史版本链仍未解决；用户提供的 source-version-note 仅作未验证说明。

## 新 ICL 的前置条件

icl_revised.py 是可执行的新 TF-IDF 检索+本地 Qwen3 生成实现。它要求独立 `--train-csv`、`--test-csv`、`--clone-provenance`、`--model-dir`、`--output-dir`，并支持显式列名。默认 gold 为 ML TD Type；明确禁止传统 satd_type，所有 gold 必须映射到23类。

train/test 中必须有 Clone Group（可用 `--clone-column` 指定）且每条有值。来源 JSON 必须有绑定实际输入字节的 train_sha256、test_sha256 和非空 method。每个输入的ID唯一，且跨集合的ID、规范化注释、项目、克隆组均不得相交。克隆组判定质量仍是外部研究责任；哈希只绑定该声明与数据，不能证明其正确。缺失这些材料时明确报错，绝不借用旧 fold 或固定示例。

检索只拟合 supplied train，示例来源ID逐样本记录；不使用 embedding tokenizer，生成 tokenizer 始终独立。默认3个示例，超输入预算显式记失败，不自动截断或减少shot。R44 历史 GPT-5 示例来源、seed、生成过程仍缺失；新协议不推断这些历史信息，也不声称与旧 ICL 方法等价。

## 新两阶段 CoT / 随机基线

original_COT.py 现在要求显式 input-csv/model-dir/output-dir。两阶段均 `enable_thinking=False`，阶段1采样、阶段2显式 greedy；每次完整原文、提示、有效配置、token数、结束状态和样本/重复 seed 都保存。最多票平局/无合法票按严格失败计分，另保存原 Data Conf 策略的对照预测，避免把回退当真实输出。输入不截断，预留实际上下文的输出空间。

qwen_auxiliary_runtime.py 只在显式调用新实验时加载本地模型。它记录实际本地配置、模板、权重文件 SHA256 和依赖版本；路径或 `_commit_hash` 不能证明官方或历史服务器身份，此限制被明确记录。没有下载或修改模型链接。

random_classify/random_classify.py 必须显式输入和输出，预先选择 `--seeds`（默认0到29、至少两个不同seed）。每个seed都保留完整预测/混淆矩阵/报告/设置，并额外输出均值、样本标准差、经验2.5%–97.5% seed波动分位数。该分位范围不是总体均值置信区间；解析期望 accuracy=1/23。旧单次run和未知旧seed不改。

## 修订 pilot 的计分与历史核验

- prepare_pilot_revised.py：显式 final-csv/context-csv/output-dir；沿用原筛选算法，输出到新目录。
- run_pilot_revised.py：显式 cohort/run-dir；使用 reference_revised.py 的配对 JSON 注释边界与统一的整段大小写/空白规范化缩写解析。历史 frozen 不修改；这是新协议、新run。
- finish_reason 只接受 stop 或 length；空/未知结束状态失败。默认 length-policy=accept_valid_label：即使耗尽上限，整段可见输出恰为一个合法缩写时仍接受，并写 hit_limit 与 accepted_at_limit。可在新run显式选 reject；设置与解析器一致，绝不回写历史300条分数。
- run settings 记录实际本地模型文件hash，并把权重与固定下载manifest的checksum核对；新runner源码、reference、辅助模块、模板token和cohort都绑定hash。不能把这些新记录当成历史服务器配置证明。
- verify_historical_revised.py：显式 cohort/output-dir，只请求记录的40位 introducing commit 与文件路径；证据写新目录。
- verified_subset_revised.py：必须显式指定 cohort/history-manifest/history-summary/evidence-root/predictions/settings/prompts/runner-source/reference-source/output-dir（修订run另需与实际模块一致的helper-source）。先核对 summary/cohort hash及数量，再核对每条 ID、snapshot hash、引入URL与路径；对入选项实际读取历史证据，验证字节hash、完整snapshot hash、目标行和函数/窗口内容。预测必须绑定同cohort且每个ID/arm唯一完整，gold一致。父prompt日志实际原文与已支持reference的静态受限构造、cohort/arm须逐项一致，prediction.prompt_sha256必须匹配日志原文；runner/reference源码hash、seed、arms、selected_ids也必须与settings一致。不导入或执行历史runner/reference；缺少原始源码快照或日志就拒绝，不能推定谱系。失配拒绝复用，要求重新核验；不会只靠ID判“已历史验证”。
- analyze_pilot_revised.py：显式 cohort/predictions/output，沿用历史分析算法，拒绝覆盖旧分析目录。

旧 test_runner.py/test_analyze_pilot.py 是历史协议测试材料，并未在本批执行或当作新协议通过证据。修订代码的运行正确性仍待用户未来授权执行后验证。

## 未解决的外部材料

R30 的历史原始数据/处理中间版本、旧结果的模型/模板/seed、R44 的 GPT-5 示例生成记录，以及独立 ICL train/test 和人工/工具克隆组来源，不能从现有代码推造。代码已提供拒绝误用和前瞻可复现的入口；这些历史来源问题仍明确未解决。
