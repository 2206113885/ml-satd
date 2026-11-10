#!/usr/bin/env python3
"""Static/pure-function review. Never executes original module top levels."""
import ast
import copy
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

BASE=Path(__file__).resolve().parent
DATA=BASE.parent
EXCLUDE={'pilot_context_20261016','paper_generation_audit_20261016','code_review_20261016_round2','models'}
FILES=sorted(p for p in DATA.rglob('*.py') if not set(p.relative_to(DATA).parts)&EXCLUDE)
def relative(path):return str(path.relative_to(DATA))
def save(name,obj):(BASE/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def location(path,node):return {'file':relative(path),'line':node.lineno,'end_line':getattr(node,'end_lineno',node.lineno)}
def matching_lines(path,pattern):
    return [{'file':relative(path),'line':i,'end_line':i} for i,line in enumerate(path.read_text().splitlines(),1) if re.search(pattern,line)]
def pure_namespace(tree):
    namespace={'re':re,'json':json,'Counter':Counter}
    allowed={'LABEL_ABBR','ABBR2FULL','ABBR_SET','ABBR_LOWER','ABBR_REGEX','LABELS','FULL2ABBR','FULL_BY_CANON','FULL_BY_ABBR_CANON','COARSE_LABELS',
             'FINE_LABEL_ABBR','FINE_ABBR2FULL','FINE_ABBR_SET','FINE_ABBR_LOWER','COARSE_LABEL_ABBR',
             'COARSE_ABBR_SET','COARSE_ABBR_LOWER','COARSE2FINE_ABBR','_CANON2ABBR'}
    # Only label maps and their derived comprehensions, never model/load/path/auth expressions.
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and (node.name.startswith(('simple_extract','extract','parse','normalize','canon','_canon','_visible','_extract'))):
            node=copy.deepcopy(node);node.decorator_list=[]
            node.returns=None
            for argument in node.args.args+node.args.kwonlyargs:argument.annotation=None
            try:exec(compile(ast.Module(body=[node],type_ignores=[]),'<pure-parser>','exec'),namespace)
            except Exception:pass
    for _ in range(3):
        for node in tree.body:
            if isinstance(node,ast.Assign) and all(isinstance(t,ast.Name) and t.id in allowed for t in node.targets):
                try:exec(compile(ast.Module(body=[node],type_ignores=[]),'<label-definitions>','exec'),namespace)
                except Exception:pass
    return namespace

def main():
    BASE.mkdir(parents=True,exist_ok=True)
    inventory=[];coverage=[];trees={};parsertests=[];firstword=[];spacefail=[];nonanswer=[]
    seedless=[];unversioned=[];truncated=[];fallback=[];binaryraw=[];binarylabelonly=[];binarypartial=[];treelog=[];gptharmony=[];thinkrisk=[];yesnoloose=[]
    for path in FILES:
        text=path.read_text();entry={'file':relative(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        try:tree=ast.parse(text)
        except SyntaxError as exc:
            entry.update(status='reviewed-syntax-error',syntax_error={'line':exc.lineno,'message':exc.msg},calls=[])
            inventory.append(entry);coverage.append({'file':relative(path),'status':'reviewed','inference_scope':'unrunnable legacy script; syntax invalid','notes':'No original top level executed.'});continue
        trees[path]=tree
        nativeqwen=any(isinstance(n,ast.Assign) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str) and 'Qwen3' in n.value.value and any(isinstance(t,ast.Name) and t.id=='MODEL_PATH' for t in n.targets) for n in tree.body)
        parents={child:parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        def fn_for(n):
            while n in parents:
                n=parents[n]
                if isinstance(n,ast.FunctionDef):return n.name
            return '<module>'
        calls=[];gens=[];seeds=[];loads=[]
        for node in sorted((n for n in ast.walk(tree) if isinstance(n,ast.Call)),key=lambda x:x.lineno):
            name=node.func.attr if isinstance(node.func,ast.Attribute) else node.func.id if isinstance(node.func,ast.Name) else ''
            kwargs={k.arg:ast.unparse(k.value) for k in node.keywords if k.arg and k.arg not in {'token','api_key','use_auth_token','hf_token','password','secret'}}
            factory=name=='pipeline' and bool(node.args) and isinstance(node.args[0],ast.Constant) and node.args[0].value=='text-generation'
            if name in {'generate','pipe','pipeline','apply_chat_template','decode','batch_decode','encode'} or 'truncation' in kwargs:
                calls.append({'line':node.lineno,'end_line':node.end_lineno,'method':name,'function':fn_for(node),'factory':factory,'kwargs':kwargs})
            if name in {'generate','pipe','pipeline'} and not factory:gens.append(node)
            if name in {'manual_seed','manual_seed_all','set_seed','seed'}:seeds.append(node)
            if name=='from_pretrained' or factory:loads.append(node)
            if kwargs.get('truncation')=='True':truncated.append(location(path,node))
        for node in ast.walk(tree):
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['abbr','final_abbr','fine_abbr','pred_full','pred'] for t in node.targets):
                value=ast.unparse(node.value)
                if 'FALLBACK' in value or value in ('yes_candidates[0]','candidates[0]'):
                    fallback.append(location(path,node))
        entry.update(status='reviewed',calls=calls,seeds=[n.lineno for n in seeds],generation_calls=[n.lineno for n in gens],
                     parser_functions=[{'name':n.name,**location(path,n)} for n in tree.body if isinstance(n,ast.FunctionDef) and n.name.startswith(('extract','parse','_visible','_extract'))])
        inventory.append(entry)
        coverage.append({'file':relative(path),'status':'reviewed' if gens else 'not-applicable',
                         'inference_scope':'generation/configuration/control-flow/parsing reviewed' if gens else 'no inference generation call; supporting data/download/baseline tool',
                         'notes':'All calls reviewed statically; pure parser examples executed where applicable. No original top level executed.'})
        if gens and not seeds:seedless.extend(location(path,n) for n in gens)
        if gens and not any(any(k.arg=='revision' for k in n.keywords) for n in loads):
            unversioned.extend(location(path,n) for n in loads)
        namespace=pure_namespace(tree)
        for node in tree.body:
            if not isinstance(node,ast.FunctionDef):continue
            source=ast.get_source_segment(text,node) or ''
            if 'full.split()[0].lower() in token' in source:firstword.append(location(path,node))
            if node.name in ('simple_extract','extract_abbr','extract_abbr_from_text') and node.name in namespace and len(node.args.args)==1:
                for supplied,expected,kind in [('Data Conf','Data Conf','canonical_control'),('data conf','Data Conf','case_normalization'),('**Data Conf**','Data Conf','markdown_normalization'),('Data Storage Configuration','DS Conf','full_name'),('Model Code Reusability','MCR','full_name'),('I cannot determine a label.','<UNK>','nonanswer'),('No model label applies.','<UNK>','nonanswer')]:
                    try:actual=namespace[node.name](supplied)
                    except Exception:continue
                    result={'file':relative(path),'function':node.name,'line':node.lineno,'input':supplied,'expected':expected,'actual':actual,'kind':kind,'synthetic':True}
                    parsertests.append(result)
                    if kind in ('case_normalization','markdown_normalization') and actual!=expected:spacefail.append(location(path,node))
                    if kind=='nonanswer' and actual!='<UNK>':nonanswer.append(location(path,node))
            if node.name in ('extract_yes_no','parse_yesno') and node.name in namespace:
                for supplied in ['Neither option can be determined.','Not sure.']:
                    try:actual=namespace[node.name](supplied)
                    except Exception:continue
                    if actual in ('yes','no'):
                        yesnoloose.append(location(path,node));parsertests.append({'file':relative(path),'function':node.name,'line':node.lineno,'input':supplied,'expected':'unknown/unk','actual':actual,'kind':'uncertain_yesno','synthetic':True})
            if node.name in ('_visible_only','classify') and "split('</think>')[-1]" in source:
                if 'enable_thinking=True' in source and nativeqwen:thinkrisk.append(location(path,node))
            if node.name=='_visible_only' and re.search(r'<think>\.\*\?</think>',source):
                if nativeqwen:thinkrisk.append(location(path,node))
        if path.parent.name.startswith('bisection') or path.name in ('test1.py','test3.py'):
            if '"raw_output": final_raw' in text and 'yn_raw_all' in text:
                locs=matching_lines(path,r'yn_raw_all\.append|"raw_output":\s*final_raw')
                binarylabelonly.extend(locs);binaryraw.extend(locs)
            else:
                locs=matching_lines(path,r'^\s*raw_yn\s*=|^\s*rows.append|^def yes_or_no')
                binarypartial.extend(locs);binaryraw.extend(locs)
        declares_false=any(isinstance(n,ast.Assign) and isinstance(n.value,ast.Constant) and n.value.value is False and any(isinstance(t,ast.Name) and t.id=='DO_SAMPLE' for t in n.targets) for n in tree.body)
        if path.parent.name.startswith('tree') and declares_false and gens and not any(any(k.arg=='do_sample' for k in n.keywords) for n in gens):
            treelog.extend(matching_lines(path,r'^\s*# do_sample=DO_SAMPLE|"do_sample":\s*DO_SAMPLE'))
            if not any('# do_sample=DO_SAMPLE' in line for line in text.splitlines()):treelog.extend(location(path,n) for n in gens)
        if 'assistantfinal' in text:
            gptharmony.extend(matching_lines(path,r"\.split\('assistantfinal'\)|assistantfinal.*\$"))
    # Unique exact locations, sorted to make a manual fix checklist stable.
    def unique(items):return sorted({(x['file'],x['line'],x['end_line']):x for x in items}.values(),key=lambda x:(x['file'],x['line']))
    findings=[]
    def finding(id,title,status,severity,description,impact,recommendation,locations,evidence):
        findings.append(dict(id=id,title=title,status=status,severity=severity,description=description,impact=impact,recommendation=recommendation,locations=unique(locations),evidence=evidence))
    p=DATA/'tree_zero_shot/gemma-2-9b_zero_shot.py'
    finding('INF-01','Gemma2层次零样本丢弃聊天模板且忽略max_len','confirmed','P1',
        'generate_once计算text后实际编码prompt；max_len未传入tokenizer，且解码完整输入输出再按Answer:切分。',
        '该条件未按其他层次脚本的聊天模板和输入预算运行；全量解码/分隔符也不能证明只解析了新生成内容。',
        '实际编码text并执行max_len；仅解码outputs中输入长度之后的新token，去除对提示文本分隔符的依赖。复核对应Gemma2层次结果。',
        [{'file':relative(p),'line':276,'end_line':288}],{'method':'source control-flow inspection','max_len_used_in_function_body':False})
    finding('INF-02','首词兜底将不同完整类名映射成同一类','confirmed','P1',
        '使用full.split()[0]是否在输出中出现判类，Model和Data分别对应多个不同标签。',
        '合法完整类名可能映射为首个共享首词的类别；拒答也可能被常见子串误认。',
        '删除首词/任意包含兜底；使用完整名称精确归一化或唯一缩写匹配，歧义/拒答保留解析失败。',firstword,
        [r for r in parsertests if r['kind']=='full_name' and r['actual']!=r['expected'] and any(x['file']==r['file'] and x['line']==r['line'] for x in firstword)])
    finding('INF-03','多词缩写归一化与字典键不一致','confirmed','P2',
        '将输入去掉空白/标点后用仍保留空格的ABBR_LOWER查找，导致data conf和**Data Conf**等形式被拒绝。',
        '格式差异进入默认类别回填，和语义判错混在一起；单词缩写与多词缩写待遇不一致。',
        '以同一个canonical函数构建标签字典和处理输出；单独记录非规范格式，避免静默改成Data Conf。',spacefail,
        [r for r in parsertests if r['kind'] in ('case_normalization','markdown_normalization') and r['actual']!=r['expected']])
    finding('INF-04','未闭合thinking没有统一无效状态','current-risk','P1',
        '思考块结束后切割或只删除完整块，均未在缺少结束标记时明确拒绝；binary解析会搜索剩余全文的yes/no及标签。',
        '生成触及合理预算而仍未完成思考时，可能把中间候选当答案；并非预算本身有错。旧分类原文丢失，实际污染数未知。',
        '解析前验证thinking是否正常闭合及最终段是否存在；未完成保存为独立失败，记录finish_reason和完整raw，不从思考中搜索答案。',thinkrisk,
        json.loads((DATA/'paper_generation_audit_20261016/qwen_official_audit.json').read_text())['binary_cot_output_evidence']['proof'])
    finding('INF-05','解析失败/仲裁失败被回填为正常类别','design-choice','P1',
        '多处将UNK改成Data Conf、粗类默认子类或yes_candidates[0]，报告只评估替代后的类别。',
        '掩盖格式失败、无答案与真实预测之间的差别，仲裁首候选还引入标签顺序偏好；不能仅凭有fallback否定原实验，但需披露。',
        '保留parsed_label、parse_status、fallback_label、最终策略预测分别列；主表报告失败计错及可选原回填策略敏感性结果。',fallback,
        {'method':'AST assignment/control-flow inspection','policy':'合理回退策略可以保留，但必须单独统计其贡献，勿把所有回退当成模型自主答案。'})
    finding('INF-06','层次脚本记录DO_SAMPLE=False但调用没有传入','confirmed','P1',
        'DO_SAMPLE写入settings，实际generate参数do_sample=DO_SAMPLE被注释；涉及多个模型而非只有Qwen。',
        'effective配置可能来自模型generation_config，与保存的false不符；不能按注释或settings倒推历史解码。',
        '按实验意图显式传参，并保存合并后的model.generation_config和调用覆盖值；历史文件保留原样并标记不确定。',treelog,
        {'method':'unused-parameter and settings inspection','official_Qwen_configs_do_sample':True})
    finding('INF-07','Qwen14B binary零样本显式greedy不同于相邻条件','confirmed','P2',
        'Qwen3-14B binary零样本明确do_sample=False；8B等条件未覆盖且官方原配置为采样。',
        '模型/提示对比包含解码策略差异，结果不宜解释成仅提示造成。',
        '披露该条件实际策略；如需控制变量比较，仅重评受影响条件并固定/记录有效配置。',
        matching_lines(DATA/'bisection_zero_shot/Qwen3-14B_zero_shot.py',r'^\s*do_sample=False'),
        {'official_source_artifact':'../paper_generation_audit_20261016/qwen_official_audit.json','historical_effective_config':'conditional on original server checkpoint config'})
    finding('INF-08','binary阶段完整原始输出被舍弃','confirmed','P1',
        'A类4个Qwen3 binary-CoT/native源文件把最终raw_output压成缩写；B类其余binary源文件通常保留最终选择文本或候选，但不持久化各标签阶段1原文，旧bin_test甚至只保存布尔候选和解释。阶段完成原因均未记录。',
        'A类不能恢复最终生成，A/B类均不能离线重放全部yes/no解析；并非所有binary的最终raw都仅有缩写。',
        '每个样本保存各标签调用、候选、仲裁、回退、完整原文和finish_reason；现有缩写字段改名parsed_output。',binaryraw,
        {'A_label_only':unique(binarylabelonly),'B_stage1_raw_missing':unique(binarypartial),'Qwen_binary_CoT_runs':8,'n':3144,'observed_Qwen_CoT_all_raw_output_already_parsed':True,'source_artifact':'../paper_generation_audit_20261016/qwen_official_audit.json'})
    finding('INF-09','依赖未固定的pipeline内容/assistantfinal字符串约定','current-risk','P2',
        'GPT-OSS路径假定generated_text是消息列表并用assistantfinal字符串切分；没有统一校验消息role、channel及最终答案是否存在。',
        '不同Transformers/Harmony输出结构或未完成final时，可能异常或把analysis/其他内容送入宽松标签搜索。标准pipeline消息结构本身并非错误。',
        '固定Transformers版本，保存结构化完整响应；明确提取最后assistant final内容，缺失final标记为失败。加入正常及截断结构的纯解析测试。',gptharmony,
        {'status_boundary':'需原包版本和结构化输出确认实际是否发生；本次仅证实源码缺少校验，未运行模型。'})
    finding('INF-10','缺少样本级种子、完整有效配置及版本快照','needs-history','P2',
        '生成调用常省略可从checkpoint继承的采样值，未固定revision/环境；多数生成脚本没有设置随机种子。',
        '不能仅凭当前脚本重建历史实际配置；省略do_sample不等于greedy。单次随机结果缺少跨种子不确定性。',
        '保存每臂有效generation_config、模板/输入hash、checkpoint revision、依赖版本、seed、原始长度和设备。对新随机结果使用预先约定的重复/配对分析。',seedless,
        {'seedless_generation_call_locations':len(seedless),'unpinned_loader_locations':unversioned,'exception':'非零temperature以及合理max_new_tokens本身均不列为bug。'})
    finding('INF-11','输入截断没有记录目标/示例保留情况','current-risk','P2',
        '多个调用启用truncation=True但没有保存原始token数、截断方向及目标所在位置；少数脚本预算512。',
        '右截断超长类别定义/示例提示可能丢掉末尾目标，左截断可能丢规则/示例；仅看到cap不能断言实际已截断。',
        '在生成前统计真实token长度并验证目标和输出指令保留；超限时显式压缩示例/上下文或跳出计为输入预算失败。',truncated,
        {'method':'AST tokenization call inventory','historical_boundary':'实际截断比例需当时完整prompt/tokenizer恢复；未据cap大小判定所有条件有错。'})
    p=DATA/'download_model.py'
    finding('INF-12','下载工具硬编码认证信息','confirmed','P1',
        '模型下载工具把认证信息作为字符串字面量传入。此问题属于辅助工具维护范围。',
        '源码复制/共享时可能一并传播认证信息；审查产物不包含其值。',
        '手动移除字面量，改从环境变量或已有Hugging Face登录缓存读取；不要在日志或配置产物中写入认证值。',
        [{'file':'download_model.py','line':32,'end_line':32}],{'method':'AST detects string-literal authentication keyword; value intentionally redacted'})
    p=DATA/'ICL_Ob_qwen-8b_new.py'
    finding('INF-13','旧ICL工具的生成tokenizer被检索tokenizer覆盖','confirmed','P1',
        '全局tokenizer最初属于Qwen，随后被all-MiniLM-L6-v2的tokenizer覆盖，最终再次传给Qwen生成函数。仅涉及旧ICL工具。',
        '如修复启动问题继续执行，输入token ID及特殊标记与生成模型不匹配。不能据此推断论文九模型主实验受影响。',
        '分别使用generation_tokenizer和embedding_tokenizer，显式传给各自函数；记录各自模型标识。',
        [{'file':relative(p),'line':65,'end_line':65},{'file':relative(p),'line':304,'end_line':305},{'file':relative(p),'line':413,'end_line':413}],
        {'method':'global name rebinding and call-site inspection','scope':'legacy auxiliary ICL prototype'})
    finding('INF-14','Y/N解析把不确定答复当作No','confirmed','P2',
        '部分Y/N解析使用startswith("no")或更宽的startswith("n")；Not sure被判为no，更宽版本还把Neither option等不确定输出判为no。',
        '模型拒答/不确定变成负类投票，影响候选集及后续仲裁路径，无法单独统计格式失败。',
        '仅接受完整规范化yes/no（可允许严格定义的标点）；不确定、双答案和解释前缀记录unknown。',yesnoloose,
        [r for r in parsertests if r['kind']=='uncertain_yesno'])
    finding('INF-15','旧显式CoT工具没有关闭Qwen3原生thinking','current-risk','P2',
        'original_COT.py声称使用显式自然语言CoT，但_encode_chat没有传enable_thinking=False，生成阶段又显式采用greedy；当前官方Qwen3模板默认开启thinking。',
        '在未修改的官方模板下，阶段2的64-token分类预算可能仍生成原生思考；这是意图与模式控制不一致，非预算本身错误。历史本地模板未知。',
        '若设计为两阶段显式CoT，明确关闭原生thinking并保留原始输出；若设计为原生thinking，则相应记录模式并验证final段存在。',
        [{'file':'original_COT.py','line':156,'end_line':177},{'file':'original_COT.py','line':205,'end_line':213}],
        {'scope':'legacy auxiliary prototype','condition':'official unmodified Qwen3 template defaults enable_thinking=True','no_model_execution':True})
    save('inference_findings.json',{'scope':'114 original Python files; inference/config/control-flow/parsing/reproducibility. No model inference and no original file changes.',
        'definitions':{'confirmed':'source defect or observed output directly established','current-risk':'reachable risky path; actual affected historical count unavailable','needs-history':'requires original checkpoint/environment/raw logs','design-choice':'valid policy requiring explicit reporting, not automatically a bug'},
        'findings':findings})
    save('inference_coverage.json',{'files_total':len(FILES),'files':coverage})
    save('inference_call_inventory.json',inventory)
    save('pure_parser_counterexamples.json',parsertests)
    print(json.dumps({'files':len(FILES),'findings':len(findings),'parser_examples':len(parsertests),'firstword_files':len(firstword),'space_failure_locations':len(unique(spacefail)),'tree_logging_locations':len(treelog)}))

if __name__=='__main__':main()
