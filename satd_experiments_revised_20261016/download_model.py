# [R12 原代码保留] 历史入口整体归档；以下原文不参与执行。
# from huggingface_hub import snapshot_download
# 
# # snapshot_download(repo_id='Qwen/Qwen3-8B',
# #                   repo_type='model',
# #                   local_dir='/data/yangzhifei/pro/ML_SATD/models/Qwen3-8B',
# #                   force_download=True)
# 
# 
# # not complate
# # snapshot_download(repo_id='Qwen/Qwen3-14B',
# #                   repo_type='model',
# #                   local_dir='/data/yangzhifei/pro/ML_SATD/models/Qwen3-14B',
# #                   token='[REDACTED_R12_SECRET]',
# #                   force_download=True)
# 
# 
# # snapshot_download(repo_id='Qwen/Qwen2.5-Coder-7B-Instruct',
# #                   repo_type='model',
# #                   local_dir='/data/yangzhifei/pro/ML_SATD/models/Qwen2.5-Coder-7B-Instruct',
# #                   force_download=True)
# 
# 
# # snapshot_download(repo_id='meta-llama/Llama-3.1-8B-Instruct',
# #                   repo_type='model',
# #                   local_dir='/data/yangzhifei/pro/ML_SATD/models/Llama-3.1-8B-Instruct',
# #                   token='[REDACTED_R12_SECRET]',
# #                   force_download=True)
# 
# snapshot_download(repo_id='openai/gpt-oss-20b',
#                   repo_type='model',
#                   local_dir='/data/yangzhifei/pro/ML_SATD/models/gpt-oss-20b-2',
#                   token='[REDACTED_R12_SECRET]',
#                   resume_download=True,
#                   force_download=True)
# 
#                 #   token='[REDACTED_R12_SECRET]',
# 
# 

# [R12 修订] 凭据只从环境或Hugging Face登录缓存获取，禁止写入配置/日志。
"""Explicit pinned checkpoint download; importing this file never downloads."""
import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-id', required=True)
    parser.add_argument('--revision', required=True, help='Pinned commit SHA; do not use a moving branch')
    parser.add_argument('--local-dir', required=True, type=Path)
    args = parser.parse_args()
    if len(args.revision) != 40 or any(c not in '0123456789abcdefABCDEF' for c in args.revision):
        raise ValueError('A 40-hex commit revision is required')
    target = args.local_dir.expanduser()
    # Model directories in the copied workspace may point to originals; never write through a symlink.
    if any(part.is_symlink() for part in [target, *target.parents]):
        raise ValueError('Refusing a model target with any symlink ancestor')
    if target.exists() and any(target.iterdir()):
        raise ValueError('Choose a new empty download target')
    from huggingface_hub import snapshot_download
    # None lets the library use an existing login cache; the secret is never serialized.
    snapshot_download(repo_id=args.repo_id, repo_type='model', revision=args.revision,
                      local_dir=target, token=os.environ.get('HF_TOKEN') or None,
                      force_download=False)


if __name__ == '__main__':
    main()
