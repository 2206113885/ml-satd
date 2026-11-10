from huggingface_hub import snapshot_download

# snapshot_download(repo_id='Qwen/Qwen3-8B',
#                   repo_type='model',
#                   local_dir='/data/yangzhifei/pro/ML_SATD/models/Qwen3-8B',
#                   force_download=True)


# not complate
# snapshot_download(repo_id='Qwen/Qwen3-14B',
#                   repo_type='model',
#                   local_dir='/data/yangzhifei/pro/ML_SATD/models/Qwen3-14B',
#                   token='[REDACTED_R12_SECRET]',
#                   force_download=True)


# snapshot_download(repo_id='Qwen/Qwen2.5-Coder-7B-Instruct',
#                   repo_type='model',
#                   local_dir='/data/yangzhifei/pro/ML_SATD/models/Qwen2.5-Coder-7B-Instruct',
#                   force_download=True)


# snapshot_download(repo_id='meta-llama/Llama-3.1-8B-Instruct',
#                   repo_type='model',
#                   local_dir='/data/yangzhifei/pro/ML_SATD/models/Llama-3.1-8B-Instruct',
#                   token='[REDACTED_R12_SECRET]',
#                   force_download=True)

snapshot_download(repo_id='openai/gpt-oss-20b',
                  repo_type='model',
                  local_dir='/data/yangzhifei/pro/ML_SATD/models/gpt-oss-20b-2',
                  token='[REDACTED_R12_SECRET]',
                  resume_download=True,
                  force_download=True)

                #   token='[REDACTED_R12_SECRET]',


