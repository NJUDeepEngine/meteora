from huggingface_hub import snapshot_download

# llama2_13b_repo_id = 'hDPQ4gi9BG/MeteoRA_llama2_13b'
llama3_8b_repo_id = 'hDPQ4gi9BG/MeteoRA_llama3_8b'

# snapshot_download(repo_id=llama2_13b_repo_id, local_dir='ckpt/llama2_13b')
snapshot_download(repo_id=llama3_8b_repo_id, local_dir='ckpt/llama3_8b')
