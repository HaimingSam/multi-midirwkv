from huggingface_hub import snapshot_download

# 下载 GigaMIDI 数据集到本地
snapshot_download(
    repo_id="Metacreation/GigaMIDI",
    local_dir="./hf-mirrors/GigaMIDI",
    repo_type="dataset",
    resume_download=True,  # 支持断点续传
    token="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"  # 访问令牌
)

# Final_GigaMIDI_V1.1_Final.zip 这个没下，其他都下载了