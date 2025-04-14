from huggingface_hub import login, upload_file

token = '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'
login(token=token)

upload_file(
    path_or_fileobj="/root/LayAlign/outputs/LayAlign-xnli-test1/epoch_2_augmentation/pytorch_model.bin",
    path_in_repo="pytorch_model.bin",  # Path where the file will be uploaded in the repo
    repo_id=f"Themira/layalign_xnli",
)