
from huggingface_hub import HfApi
api = HfApi()
# Upload all the content from the local folder to your remote Space.
# By default, files are uploaded at the root of the repo
api.create_repo(repo_id="Sombit/orig_fractal_vision_true", repo_type="model")

api.upload_folder(
    folder_path="/home/sombit_dey/vision_code_results/original_ft_frz_vision/orig_fractal_vision_true/hf",
    repo_id="Sombit/orig_fractal_vision_true",
    # repo_type="space",
)