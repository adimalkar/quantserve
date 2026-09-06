"""Download models and tokenizers directly to /mnt/1TB_Drive/Data/MyFiles/models/huggingface."""
import os
import sys

# Set cache directories to 1TB Drive
TARGET_DIR = "/mnt/1TB_Drive/Data/MyFiles/models/huggingface"
os.environ["HF_HOME"] = TARGET_DIR
os.environ["TRANSFORMERS_CACHE"] = TARGET_DIR
os.environ["TORCH_HOME"] = "/mnt/1TB_Drive/Data/MyFiles/models/torch"

print(f"Target download directory set to: {TARGET_DIR}")

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageClassification

MODELS_TO_DOWNLOAD = [
    {
        "name": "Qwen/Qwen2.5-0.5B-Instruct",
        "type": "llm",
        "description": "Compact GQA open-weights LLM (0.5B params, ~950MB)",
    },
    {
        "name": "google/vit-base-patch16-224",
        "type": "vit",
        "description": "Vision Transformer (86M params, ~340MB)",
    },
]


def download_all():
    os.makedirs(TARGET_DIR, exist_ok=True)
    for m in MODELS_TO_DOWNLOAD:
        model_id = m["name"]
        print(f"\n=======================================================")
        print(f"Downloading {model_id} ({m['description']})...")
        print(f"=======================================================")

        if m["type"] == "llm":
            print("Fetching tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=TARGET_DIR)
            print("Fetching model weights...")
            model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=TARGET_DIR)
            print(f"Successfully downloaded {model_id}!")
        elif m["type"] == "vit":
            print("Fetching ViT weights...")
            model = AutoModelForImageClassification.from_pretrained(model_id, cache_dir=TARGET_DIR)
            print(f"Successfully downloaded {model_id}!")

    print("\nAll target models downloaded successfully to /mnt/1TB_Drive/Data/MyFiles/models/huggingface!")


if __name__ == "__main__":
    download_all()
