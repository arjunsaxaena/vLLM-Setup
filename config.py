import os
from dotenv import load_dotenv

load_dotenv()

def get_model_name(default: str = "Qwen/Qwen2-0.5B-Instruct") -> str:
    return os.getenv("VLLM_MODEL_NAME", default)
