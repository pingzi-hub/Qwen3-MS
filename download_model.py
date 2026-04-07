from modelscope import snapshot_download, AutoTokenizer
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import torch
import os

# ====================== 🔥 自动判断 CPU/GPU ======================
USE_CPU = not torch.cuda.is_available()  # 自动判断是否用CPU

# 获取脚本所在目录
script_path = os.path.dirname(os.path.abspath(__file__))
cache_path = os.path.join(script_path, "models")

# 下载模型
model_dir = snapshot_download("Qwen/Qwen3-0.6B", cache_dir=cache_path, revision="master")

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)

# ====================== 🔥 关键：自动适配设备 ======================
if USE_CPU:
    # ✅ CPU 环境
    print("✅ 检测到 CPU，使用 CPU 运行")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="cpu",  # 强制CPU
        torch_dtype=torch.float32  # CPU只能用float32
    )
else:
    # ✅ GPU 环境
    print("✅ 检测到 GPU，使用 GPU 运行")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

# 输出当前设备
print("✅ 模型运行在：", model.device)