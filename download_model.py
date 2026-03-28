from modelscope import snapshot_download, AutoTokenizer #snapshot_download（模型快照下载器）；AutoTokenizer（自动分词器）
#AutoModelForCausalLM因果语言模型；TrainingArguments定义训练参数的配置类；Trainer：训练循环的封装类；DataCollatorForSeq2Seq：序列到序列任务的数据整理器
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import torch
import os

"""下载模型到本地--保存到本地--加载模型权重"""

# 获取脚本所在目录，并创建模型缓存路径
script_path = os.path.dirname(os.path.abspath(__file__))
cache_path = os.path.join(script_path, "models")

# 在modelscope上下载Qwen模型到本地目录下，cache_dir=cache_path模型存储路径，revision="master"确保模型版本的确定性和可重现性
model_dir = snapshot_download("Qwen/Qwen3-0.6B", cache_dir=cache_path, revision="master")

# Transformers加载模型权重，use_fast=False使用Python慢速分词器，trust_remote_code=True信任并执行远程自定义代码
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)#必须传入 use_fast=False, trust_remote_code=True才能正确加载分词器
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.bfloat16)#device_map="auto"，自动分配模型到设备，torch_dtype=torch.bfloat16设置模型精度为BF16