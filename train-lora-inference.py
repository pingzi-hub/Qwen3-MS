import os
import json
import torch
from datasets import Dataset
from modelscope import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
import swanlab

swanlab.login(api_key="Jwl0jln3DwyRW7UTEvIXB")
import warnings
warnings.filterwarnings("ignore")

# ================= SwanLab =================
swanlab.init(
    project="qwen3-medical-cpu",
    experiment_name="lora-faster-1000",
    config={
        "model": "Qwen3-0.6B",
        "r": 2,
        "lr": 1e-4,
        "max_length": 256,
        "device": "cpu"
    }
)

# ================= 配置 =================
device = torch.device("cpu")
MAX_LENGTH = 256

# ================= 数据格式转换 =================
def dataset_jsonl_transfer(origin_path, new_path):
    messages = []
    with open(origin_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            messages.append({
                "input": data["input"],
                "output": data.get("answer", "")
            })
    with open(new_path, "w", encoding="utf-8") as f:
        for m in messages:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

# ================= 数据预处理 =================
def process_func(example):
    input_text = f"user:{example['input']}  assistant:"
    target_text = example['output']

    input_ids = tokenizer.encode(input_text, add_special_tokens=False)
    target_ids = tokenizer.encode(target_text, add_special_tokens=False)

    input_ids = input_ids + target_ids + [tokenizer.pad_token_id]
    labels = [-100] * len(input_ids[:-len(target_ids)-1]) + target_ids + [tokenizer.pad_token_id]

    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "labels": labels
    }

# ================= 模型加载 =================
model_name = "Qwen/Qwen3-0.6B"
cache_dir = "./models"
model_dir = snapshot_download(model_name, cache_dir=cache_dir)

tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.float32,
    device_map="cpu"
)

# ================= LoRA =================
lora_config = LoraConfig(r=2, lora_alpha=8, target_modules=["q_proj"])
model = get_peft_model(model, lora_config)

# ================= 加载数据（只取2000条！）=================
if not os.path.exists("train_format.jsonl"):
    dataset_jsonl_transfer("train.jsonl", "train_format.jsonl")

with open("train_format.jsonl", "r", encoding="utf-8") as f:
    data_list = [json.loads(line) for line in f]

ds = Dataset.from_list(data_list)
tokenized_ds = ds.map(process_func)
tokenized_ds = tokenized_ds.remove_columns([c for c in tokenized_ds.column_names if c not in ["input_ids", "labels"]])

# ✅ 只训练 2000 条
tokenized_ds = tokenized_ds.select(range(2000))

# ================= 训练 =================
dataloader = DataLoader(
    tokenized_ds,
    batch_size=1,
    shuffle=True,
    collate_fn=DataCollatorForSeq2Seq(tokenizer, padding=True)
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
model.train()

print("开始训练...")
pbar = tqdm(dataloader)
for batch in pbar:
    optimizer.zero_grad()
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()

    # ✅ 实时显示 loss（不报错版本）
    pbar.set_postfix(loss=f"{loss.item():.4f}")
    swanlab.log({"loss": loss.item()})

# ================= 保存 =================
model.save_pretrained("./final_model")
tokenizer.save_pretrained("./final_model")
print("✅ 训练完成！")

swanlab.finish()