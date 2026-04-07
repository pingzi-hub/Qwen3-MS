import os
import json
import torch
from datasets import Dataset
from modelscope import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, Trainer,TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel
import swanlab
import pandas as pd


swanlab.login(api_key="Jwl0jln3DwyRW7UTEvIXB")
import warnings
warnings.filterwarnings("ignore")

PROMPT = "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。"
# ================= SwanLab =================
swanlab.init(
    project="medical-qwen-lora",  # 必须字符串
    config={
        "model": "Qwen/Qwen3-0.6B",
        "prompt": PROMPT,
        "data_max_length": 256,
    }
)

# ================= 配置 =================
device = torch.device("cpu")
MAX_LENGTH = 256

# ================= 数据格式转换 =================
def dataset_jsonl_transfer(origin_path,new_path):
    messages = []
    with open(origin_path, "r",encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            input = data["question"]
            think = data["think"]
            answer = data["answer"]
            output = f"<think>{think}</answer> \n {answer}"
            #把信息一条一条读取出来，添加到messages中
            messages.append({
                 "instruction":PROMPT,
                "input": f"{input}",
                "output": output,
            })
    #再把信息新建一个文件中
    with open(new_path, "w",encoding="utf-8") as f:
        for m in messages:
            f.write(json.dumps(m,
                               ensure_ascii=False#保留中文，不乱码，不乱意
                               )+"\n")

# ================= 数据预处理 =================
def process_func(example):
    instruction = tokenizer(
        f"<|im_start|>system\n{PROMPT}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}",add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = (instruction["attention_mask"] + response["attention_mask"] + [1])
    labels = [-100] *len(instruction["input_ids"]) + response["input_ids"] + [-100]

    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

# ================= 模型加载 =================
model_name = "Qwen/Qwen3-0.6B"
cache_dir = "./models"
model_dir = snapshot_download(model_name, cache_dir=cache_dir)

tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_dir,
                                            torch_dtype=torch.float32,
                                             device_map="cpu")

# ================= LoRA =================
lora_config = LoraConfig(
        r=2,
        lora_alpha=8,
        target_modules=["q_proj"],
        task_type="CAUSAL_LM"  # 必须加
    )
#把 LoRA 插件装到模型里，以后只训练插件，不训练大模型！
model = get_peft_model(model, lora_config)

# ================= 加载数据=================
# 加载、处理数据集和测试集
train_dataset_path = "train.jsonl"
test_dataset_path = "val.jsonl"

train_jsonl_new_path = "train_format.jsonl"
test_jsonl_new_path = "val_format.jsonl"

if not os.path.exists(train_jsonl_new_path):
    dataset_jsonl_transfer(train_dataset_path, train_jsonl_new_path)
if not os.path.exists(test_jsonl_new_path):
    dataset_jsonl_transfer(test_dataset_path, test_jsonl_new_path)

#得到训练集
train_df = pd.read_json(train_jsonl_new_path,lines=True)
train_ds = Dataset.from_pandas(train_df)
train_dataset = train_ds.map(process_func,remove_columns=train_ds.column_names)

# 得到验证集
eval_df = pd.read_json(test_jsonl_new_path, lines=True)
eval_ds = Dataset.from_pandas(eval_df)
eval_dataset = eval_ds.map(process_func, remove_columns=eval_ds.column_names)


# ================= 训练 =================
args = TrainingArguments(
    output_dir="./output/Qwen3-0.6B",
    per_device_train_batch_size=1,  # i7 跑 1 最合适
    per_device_eval_batch_size=1,
    # ✅ 关键优化：CPU 必须小梯度累积，否则会卡死
    gradient_accumulation_steps=2,  # i7 用 2 平衡速度与显存，不要用 8
    # ✅ 评估策略：CPU 训练慢，不要等 500 步才评估
    eval_strategy="steps",
    eval_steps=200,  # 每 200 步评估一次，兼顾速度和效果
    logging_strategy="steps",
    logging_steps=10,
    bf16=False,  # CPU 关闭 bf16
    bf16_full_eval=False,  # ❌ 必须关闭，否则 CPU 评估报错
    num_train_epochs=2,
    save_steps=400,
    # ✅ 核心降速：CPU 必须用小学习率，否则 loss 直接爆炸
    learning_rate=1e-5,  # 推荐 1e-5 到 2e-5 之间
    save_on_each_node=False,  # 单机器不需要
    gradient_checkpointing=True,  # 开启梯度检查点，省内存
    report_to="swanlab",
    run_name="qwen3-0.6B-i7-cpu",
    fp16=False  # CPU 不支持 fp16 (除非用特殊编译)
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer,padding=True),
)

trainer.can_return_loss = True
trainer.train()
print("开始训练...")

# ================= 保存 =================
model.save_pretrained("./final_model")
tokenizer.save_pretrained("./final_model")
print("✅ 训练完成！")

# ==================== 【融合推理代码】 ====================
def predict(messages,model,tokenizer):
    device = "cpu"
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(model_inputs["input_ids"],max_new_tokens=2048)
    #截掉开头的「输入文本部分」，只保留模型新生成的 token ID。
    generated_ids = [output_ids[len(input_ids):] for input_ids,output_ids in zip(model_inputs["input_ids"], generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

#加载训练好的LoRa模型,因果语言模型
infer_model = AutoModelForCausalLM.from_pretrained(
    model_dir,
torch_dtype=torch.float32)

#专门用来只训练极少参数，就能让大模型学会新技能。
infer_model = PeftModel.from_pretrained(infer_model,
                                        "./final_model")
infer_model.to(device)

#测试问题
test_texts = {
    'instruction': "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。",
    'input': "医生，我最近被诊断为糖尿病，听说碳水化合物的选择很重要，我应该选择什么样的碳水化合物呢？"
}

messages = [
    {"role": "system", "content": test_texts['instruction']},
    {"role": "user", "content": test_texts['input']}
]

#预测输出
print("\n" + "="*60)
print("🩺 模型回答：")
print("="*60)
response = predict(messages, infer_model, tokenizer)
print(response)

swanlab.finish()




