from typing import List

from transformers import TrainerCallback

import torch
import os
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PrefixTuningConfig,
    TaskType
)
from datasets import Dataset
import pandas as pd
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq, AutoTokenizer, LlamaForCausalLM, AutoModelForCausalLM, LlamaTokenizer
import transformers
from datasets import Dataset, ClassLabel
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wandb

# === 1. Load and preprocess data ===
train_df = pd.read_csv("/home/yena/Classification_module/train_data.csv", encoding="utf-8-sig")
train_df["제목+내용"] = train_df["제목"] + " " + train_df["내용"]
train_df = train_df[["제목+내용", "대분류", "중분류", "소분류"]].astype(str)

valid_df = pd.read_csv("/home/yena/Classification_module/val_data.csv", encoding="utf-8-sig")
valid_df["제목+내용"] = valid_df["제목"] + " " + valid_df["내용"]
valid_df = valid_df[["제목+내용", "대분류", "중분류", "소분류"]].astype(str)

# === 2. Define system prompt (고정) ===
# === 2. Define system prompt (고정) ===
system_prompt = """You are a helpful assistant that classifies documents into the most appropriate label based on their content.

- You must choose **only one** label from the given candidate list.
- Do not make up any new label that is not in the candidate list.
- Your answer must be in **Korean only**.
- Do not include explanations, reasoning, or phrases like "Answer:" — just output the label itself.

Here are some examples:

---

Document: 고기에서 대장균이 검출되었다는 보고가 있습니다.  
Candidate labels: 미생물일반, 대장균, 노로바이러스  
Output: 대장균

---

Document: 시료에서 살모넬라균이 검출되어 회수 조치가 진행 중입니다.  
Candidate labels: 대장균, 살모넬라, 곰팡이수  
Output: 살모넬라

---

Document: 해당 제품은 유통기한이 지난 후 진균수가 기준치를 초과했습니다.  
Candidate labels: 곰팡이수, 진균수, 황색포도상구균  
Output: 진균수

---

Now, classify the following document. Respond with **only one Korean label from the candidates**."""

# === 3. Prompt builder ===
def build_prompt(document: str, candidate_labels: list, label: str):
    user_prompt = f"""
Document: {document}

Candidate labels: {', '.join(candidate_labels)}

Output:"""
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{label}<|eot_id|>"""

# === 4. 문서별 prompt 생성 ===
train_prompts = []
train_labels = []

for idx, row in tqdm(train_df.iterrows()):
    # 같은 대분류 + 중분류 내의 소분류 목록을 후보군으로 사용
    category = row["대분류"]
    subcategory = row["중분류"]
    candidate_labels = sorted(train_df[(train_df["대분류"] == category) & (train_df["중분류"] == subcategory)]["소분류"].unique().tolist())

    prompt = build_prompt(row["제목+내용"], candidate_labels, row["소분류"])
    train_prompts.append(prompt)
    train_labels.append(row["소분류"])  # 참고용, 실제 LLM target은 prompt 마지막에 포함됨
    
train_dataset = Dataset.from_dict({
    "input": train_prompts,
    "output": train_labels,
})

# === 2. Define system prompt (고정) ===
system_prompt = """You are a helpful assistant that classifies documents into the most appropriate label based on their content.

- You must choose **only one** label from the given candidate list.
- Do not make up any new label that is not in the candidate list.
- Your answer must be in **Korean only**.
- Do not include explanations, reasoning, or phrases like "Answer:" — just output the label itself.

Here are some examples:

---

Document: 고기에서 대장균이 검출되었다는 보고가 있습니다.  
Candidate labels: 미생물일반, 대장균, 노로바이러스  
Output: 대장균

---

Document: 시료에서 살모넬라균이 검출되어 회수 조치가 진행 중입니다.  
Candidate labels: 대장균, 살모넬라, 곰팡이수  
Output: 살모넬라

---

Document: 해당 제품은 유통기한이 지난 후 진균수가 기준치를 초과했습니다.  
Candidate labels: 곰팡이수, 진균수, 황색포도상구균  
Output: 진균수

---

Now, classify the following document. Respond with **only one Korean label from the candidates**."""

# === 3. Prompt builder ===
def build_prompt(document: str, candidate_labels: list, label: str):
    user_prompt = f"""
Document: {document}

Candidate labels: {', '.join(candidate_labels)}

Output:"""
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{label}<|eot_id|>"""

# === 4. 문서별 prompt 생성 ===
valid_prompts = []
valid_labels = []

for idx, row in tqdm(valid_df.iterrows()):
    # 같은 대분류 + 중분류 내의 소분류 목록을 후보군으로 사용
    category = row["대분류"]
    subcategory = row["중분류"]
    candidate_labels = sorted(valid_df[(valid_df["대분류"] == category) & (valid_df["중분류"] == subcategory)]["소분류"].unique().tolist())

    prompt = build_prompt(row["제목+내용"], candidate_labels, row["소분류"])
    valid_prompts.append(prompt)
    valid_labels.append(row["소분류"])  # 참고용, 실제 LLM target은 prompt 마지막에 포함됨

valid_dataset = Dataset.from_dict({
    "input": valid_prompts,
    "output": valid_labels,
})

# === 2. 모델 및 토크나이저 로드 ===
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
base_model = "meta-llama/Llama-3.2-3B-Instruct"

model = LlamaForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(
    base_model,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 1. tokenizer 적용 함수 수정
def tokenize(data):
    text = str(data["input"])
    label = str(data["output"])  

    source_ids = tokenizer.encode(text, truncation=True, max_length=512)
    target_ids = tokenizer.encode(label, truncation=True, max_length=128)

    input_ids = source_ids + target_ids + [tokenizer.eos_token_id]
    labels = [-100] * len(source_ids) + target_ids + [tokenizer.eos_token_id]

    return {
        "input_ids": input_ids,
        "labels": labels
    }

train_data = train_dataset.shuffle(seed=42).map(tokenize)
val_data = valid_dataset.shuffle(seed=42).map(tokenize)

# model/data params
base_model = "meta-llama/Llama-3.2-3B-Instruct"
data_path: str = ""
output_dir: str = "./0816_loRA_fine_tuned_model_few"

micro_batch_size: int = 2
gradient_accumulation_steps: int = 4
num_epochs: int = 3
learning_rate: float = 3e-4
val_set_size: int = 2000

# lora hyperparams
lora_r: int = 8
lora_alpha: int = 16
lora_dropout: float = 0.05
lora_target_modules: List[str] = [
    "q_proj",
    "v_proj",
]

config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    target_modules=lora_target_modules,
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)

# Step 4: Initiate the trainer
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

class WandbLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            wandb.log(logs, step=state.global_step)

wandb.init(
    project="food_classification_llama3_lora_few",
    name="20240816_llama3-3b_lora_cls_v1_few"
)

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=100,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=10,
        optim="adamw_torch",
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=200,
        save_steps=200,
        output_dir=output_dir,
        save_total_limit=3
    ),
    data_collator=transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    ),
    callbacks=[WandbLoggingCallback()], 
)

trainer.train()

# 학습 완료 후 모델 저장
model.save_pretrained("./0816_loRA_fine_tuned_model_few")
tokenizer.save_pretrained("./0816_loRA_fine_tuned_tokenizer_few")