import argparse
import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    LoraConfig,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
)

import evaluate
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 5
batch_size = 16

peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1)
lr = 3e-4

if any(k in "skt/kobert-base-v1" for k in ("gpt", "opt", "bloom")):
    padding_side = "left"
else:
    padding_side = "right"

tokenizer = AutoTokenizer.from_pretrained("skt/kobert-base-v1", padding_side=padding_side)
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# 감정 레이블을 숫자로 매핑하는 딕셔너리 추가
sentiment_to_label = {
    "공포": 0,
    "놀람": 1,
    "분노": 2,
    "슬픔": 3,
    "중립": 4,
    "행복": 5,
    "혐오": 6
}

# 레이블 수를 명시적으로 설정
num_labels = len(sentiment_to_label)  # 7개의 감정 클래스

datasets = load_dataset(
    "csv", 
    data_files={
        "train": "./datasets/talk_data.txt", 
        "validation": "./datasets/talk_data.txt"
    },
    delimiter="\t\t"  # 탭 두 개로 구분
)
metric = evaluate.load("accuracy")


def tokenize_function(examples):
    outputs = tokenizer(
        examples["sentence"],
        truncation=True,
        padding=True,  # 'max_length' 대신 True 사용
        max_length=512,  # 2048에서 512로 변경
        return_tensors=None
    )
    # token_type_ids를 올바른 크기로 설정
    outputs["token_type_ids"] = [[0] * len(input_ids) for input_ids in outputs["input_ids"]]
    outputs["labels"] = [sentiment_to_label[label] for label in examples["sentiment"]]
    return outputs


tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=["id", "sentence", "sentiment"],
)

# DataLoader의 collate_fn 수정
def collate_fn(examples):
    # 배치 내의 모든 시퀀스 길이를 동일하게 맞추기
    max_len = max(len(example["input_ids"]) for example in examples)
    
    # 패딩 추가
    for example in examples:
        padding_length = max_len - len(example["input_ids"])
        example["input_ids"].extend([tokenizer.pad_token_id] * padding_length)
        example["attention_mask"].extend([0] * padding_length)
        example["token_type_ids"].extend([0] * padding_length)

    return {
        key: torch.tensor([example[key] for example in examples])
        for key in examples[0].keys()
    }


# Instantiate dataloaders.
train_dataloader = DataLoader(
    tokenized_datasets["train"], 
    shuffle=True, 
    collate_fn=collate_fn, 
    batch_size=batch_size,
    drop_last=True  # 마지막 배치가 불완전한 경우 제외
)

eval_dataloader = DataLoader(
    tokenized_datasets["validation"], 
    shuffle=False, 
    collate_fn=collate_fn, 
    batch_size=batch_size,
    drop_last=True  # 마지막 배치가 불완전한 경우 제외
)

model = AutoModelForSequenceClassification.from_pretrained(
    "skt/kobert-base-v1", 
    num_labels=num_labels,  # 레이블 수 지정
    return_dict=True
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

optimizer = AdamW(params=model.parameters(), lr=lr)

# Instantiate scheduler
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0.06 * (len(train_dataloader) * num_epochs),
    num_training_steps=(len(train_dataloader) * num_epochs),
)



model.to(device)
# 학습 루프
for epoch in range(num_epochs):
    model.train()
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    # 평가 루프
    model.eval()
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = predictions.cpu(), batch["labels"].cpu()
        metric.add_batch(
            predictions=predictions,
            references=references,
        )

    eval_metric = metric.compute()
    print(f"epoch {epoch}:", eval_metric)
    
    # 모델 저장
    save_dir = f"./models/epoch_{epoch}"
    os.makedirs(save_dir, exist_ok=True)
    
    # PEFT 모델 상태 저장
    peft_model_state_dict = get_peft_model_state_dict(model)
    torch.save(peft_model_state_dict, os.path.join(save_dir, "peft_model.bin"))
    
    # 토크나이저 저장
    tokenizer.save_pretrained(save_dir)
    
    print(f"모델이 {save_dir}에 저장되었습니다.")