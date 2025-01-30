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
num_epochs = 10000
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
        padding="max_length",
        max_length=2048,
        return_tensors=None,
        return_token_type_ids=True  # token_type_ids 명시적으로 추가
    )
    # token_type_ids가 없는 경우 0으로 채워진 리스트 추가
    if "token_type_ids" not in outputs:
        outputs["token_type_ids"] = [0] * len(outputs["input_ids"])
    outputs["labels"] = [sentiment_to_label[label] for label in examples["sentiment"]]
    return outputs


tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=["id", "sentence", "sentiment"],
)

# DataLoader의 collate_fn 수정
def collate_fn(examples):
    return {
        key: torch.tensor([example[key] for example in examples])
        for key in examples[0].keys()
    }


# Instantiate dataloaders.
train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=batch_size)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], shuffle=False, collate_fn=collate_fn, batch_size=batch_size
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

# 모델 저장을 위한 변수 초기화
best_accuracy = 0.0
output_dir = "./models"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model.to(device)
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    # 평가 모드
    model.eval()
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = predictions, batch["labels"]
        metric.add_batch(
            predictions=predictions,
            references=references,
        )

    eval_metric = metric.compute()
    current_accuracy = eval_metric["accuracy"]
    print(f"epoch {epoch}: loss = {total_loss/len(train_dataloader):.4f}, accuracy = {current_accuracy:.4f}")

    # 현재 모델이 이전 최고 성능보다 좋으면 저장
    if current_accuracy > best_accuracy:
        best_accuracy = current_accuracy
        # 모델 저장
        model_save_path = os.path.join(output_dir, f"best_model_acc_{best_accuracy:.4f}")
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        print(f"New best model saved with accuracy: {best_accuracy:.4f}")

print(f"Training completed! Best accuracy: {best_accuracy:.4f}")