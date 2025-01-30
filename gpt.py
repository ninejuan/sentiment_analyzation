import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_scheduler
from datasets import load_dataset
from tqdm import tqdm

# 디바이스 설정
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 데이터셋 로드 및 전처리
dataset = load_dataset("csv", data_files="./datasets/talk_data.txt", column_names=["sentence", "sentiment"])
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


# 토크나이저 및 모델 초기화
tokenizer = AutoTokenizer.from_pretrained("skt/kobert-base-v1")
model = AutoModelForSequenceClassification.from_pretrained("skt/kobert-base-v1", num_labels=len(sentiment_to_label))
model.to(device)

# 패딩 토큰 설정
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# 토큰화 함수 정의
def tokenize_function(examples):
    outputs = tokenizer(
        examples["sentence"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors=None,
        return_token_type_ids=True
    )
    if "token_type_ids" not in outputs or outputs["token_type_ids"] is None:
        outputs["token_type_ids"] = [0] * len(outputs["input_ids"])
    outputs["labels"] = [sentiment_to_label[label] for label in examples["sentiment"]]
    return outputs

# 데이터셋 전처리
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "sentiment"])
tokenized_datasets.set_format("torch")

train_dataset = tokenized_datasets["train"]

# DataLoader 설정
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# 옵티마이저 및 스케줄러 설정
optimizer = AdamW(model.parameters(), lr=5e-5)
num_training_steps = len(train_dataloader) * 3  # 에포크 수
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# 학습 루프
epochs = 3
progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(epochs):
    for step, batch in enumerate(train_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}  # 배치를 device로 이동
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

# 모델 저장
model.save_pretrained("./models")
tokenizer.save_pretrained("./models")
