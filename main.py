import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained("skt/kobert-base-v1", torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("skt/kobert-base-v1")
dataset = load_dataset("imdb", split="train[:1%]")
lora_config = LoraConfig(
    init_lora_weights="olora"
)
peft_model = get_peft_model(model, lora_config)
training_args = SFTConfig(dataset_text_field="text", max_seq_length=128)
trainer = SFTTrainer(
    model=peft_model,
    train_dataset=dataset,
    tokenizer=tokenizer,
)
trainer.train()
peft_model.save_pretrained("olora-opt-350m")