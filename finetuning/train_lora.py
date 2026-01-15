from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer
)
from peft import get_peft_model
from datasets import load_dataset
from finetuning.lora_config import lora_config

MODEL_NAME = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

model = get_peft_model(model, lora_config)

dataset = load_dataset(
    "json",
    data_files="data/finetune/airline_recovery_dataset.json"
)

def preprocess(example):
    prompt = example["instruction"]
    inputs = tokenizer(prompt, truncation=True, padding="max_length", max_length=256)
    outputs = tokenizer(example["output"], truncation=True, padding="max_length", max_length=256)
    inputs["labels"] = outputs["input_ids"]
    return inputs

tokenized_dataset = dataset.map(preprocess)

training_args = TrainingArguments(
    output_dir="models/finetuned_model",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"]
)

trainer.train()
model.save_pretrained("models/finetuned_model")

print("LoRA fine-tuning completed successfully")
