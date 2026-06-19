import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType

# загрузка токенизатора и модели
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

os.makedirs("adapters/tinyllama-lora-style", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("Загрузка датасета...")

dataset = load_dataset(
    "json", data_files={"train": "data/processed/train.jsonl", "eval": "data/processed/eval.jsonl"}
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)

 # отключаем кэширование для обучения LoRA
model.config.use_cache = False 

print("Загрузка токенизатора и модели...")

tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)

model.config.use_cache = False  # отключаем кэширование для обучения LoRA

print("Добавление LoRA адаптера...")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

def preprocess_function(examples):
    text = (
        "Instruction: " + examples["instruction"] + "\nResponse: " + examples["response"]
    )

    result = tokenizer(text, padding="max_length", truncation=True, max_length=512)
    return result

print("Токенизация данных...")

tokenized_datasets = dataset.map(
    tokenize,
    remove_columns=dataset["train"].column_names,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="results",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=5,
    eval_strategy="steps",
    eval_steps=20,
    save_steps=20,
    save_total_limit=1,
    fp16=torch.cuda.is_available(),
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["eval"],
    data_collator=data_collator,
)

print("Начало обучения...")

trainer.train()

print("Сохранение адаптера...")

model.save_pretrained("adapters/tinyllama-lora-style")
tokenizer.save_pretrained("adapters/tinyllama-lora-style")

print("Обучение завершено. Адаптер сохранен в папке adapters/tinyllama-lora-style")

