from datasets import load_dataset
import json
import os
import re
import random

# тут создается папка для обработанных данных
os.makedirs("data/processed", exist_ok=True)

#загрузка датасета
dataset = load_dataset("databricks/databricks-dolly-15k", split="train")  
rows = []
used = set()

# чистим и фильтруем данные
for item in dataset:
    instruction = item["instruction"]
    response = item["response"]
    context = item["context"]
    
    # убираем лишние пробелы и переносы строк
    instruction = re.sub(r"\s+", " ", instruction).strip()
    response = re.sub(r"\s+", " ", response).strip()

    # добавляем контекст к инструкции, если он есть
    if context:
        context = re.sub(r"\s+", " ", context).strip()
        instruction = instruction + "Context:  " + context
    # фильрация
    if len(instruction) < 10:
        continue

    if len(response) < 20:
        continue
    # -дубликаты
    if instruction.lower() in used:
        continue

    used.add(instruction.lower())
    rows.append({"instruction": instruction, "response": response})

    if len(rows) == 250:
        break

#    перемешиваем до разделения
    random.seed(42)

    random.shuffle(rows)

    # смешиваем данные и делим на обучающую, валидационную и тестовую выборки
    train = rows[:220]
    eval_data = rows[220:]
    test = eval_data[:15]

    def write_jsonl(data, filename):
        with open(filename, "w", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    
write_jsonl(train, "data/processed/train.jsonl")
write_jsonl(eval_data, "data/processed/eval.jsonl")
write_jsonl(test, "data/processed/test.jsonl")

print("Датасет готов. Файлы сохранены в папке data/processed.")
print(f"Количество обучающих примеров: {len(train)}")
print(f"Количество примеров для оценки: {len(eval_data)}")
print(f"Количество тестовых примеров: {len(test)}") 

