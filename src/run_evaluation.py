import os
import json
import torch
import evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_path = "adapters/tinyllama-lora-style"

test_file = "data/processed/test.jsonl"

if not os.path.exists(test_file):
    test_file = "data/processed/test_prompts.jsonl"

os.makedirs("results", exist_ok=True)


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line))
    return data


def generate_answer(model, tokenizer, instruction):
    prompt = "Instruction: " + instruction + "\nResponse:"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)

    if "Response:" in text:
        text = text.split("Response:")[-1].strip()

    return text


print("Loading test examples...")
test_data = load_jsonl(test_file)[:20]

references = [x["response"] for x in test_data]


print("Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)

base_model.eval()

base_outputs = []

print("Generating base model answers...")

for item in test_data:
    answer = generate_answer(base_model, tokenizer, item["instruction"])
    base_outputs.append(answer)

del base_model

if torch.cuda.is_available():
    torch.cuda.empty_cache()


print("Loading fine-tuned model with LoRA adapter...")

base_for_lora = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)

finetuned_model = PeftModel.from_pretrained(base_for_lora, adapter_path)
finetuned_model.eval()

finetuned_outputs = []

print("Generating fine-tuned model answers...")

for item in test_data:
    answer = generate_answer(finetuned_model, tokenizer, item["instruction"])
    finetuned_outputs.append(answer)


print("Calculating ROUGE...")

rouge = evaluate.load("rouge")

base_scores = rouge.compute(
    predictions=base_outputs,
    references=references
)

finetuned_scores = rouge.compute(
    predictions=finetuned_outputs,
    references=references
)


comparison = []

for i, item in enumerate(test_data):
    comparison.append({
        "instruction": item["instruction"],
        "reference_response": item["response"],
        "base_response": base_outputs[i],
        "finetuned_response": finetuned_outputs[i]
    })


with open("results/base_vs_finetuned.jsonl", "w", encoding="utf-8") as file:
    for row in comparison:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


metrics = {
    "base_model": base_scores,
    "finetuned_model": finetuned_scores,
    "notes": "ROUGE was used to compare generated answers with reference responses on 20 test examples."
}


with open("results/metrics.json", "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=2, ensure_ascii=False)


print("Evaluation finished")
print(json.dumps(metrics, indent=2, ensure_ascii=False))
print("Saved: results/base_vs_finetuned.jsonl")
print("Saved: results/metrics.json")
