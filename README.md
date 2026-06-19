# Дообученная языковая модель

## Использованный датасет

Я использовала открытый датасет с HuggingFace:

databricks/databricks-dolly-15k

Из датасета были выбраны instruction-response пары.
Сохранила в формате JSONL

### Preprocessing (шаг 1)

Подготовка данных есть в src/prepare_dataset.py

На данном этапе я:
- извлекла поля instructions, context и response;
- обьединила context with instruction, если context есть;
- очистила лишние пробелы и переносы строк;
- удалила слишком короткие instruction and response;
- удалила дубликаты по instruction;

Готовые файлы:

- data/processed/train.jsonl
- data/processed/eval.jsonl
- data/processed/test.jsonl

### Fine-tuning (шаг 2)

В качестве базовой модели была использована открытая модель: 
TinyLlama/TinyLlama-1.1B-Chat-v1.0

Дообучение выполняется в файле:

src/train_lora.py

Для обучения использовались:

- Hugging Face Transformers;
- Hugging Face PEFT;
- LoRA adapter;
- Google Colab T4 GPU.

Основные веса базовой модели не изменялись, обучались только LoRA-адаптеры

Параметры LoRA:

- r = 8
- lora_alpha = 16
- lora_dropout = 0.05
- target modules: q_proj, v_proj, k_proj, o_proj

Путь к сохранённому адаптеру:

adapters/tinyllama-lora-style

В папке адаптера находятся файлы:

* adapter_config.json
* adapter_model.safetensors
* tokenizer.json
* tokenizer_config.json

Логирование loss

Во время обучения логировались значения loss и eval_loss. Скриншот обучения сохранён в папке:

screenshots/training_loss.png

По результатам обучения:

* final train loss: примерно 1.773
* final eval loss: примерно 1.835

Скриншот сохранённого адаптера:

screenshots/saved_adapter.png

### Оценка (шаг 3)

Оценка выполняется в файле:

src/run_evaluation.py

Для оценки были сравнены две модели:

1. базовая модель TinyLlama/TinyLlama-1.1B-Chat-v1.0;
2. та же модель с подключённым LoRA-адаптером.

Сравнение проводилось на 20 тестовых instruction-response примерах.

Для каждого примера были сохранены:

* instruction;
* reference response;
* ответ базовой модели;
* ответ дообученной модели.

Файл с результатами сравнения:

results/base_vs_finetuned.jsonl

Метрика

Для автоматической оценки была использована метрика ROUGE.

Файл с метриками:

results/metrics.json

Скриншот с результатами оценки сохранён в:

screenshots/evaluation_metrics.png

# Вывод

Результат не является идеальным, так как использовался небольшой датасет и лёгкая модель TinyLlama. Однако проект показывает полный процесс fine-tuning: от подготовки данных до оценки качества модели.
