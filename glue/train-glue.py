import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from datasets import load_dataset
from peft import (
    VeraConfig,
    TuckAConfig,
    LoraConfig,
    OFTConfig,
    BOFTConfig,
    HRAConfig,
    get_peft_model, 
    TaskType,
)

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DefaultDataCollator
)

import evaluate
import numpy as np

from utils.task_mapping import (
    TASK_TO_INPUT_NAME_1_MAPPING, 
    TASK_TO_INPUT_NAME_2_MAPPING, 
    TASK_TO_REMOVE_COLUMNS_MAPPING,
)

from utils.tracemalloc import TorchTracemalloc, b2mb

# ------ Set these parameters before running ------
TASK_NAME = "qqp"                           # GLUE Task name, can be cola, mrpc, qnli, qqp, rte or sst2
RANK = 8                                    # Rank for adapters, for OFT and BOFT this is the block size
EXPERT_NUM=2                                # Number of experts for TuckA
ALPHA=2                                     # Alpha for LoRA, DoRA and TuckA
ADAPTER = f"tucka-r{RANK}-k{EXPERT_NUM}"    # Specify adapter type, can be vera, tucka, lora, dora, hra, oft, or boft
TRAIN_EPOCHS = 5
EVAL_STEP = 500
LR = 6e-3
WARM_UP = 500
DISABLE_TQDM = True
# ------

TARGET_MODULES = ["query_proj", "value_proj", "dense"]

# Load dataset and model
dataset = load_dataset("nyu-mll/glue", TASK_NAME)
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-base")

# Data preprocess
SEP_TOKEN = "<SEP>"
tokenizer.pad_token = tokenizer.eos_token

def preprocess_function(examples):
    if TASK_NAME in TASK_TO_INPUT_NAME_2_MAPPING.keys():
        inputs = [s1 + SEP_TOKEN + s2 for s1, s2 in zip(examples[TASK_TO_INPUT_NAME_1_MAPPING[TASK_NAME]], examples[TASK_TO_INPUT_NAME_2_MAPPING[TASK_NAME]])]
    else:
        inputs = examples[TASK_TO_INPUT_NAME_1_MAPPING[TASK_NAME]]
    return tokenizer(
        inputs,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=TASK_TO_REMOVE_COLUMNS_MAPPING[TASK_NAME]
)

# Configure adapters
if ADAPTER.startswith("vera"):
    peft_config = VeraConfig(
        task_type=TaskType.SEQ_CLS,
        r=RANK,
        vera_dropout=0.1,
        target_modules=TARGET_MODULES,
    )
elif ADAPTER.startswith("tucka"):
    peft_config = TuckAConfig(
        task_type=TaskType.SEQ_CLS, 
        r=RANK, 
        k=EXPERT_NUM, 
        t=2, 
        p=2,
        alpha=ALPHA,
        ec_perturb_scale=10,
        log_expert_load=1000,
        target_modules=TARGET_MODULES,
    )
elif ADAPTER.startswith("lora"):
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=RANK,
        lora_alpha=ALPHA,
        target_modules=TARGET_MODULES,
    )
elif ADAPTER.startswith("dora"):
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=RANK,
        use_dora=True,
        lora_alpha=ALPHA,
        target_modules=TARGET_MODULES,
    )
elif ADAPTER == "oft":
    peft_config = OFTConfig(
        task_type=TaskType.SEQ_CLS,
        r=0,
        oft_block_size=RANK,
        target_modules=TARGET_MODULES,
    )
elif ADAPTER == "boft":
    peft_config = BOFTConfig(
        task_type=TaskType.SEQ_CLS,
        boft_n_butterfly_factor=2,
        boft_block_size=RANK,
        target_modules=TARGET_MODULES,
    )
elif ADAPTER == "hra":
    peft_config = HRAConfig(
        task_type=TaskType.SEQ_CLS,
        r=RANK,
        target_modules=TARGET_MODULES,
    )
else:
    raise NotImplementedError

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
for name, param in model.named_parameters():
    if "classifier" in name:
        param.requires_grad = True

acc = evaluate.load('utils/accuracy.py')
f1 = evaluate.load('utils/f1.py')
mcc = evaluate.load('utils/matthews_correlation.py')

def compute_f1_and_acc(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    _acc = acc.compute(predictions=predictions, references=labels)['accuracy']
    _f1 = f1.compute(predictions=predictions, references=labels)['f1']
    return {"accuracy": _acc, "f1": _f1, "avg_metric": (_f1 + _acc)/2}

def compute_acc(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return acc.compute(predictions=predictions, references=labels)

def compute_mcc(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    _acc = acc.compute(predictions=predictions, references=labels)['accuracy']
    _mcc = mcc.compute(predictions=predictions, references=labels)['matthews_correlation']
    return {"accuracy": _acc, "mcc": _mcc}
    

if TASK_NAME == "qqp" or TASK_NAME == "mrpc":
    compute_metrics = compute_f1_and_acc
    metric_for_best_model = "eval_avg_metric"
elif TASK_NAME == "cola":
    compute_metrics = compute_mcc
    metric_for_best_model = "eval_mcc"
else:
    compute_metrics = compute_acc
    metric_for_best_model = "eval_accuracy"

training_args = TrainingArguments(
    output_dir="./debert-" + ADAPTER + "-" + TASK_NAME,
    num_train_epochs=TRAIN_EPOCHS,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=1,
    eval_strategy="steps",
    eval_steps=EVAL_STEP,
    save_strategy="epoch",
    learning_rate=LR,
    weight_decay=0.01,
    adam_beta2=0.999,
    lr_scheduler_type="linear",
    warmup_steps=WARM_UP,
    logging_dir="./logs/" + ADAPTER + '-' + TASK_NAME,
    logging_steps=100,
    metric_for_best_model=metric_for_best_model,  # 修改为 eval_accuracy
    report_to="tensorboard",
    disable_tqdm=DISABLE_TQDM,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=DefaultDataCollator(),
    compute_metrics=compute_metrics,
)
start = time.time()
with TorchTracemalloc() as tracemalloc:
    train_result = trainer.train()
end = time.time()

print(f"\n===== Result of {ADAPTER} on {TASK_NAME} =====")
best_metric = trainer.state.best_metric
best_checkpoint = trainer.state.best_model_checkpoint
print(f"Best metric: {best_metric:.4f} achieved at {best_checkpoint}")

print(f"\n===== Training info =====")
trainer.log_metrics("train", train_result.metrics)
print(f"Traning time {end-start}s")
print(f"GPU Memory before entering the train : {b2mb(tracemalloc.begin)}")
print(f"GPU Memory consumed at the end of the train (end-begin): {tracemalloc.used}")
print(f"GPU Peak Memory consumed during the train (max-begin): {tracemalloc.peaked}")
print(f"GPU Total Peak Memory consumed during the train (max): {tracemalloc.peaked + b2mb(tracemalloc.begin)}")
print(f"CPU Memory before entering the train : {b2mb(tracemalloc.cpu_begin)}")
print(f"CPU Memory consumed at the end of the train (end-begin): {tracemalloc.cpu_used}")
print(f"CPU Peak Memory consumed during the train (max-begin): {tracemalloc.cpu_peaked}")
print(f"CPU Total Peak Memory consumed during the train (max): {tracemalloc.cpu_peaked + b2mb(tracemalloc.cpu_begin)}")