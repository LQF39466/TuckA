import os, time
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
    DefaultDataCollator
)
from peft import (
    VeraConfig,
    TuckAConfig,
    LoraConfig,
    OFTConfig,
    BOFTConfig,
    HRAConfig,
    get_peft_model,
)
from torchvision.transforms import Resize, Compose, ToTensor, Normalize, RandomHorizontalFlip
import evaluate

from utils.tracemalloc import TorchTracemalloc, b2mb

# ------ Set these parameters before running ------
np.random.seed(42)
RANK=2                          # Rank for adapters, for OFT and BOFT this is the block size 
EXPERT_NUM=5                    # Number of experts for TuckA
ADAPTER=f"tucka-r{RANK}"        # Specify adapter type, can be vera, tucka, lora, dora, hra, oft, or boft
LR=4e-3
TRAIN_EPOCHS=20
EVAL_STEPS=50
WARM_UP=25
DISABLE_TQDM=False
# ------

TARGET_MODULES = ["query", "value", "dense"]

dataset = load_dataset("uoft-cs/cifar100")

num_samples_per_class = 10
selected_indices = []

train_dataset = dataset["train"]
def create_subset(dataset, num_samples_per_class=10):
    from collections import defaultdict
    label_to_indices = defaultdict(list)
    
    for idx, example in enumerate(dataset["train"]):
        label = example["fine_label"]
        label_to_indices[label].append(idx)
    
    selected_indices = []
    for label in label_to_indices:
        indices = label_to_indices[label]
        selected = np.random.choice(indices, num_samples_per_class, replace=False)
        selected_indices.extend(selected)
    
    return dataset["train"].select(selected_indices).shuffle(seed=42)

subset_train = create_subset(dataset, num_samples_per_class)

model_ckpt = "google/vit-base-patch16-224-in21k"
image_processor = AutoImageProcessor.from_pretrained(model_ckpt)

size = (image_processor.size["height"], image_processor.size["width"])

train_transforms = Compose([
    Resize((224, 224)),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # timm IMAGENET_DEFAULT_MEAN & IMAGENET_DEFAULT_STD
])

def preprocess_train(examples):
    examples["pixel_values"] = [
        train_transforms(image.convert("RGB")) for image in examples["img"]
    ]
    return examples

subset_train = subset_train.map(preprocess_train, batched=True).remove_columns(["img", "coarse_label"]).rename_column("fine_label", "labels")
test_dataset = dataset["test"].map(preprocess_train, batched=True).remove_columns(["img", "coarse_label"]).rename_column("fine_label", "labels")

subset_train.set_format("torch", columns=["pixel_values", "labels"])
test_dataset.set_format("torch", columns=["pixel_values", "labels"])

model = AutoModelForImageClassification.from_pretrained(
    model_ckpt,
    num_labels=100,
    ignore_mismatched_sizes=True
)

# Configure adapters
if ADAPTER.startswith("vera"):
    peft_config = VeraConfig(
        r=RANK,
        vera_dropout=0.1,
        target_modules=TARGET_MODULES,
    )
elif ADAPTER.startswith("tucka"):
    peft_config = TuckAConfig(
        r=RANK, 
        k=EXPERT_NUM, 
        t=2, 
        p=2,
        ec_perturb_scale=10,
        log_expert_load=0,
        target_modules=TARGET_MODULES,
    )
elif ADAPTER.startswith("lora"):
    peft_config = LoraConfig(
        r=RANK,
        lora_alpha=16,
        target_modules=TARGET_MODULES,
    )
elif ADAPTER.startswith("dora"):
    peft_config = LoraConfig(
        r=RANK,
        use_dora=True,
        lora_alpha=16,
        target_modules=TARGET_MODULES,
    )
elif ADAPTER.startswith("oft"):
    peft_config = OFTConfig(
        r=0,
        oft_block_size=RANK,
        target_modules=TARGET_MODULES,
    )
elif ADAPTER.startswith("boft"):
    peft_config = BOFTConfig(
        boft_n_butterfly_factor=2,
        boft_block_size=RANK,
        target_modules=TARGET_MODULES,
    )
elif ADAPTER.startswith("hra"):
    peft_config = HRAConfig(
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

time_str = time.strftime("%m-%d-%H-%M", time.localtime())

training_args = TrainingArguments(
    output_dir="./vit-" + ADAPTER + "-cifar100",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    dataloader_num_workers=6,
    dataloader_prefetch_factor=4,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    num_train_epochs=TRAIN_EPOCHS,
    save_strategy="epoch",
    learning_rate=LR,
    logging_steps=20,
    lr_scheduler_type="linear",
    weight_decay=0.,
    warmup_steps=20,
    logging_dir=f"./logs/{ADAPTER}-cifar100-{LR}-{time_str}",
    metric_for_best_model='eval_accuracy',
    label_names=['labels'],
    report_to="tensorboard",
    disable_tqdm=DISABLE_TQDM,
)

metrics = evaluate.load("utils/accuracy.py")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metrics.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=subset_train,
    eval_dataset=test_dataset,
    data_collator=DefaultDataCollator(),
    compute_metrics=compute_metrics,
)

start = time.time()
with TorchTracemalloc() as tracemalloc:
    train_result = trainer.train()
end = time.time()

print(f"\n===== Result of {ADAPTER} on Cifar100 =====")
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