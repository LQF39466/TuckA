import os
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
    LoraConfig,
    VeraConfig,
    TuckAConfig,
    HRAConfig,
    OFTConfig,
    BOFTConfig,
    get_peft_model,
)
from torchvision.transforms import Resize, Compose, ToTensor, Normalize, RandomHorizontalFlip
import evaluate

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


dataset = load_dataset("bitmind/caltech-256")
num_samples_per_class = 10
selected_indices = []
train_dataset = dataset["train"]
def create_subset(dataset, num_samples_per_class=10):
    from collections import defaultdict
    label_to_indices = defaultdict(list)
    dataset_mask = np.ones(len(dataset["train"]), dtype=bool)
    selected_indices = []
    for idx, example in enumerate(dataset["train"]):
        # Caltech256 labels are marked as 1-257, where 257 is background only, hence the "-1"
        label = int(example["filename"][:3]) - 1
        label_to_indices[label].append(idx)

    for label in label_to_indices:
        indices = label_to_indices[label]
        selected = np.random.choice(indices, num_samples_per_class, replace=False)
        dataset_mask[selected] = False
        selected_indices.extend(selected)
    
    return dataset["train"].select(np.where(dataset_mask)[0]).shuffle(seed=42), dataset["train"].select(selected_indices).shuffle(seed=42)

subset_validation, subset_train = create_subset(dataset, num_samples_per_class)

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
        train_transforms(image.convert("RGB")) for image in examples["image"]
    ]
    examples["labels"] = [
        int(filename[:3]) - 1 for filename in examples["filename"]
    ]
    return examples

subset_train = subset_train.map(preprocess_train, batched=True).remove_columns(["image", "filename"])
test_dataset = subset_validation.map(preprocess_train, batched=True).remove_columns(["image", "filename"])

subset_train.set_format("torch", columns=["pixel_values", "labels"])
test_dataset.set_format("torch", columns=["pixel_values", "labels"])

model = AutoModelForImageClassification.from_pretrained(
    model_ckpt,
    num_labels=257,
    ignore_mismatched_sizes=True
)

# Configure adapters
if ADAPTER.startswith("vera"):
    peft_config = VeraConfig(
        r=512,
        vera_dropout=0.1,
        target_modules=["query", "value", "dense"],
    )
elif ADAPTER.startswith("tucka"):
    peft_config = TuckAConfig(
        r=RANK, 
        k=EXPERT_NUM, 
        t=2, 
        p=2,
        log_expert_load=200,
        ec_perturb_scale=10,
        alpha=EXPERT_NUM,
        target_modules=["query", "value", "dense"],
    )
elif ADAPTER.startswith("lora"):
    peft_config = LoraConfig(
        r=RANK,
        target_modules=["query", "value", "dense"],
    )
elif ADAPTER.startswith("dora"):
    peft_config = LoraConfig(
        r=RANK,
        use_dora=True,
        target_modules=["query", "value", "dense"],
    )
elif ADAPTER.startswith("hra"):
    peft_config = HRAConfig(
        r=RANK,
        target_modules=["query", "value", "dense"],
    )
elif ADAPTER.startswith("oft"):
    peft_config = OFTConfig(
        r=0,
        oft_block_size=RANK,
        target_modules=["query", "value", "dense"],
    )
elif ADAPTER.startswith("boft"):
    peft_config = BOFTConfig(
        boft_block_size=RANK,
        target_modules=["query", "value", "dense"],
    )
else:
    raise NotImplementedError

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
for name, param in model.named_parameters():
    if "classifier" in name:
        param.requires_grad = True

training_args = TrainingArguments(
    output_dir="./vit-" + ADAPTER + "-caltech256",
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
    weight_decay=0.01,
    warmup_steps=WARM_UP,
    logging_dir=f"./logs/{ADAPTER}-caltech256-{LR}",
    metric_for_best_model='eval_accuracy',
    label_names=['labels'],
    report_to="tensorboard",
    disable_tqdm=DISABLE_TQDM,
)

metrics = evaluate.load('utils/accuracy.py')
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

train_results = trainer.train()

eval_results = trainer.evaluate()
print(f"Final evaluation results: {eval_results}")

print(f"\n===== Result of {ADAPTER} on caltech256 =====")
best_metric = trainer.state.best_metric
print(f"Best metric: {best_metric:.4f}")