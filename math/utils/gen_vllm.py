import argparse
import torch
import sys
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, concatenate_datasets
import multiprocessing
from peft import PeftModel, PeftConfig

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help="")
parser.add_argument('--adapter', type=str, help="TFT adapter path")
parser.add_argument("--data_path", type=str, default="pissa-dataset")
parser.add_argument('--sub_task', nargs='+', help='')
parser.add_argument('--dataset_split', type=str, default="test", help='')
parser.add_argument('--output_file', type=str, default="model_response.jsonl", help="")
parser.add_argument("--batch_size", type=int, default=48, help="")
parser.add_argument('--temperature', type=float, default=0.0, help="")
parser.add_argument('--top_p', type=float, default=1, help="")
parser.add_argument('--max_tokens', type=int, default=1024, help="")
parser.add_argument('--merged', type=bool, default=False, help='TFT adapters cannot be merged due to dynamic routing')
args = parser.parse_args()

def batch_data(data_list, batch_size=1):
    batch_data = []
    for i in range(0, len(data_list), batch_size):
        batch_data.append(data_list[i:i+batch_size])
    return batch_data

def worker_process(device_idx, batch_list, args, output_file):

    torch.cuda.set_device(device_idx)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.merged:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(f"cuda:{device_idx}")
    else:
        config = PeftConfig.from_pretrained(args.adapter)
        base = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, torch_dtype=torch.bfloat16, device_map=None).to(f"cuda:{device_idx}")
        model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    for batch in batch_list:
        batch_query, batch_answer, batch_task = batch
        with torch.no_grad():
            inputs = tokenizer(batch_query, return_tensors="pt", padding=True, truncation=True).to(f"cuda:{device_idx}")
            
            generation_kwargs = {
                "max_new_tokens": args.max_tokens,
                "pad_token_id": tokenizer.eos_token_id,
            }
            if args.temperature > 0:
                generation_kwargs["temperature"] = args.temperature
                generation_kwargs["top_p"] = args.top_p
                generation_kwargs["do_sample"] = True
            else:
                generation_kwargs["do_sample"] = False

            outputs = model.generate(
                **inputs,
                **generation_kwargs
            )
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            stripped = []
            for prompt, output in zip(batch_query, decoded):
                if output.startswith(prompt):
                    stripped.append(output[len(prompt):].lstrip())
                else:
                    stripped.append(output)
        with open(output_file, 'a') as f:
            for query, completion, answer, task in zip(batch_query, stripped, batch_answer, batch_task):
                json.dump({'type': task, 'query': query, 'output': completion, 'answer': answer}, f)
                f.write('\n')

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    if args.sub_task is None:
        dataset = load_dataset(args.data_path, split=args.dataset_split)
    else:
        all_test_dataset = []
        for task in args.sub_task:
            ds = load_dataset(args.data_path, data_dir=task, split=args.dataset_split)
            all_test_dataset.append(ds)
        dataset = concatenate_datasets(all_test_dataset)

    batch_dataset_query = batch_data(dataset["instruction"], batch_size=args.batch_size)
    batch_dataset_answer = batch_data(dataset["output"], batch_size=args.batch_size)
    batch_dataset_task = batch_data(dataset["type"], batch_size=args.batch_size)

    batches = list(zip(batch_dataset_query, batch_dataset_answer, batch_dataset_task))
    num_devices = torch.cuda.device_count()
    split_batches = [[] for _ in range(num_devices)]
    for idx, batch in enumerate(batches):
        split_batches[idx % num_devices].append(batch)

    processes = []
    for device_idx in range(num_devices):
        p = multiprocessing.Process(
            target=worker_process,
            args=(device_idx, split_batches[device_idx], args, args.output_file)
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
