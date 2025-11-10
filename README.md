# Quick Start

Setup environments using anaconda:
```
conda create -n tucka python==3.10
conda activate tucka
pip install -r requirements.txt
bash install_tucka.sh
```

Running mathematical reasoning require a separate environment:
```
conda create -n tucka-math python==3.10
conda activate tucka-math
cd math
pip install -U huggingface_hub
huggingface-cli download --repo-type dataset --resume-download fxmeng/pissa-dataset --local-dir pissa-dataset
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
cd ..
bash install_tucka.sh
```

# Reproduce the Results

The baseline PEFT methods are included in the standard installation of the PEFT package. Our implementation of TuckA requires inserting additional code, so please run `bash install_tucka.sh` from the code directory before attempting to reproduce our results.

The pre-trained model weights and datasets used are all publicly available from the Hugging Face Hub and should be downloaded automatically when you execute the code.

## NLU Tasks & Image Classification

Set hyper parameters in `glue/train-glue.py`, then run:
```
conda activate tucka
python train-glue.py
```

For image classification, set hyper parameters in `image_cls/train-*.py`, then run:
```
conda activate tucka
python train-*.py
```
Replace `*` as the dataset you desired to run.

## Mathematical Reasoning

Set hyper parameters in `math/scripts/run_*.sh`, then make sure your cli is in `./math`, run:
```
conda activate tucka-math
bash scripts/run_*.sh
```
