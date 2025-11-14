<div align=center>

# [AAAI 2026] TuckA: Hierarchical Compact Tensor Experts for Efficient Fine-Tuning

[![arXiv](https://img.shields.io/badge/arXiv-2511.06859-b31b1b?style=flat&logo=arxiv)](http://arxiv.org/abs/2511.06859)

</div>

<div align="center">
  <img src="rep_assets/framework_newcolor.png" width="1100"/>
</div>

## Introduction
This repository includes the official implementation of TuckA. We provide a brief walkthrough of the code [here](https://github.com/LQF39466/TuckA/blob/main/peft/tuners/TuckA/README.md).

## Quick Start

To set up the environment using Anaconda:
```
conda create -n tucka python==3.10
conda activate tucka
pip install -r requirements.txt
bash install_tucka.sh
```

Running the mathematical reasoning task requires a separate environment:
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

## Reproduce the Results

The baseline PEFT methods are included in the standard PEFT package installation. Our implementation of TuckA requires adding some code, so please run `bash install_tucka.sh` from the root directory of this repository before attempting to reproduce our results.

The pre-trained model weights and datasets used are all publicly available from the Hugging Face Hub and will be downloaded automatically when you run the code.

### NLU Tasks & Image Classification

Set the hyperparameters in `glue/train-glue.py`, then run:
```
conda activate tucka
python train-glue.py
```

For image classification, set the hyperparameters in `image_cls/train-*.py`, then run:
```
conda activate tucka
python train-*.py
```
Replace `*` with the dataset you want to run.

### Mathematical Reasoning

Set the hyperparameters in `math/scripts/run_*.sh`, then make sure you are under the `./math` directory and run:
```
conda activate tucka-math
bash scripts/run_*.sh
```

### Acknowledgement

Our implementation is based on the [Hugging Face PEFT](https://huggingface.co/docs/peft/index) library. The code for the mathematical reasoning task is built upon the [PiSSA](https://github.com/GraphPKU/PiSSA) repository. We thank the authors for making their code publicly available.

### Citations

If you find this work useful please consider citing our paper as:

```
@inproceedings{lei2026tucka,
  title={TuckA: Hierarchical Compact Tensor Experts for Efficient Fine-Tuning},
  author={Qifeng Lei and Zhiyong Yang and Qianqian Xu and Cong Hua and Qingming Huang},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2026},
}
```

### Contact

If you encounter any problems using the code or reproducing our results, please open an issue in this repository. We will respond as soon as possible.
