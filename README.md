
# MobiZO: Enabling Efficient LLM Fine-Tuning at the Edge via Inference Engines

Code for the paper **[MobiZO: Enabling Efficient LLM Fine-Tuning at the Edge via Inference Engines](https://arxiv.org/abs/2409.15520)**, accepted to EMNLP 2025 (Main).
```
@inproceedings{
gao2025mobizo,
title={MobiZO: Enabling Efficient LLM Fine-Tuning at the Edge via Inference Engines},
author={Lei Gao and Amir Ziashahabi and Yue Niu and Salman Avestimehr and Murali Annavaram},
booktitle={The 2025 Conference on Empirical Methods in Natural Language Processing},
year={2025}
}
```

### Step 1: Create Conda Environment and Install Packages
```bash
conda create -n mobizo python=3.10
conda activate mobizo
pip install -r requirements.txt 
```

### Step 2: Run Experiments 
Detailed hyperparameter configurations can be found in the `experiments` folder. 

The scripts are organized by training type: zero-shot learning, first-order training, and zeroth-order training. Within each category, they are further grouped by model and dataset.

To reproduce the main results in **Table 1**, run:

```bash
sh experiments/zero-shot/all.sh
sh experiments/fo/tiny-llama/all.sh
sh experiments/zo/tiny-llama/all.sh
sh experiments/fo/llama2-7b/all.sh
sh experiments/zo/llama2-7b/all.sh
```

### Step 3: Check `android` Folder for On-device Experiments 

See [the Android guide](android/README.md) to build and run MobiZO training
with ExecuTorch/QNN, including adapter checkpoint save and resume.
