# SPIN-JEvo


[![Preprint DOI](https://img.shields.io/badge/Preprint-10.65215%2FLTSpreprints.2026.01.29.000103-blue)](https://doi.org/10.65215/LTSpreprints.2026.01.29.000103)
[![figshare DOI](https://img.shields.io/badge/figshare-10.6084%2Fm9.figshare.31101862-orange)](https://doi.org/10.6084/m9.figshare.31101862)

SPIN-JEvo is a few-shot virtual directed evolution framework that combines a LoRA-tuned ESM-2 protein language model with a genetic algorithm to explore protein sequence space from small binary-labeled datasets.

This repository keeps a minimal TadA workflow based directly on the original scripts:
1. train a LoRA classifier from an existing labeled CSV
2. run probability-only GA sampling with segmasker filtering from an existing 100-sequence seed pool CSV

## What is used in this repository

- `data/tadA_lora_set.csv`: labeled training set used for LoRA training
- `data/tadA_seed.csv`: 100-sequence initial seed pool used for GA sampling
- `tadA_10_model/`: original LoRA adapter used for sampling

The new model trained by `train_tada_lora.sh` is saved to a new directory and is **not** used by the default sampling script. The default sampling script uses the original `tadA_10_model/` adapter.

## Repository structure

```text
SPIN-JEvo/
├── data/
│   ├── tadA_lora_set.csv
│   └── tadA_seed.csv
├── scripts/
│   ├── train_tada_lora.sh
│   └── evolve_tada_segmasker.sh
├── src/
│   ├── train_lora_classifier.py
│   ├── evolve_with_ga.py
│   └── ga_utils.py
├── checkpoints/
├── outputs/
├── tadA_10_model/
├── environment.yml
├── requirements.txt
└── README.md
```

## Adapter weights

LoRA adapter weights for TadA and CcdA are available on figshare.

**DOI:** https://doi.org/10.6084/m9.figshare.31101862

## Installation

### Conda

```bash
conda env create -f environment.yml
conda activate spin-jevo
```

### Pip

```bash
pip install -r requirements.txt
```

## ESM-2 model path

The scripts resolve the ESM-2 model in this order:

1. `MODEL_PATH`
2. the first script argument
3. local path: `/your_own_path/esm2_t33_650M_UR50D`
4. Hugging Face: `facebook/esm2_t33_650M_UR50D`

## Train a new LoRA model

This step trains a new LoRA model from the existing labeled CSV.

```bash
bash scripts/train_tada_lora.sh
```

Optional:

```bash
MODEL_PATH=/path/to/esm2_t33_650M_UR50D bash scripts/train_tada_lora.sh
TRAIN_CSV=data/tadA_lora_set.csv OUTPUT_DIR=checkpoints/my_new_model bash scripts/train_tada_lora.sh
```

Default input and output:
- input: `data/tadA_lora_set.csv`
- output: `checkpoints/tadA_lora_newmodel/`

## Run GA sampling with the original TadA adapter

This step does **not** use the newly trained model by default. It uses the original `tadA_10_model/` adapter for sampling.

```bash
bash scripts/evolve_tada_segmasker.sh
```

Optional:

```bash
MODEL_PATH=/path/to/esm2_t33_650M_UR50D bash scripts/evolve_tada_segmasker.sh
ADAPTER_PATH=/path/to/tadA_10_model bash scripts/evolve_tada_segmasker.sh
INPUT_CSV=data/tadA_seed.csv OUTPUT_DIR=outputs RUN_TAG=1 bash scripts/evolve_tada_segmasker.sh
```

Default input and output:
- adapter: `tadA_10_model/`
- input seed pool: `data/tadA_seed.csv`
- output: `outputs/tadA_seed_prob_id15_4_ec1.csv`
- log: `outputs/tadA_seed_prob_id15_4_ec1.txt`

## Main settings kept from the original scripts

### LoRA training
- learning rate: `5e-4`
- weight decay: `1e-3`
- epochs: `3`
- train batch size: `1`
- LoRA rank: `16`
- LoRA alpha: `16`
- LoRA dropout: `0.2`
- sequence truncation: `1000`

### GA sampling
- generations: `100`
- batch size: `32`
- initial random jump rate: `0.15`
- crossover mutation rate: `0.02`
- elite divisor: `4`
- acceptance scale: `0.125`
- segmasker filtering: enabled if `segmasker` is available

## Preprint

An earlier version of this work was posted as a preprint:

**Title:** SPIN-dvEvo: Exploration of vast functional sequence space by directed virtual evolution from a local sequence cluster  
**DOI:** https://doi.org/10.65215/LTSpreprints.2026.01.29.000103

**Authors:**  
Zhihang Chen, Jinle Tang, Tingkai Zhang, Xing Zhang, Qinghui Nie, Jian Zhan, Yaoqi Zhou

> This work is currently available as a preprint and has not yet been certified by peer review.

## Citation

If you use this repository, the LoRA weights, or the method in your work, please cite the preprint and figshare record.

### Preprint

```bibtex
@article{chen2026spindvevo,
  title={SPIN-dvEvo: Exploration of vast functional sequence space by directed virtual evolution from a local sequence cluster},
  author={Chen, Zhihang and Tang, Jinle and Zhang, Tingkai and Zhang, Xing and Nie, Qinghui and Zhan, Jian and Zhou, Yaoqi},
  year={2026},
  doi={10.65215/LTSpreprints.2026.01.29.000103}
}
```

### figshare

```bibtex
@misc{spinjevo_figshare_2026,
  title={SPIN-JEvo LoRA adapter weights for TadA and CcdA},
  year={2026},
  doi={10.6084/m9.figshare.31101862}
}
```
