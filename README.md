# SPIN-JEvo

[![Preprint DOI](https://img.shields.io/badge/Preprint-10.65215%2FLTSpreprints.2026.01.29.000103-blue)](https://doi.org/10.65215/LTSpreprints.2026.01.29.000103)
[![figshare DOI](https://img.shields.io/badge/figshare-10.6084%2Fm9.figshare.31101862-orange)](https://doi.org/10.6084/m9.figshare.31101862)

**SPIN-JEvo** is a few-shot virtual directed evolution framework that combines a **LoRA-tuned ESM-2 protein language model** with a **genetic algorithm** to efficiently explore protein sequence space and discover high-activity remote homologs from small, binary-labeled sequence datasets. By decoupling sequence exploration from functional constraint, SPIN-JEvo moves beyond the local-search limitations of natural and conventional directed evolution. Starting from functional sequences, it first generates likely nonfunctional variants with approximately **20% random substitutions**, and then evolves them toward remote functional homologs using a genetic algorithm. Using only a few labeled sequences, SPIN-JEvo identified remote homologs within minutes for both **TadA** and **CcdA**, without requiring structural or target-specific information.

This repository provides a minimal and reproducible SPIN-JEvo workflow for **LoRA fine-tuning** from a FASTA file of positive protein sequences and **GA-based sampling with segmasker filtering** from a 20%-mutated seed pool. The current scripts are organized around the **TadA** example, but the workflow can be readily adapted to new protein systems by replacing the input FASTA file.

## Overview

This repository includes:
- FASTA-to-CSV preprocessing for LoRA training
- LoRA fine-tuning of **ESM-2**
- FASTA-to-seed-pool generation for virtual evolution
- Genetic algorithm sampling with **segmasker**
- Final output of evolved sequences and model scores

This repository does **not** require:
- Structural information
- Target-specific structural templates
- Quantitative fitness measurements

## Workflow

Starting from a FASTA file containing positive functional sequences, the training step labels original sequences as 1, generates matched 20%-mutated negative samples labeled as 0, writes all labeled sequences to tadA_lora_set.csv, and uses the full labeled CSV for LoRA training. Starting from the same positive FASTA file, the sampling step generates a 20%-mutated seed pool, writes the seed pool as CSV, uses the LoRA-tuned model to score variants during GA sampling, applies segmasker filtering, and writes final evolved sequences and scores to output files.


```text
SPIN-JEvo/
├── data/
│   └── tadA_VN.fasta
├── scripts/
│   ├── train_tada_lora.sh
│   └── evolve_tada_segmasker.sh
├── src/
│   ├── prepare_fasta_inputs.py
│   ├── train_lora_classifier.py
│   ├── evolve_with_ga.py
│   └── ga_utils.py
├── checkpoints/
├── outputs/
├── environment.yml
├── requirements.txt
└── README.md
````

## Adapter weights

LoRA adapter weights for TadA and CcdA are available on figshare.

**DOI:** [https://doi.org/10.6084/m9.figshare.31101862](https://doi.org/10.6084/m9.figshare.31101862)

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

The scripts use the model in this order:

1. `MODEL_PATH`
2. the first script argument
3. local path: `/your_own_path/esm2_t33_650M_UR50D`
4. Hugging Face: `facebook/esm2_t33_650M_UR50D`

## Quick start

### Use default input

```bash
bash scripts/train_tada_lora.sh
bash scripts/evolve_tada_segmasker.sh
```

### Use your own model path

```bash
MODEL_PATH=/path/to/esm2_t33_650M_UR50D bash scripts/train_tada_lora.sh
MODEL_PATH=/path/to/esm2_t33_650M_UR50D bash scripts/evolve_tada_segmasker.sh
```

### Pass model path as an argument

```bash
bash scripts/train_tada_lora.sh /path/to/esm2_t33_650M_UR50D
bash scripts/evolve_tada_segmasker.sh /path/to/esm2_t33_650M_UR50D
```

### Use your own FASTA

```bash
FASTA_PATH=data/my_protein.fasta bash scripts/train_tada_lora.sh
FASTA_PATH=data/my_protein.fasta bash scripts/evolve_tada_segmasker.sh
```

### Change seed pool size

```bash
SEED_POOL_SIZE=200 bash scripts/evolve_tada_segmasker.sh
```

## Outputs

### Training step

* `data/tadA_lora_set.csv`
* `checkpoints/tadA_lora/`

### Sampling step

* seed pool CSV
* final sampled CSV
* score log / summary text

## Main hyperparameters

### LoRA training

* learning rate: `5e-4`
* weight decay: `1e-3`
* epochs: `3`
* train batch size: `1`
* eval batch size: `1`
* LoRA rank: `16`
* LoRA alpha: `16`
* LoRA dropout: `0.2`

### GA sampling

* generations: `100`
* batch size: `32`
* mutation rate: `0.15`
* elite divisor: `4`
* acceptance scale: `0.125`
* segmasker: enabled

## Preprint

An earlier version of this work was posted as a preprint:

**Title:** SPIN-dvEvo: Exploration of vast functional sequence space by directed virtual evolution from a local sequence cluster
**DOI:** [https://doi.org/10.65215/LTSpreprints.2026.01.29.000103](https://doi.org/10.65215/LTSpreprints.2026.01.29.000103)

**Authors:**
Zhihang Chen, Jinle Tang, Tingkai Zhang, Xing Zhang, Qinghui Nie, Jian Zhan, Yaoqi Zhou

**Category:** Life Sciences × Artificial Intelligence

**Keywords:** Directed evolution; Protein engineering; Protein Language Model

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

## Notes

* This repository focuses on a minimal TadA-centered workflow.
* The method can be adapted to new proteins by replacing the FASTA input.
* Source code and datasets are available at: `https://github.com/chenzh-hash/SPIN-JEvo`

## Contact

For questions regarding the method or repository, please open an issue on GitHub.

```
```

