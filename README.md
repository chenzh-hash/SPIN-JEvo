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


## Repository structure

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

##Adapter weights

LoRA adapter weights for TadA and CcdA are available on figshare:

DOI: https://doi.org/10.6084/m9.figshare.31101862

## Installation
## Conda
conda env create -f environment.yml
conda activate spin-jevo
Pip
pip install -r requirements.txt
## ESM-2 model path

The scripts use the model in this order:

1. MODEL_PATH

2. the first script argument

3. local path: /your_own_path/esm2_t33_650M_UR50D

4. Hugging Face: facebook/esm2_t33_650M_UR50D
