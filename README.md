# SPIN-JEvo

Code and data resources for **Diversity-Oriented Protein Sequence Discovery with Sparse Functional Feedback: Jump-Initialized Genetic Search Beyond Natural Sequence Neighborhoods**.

SPIN-JEvo combines an ESM-2-based, LoRA-adapted sequence-ranking model with jump-initialized genetic search. This repository provides computational implementations, training inputs, and configuration records. The model score is used as a task-specific ranking signal, not as a calibrated probability of experimental activity.

## Code, tests, data, and model weights

The release is organized into separate resources:

- **GitHub code and training inputs** — source code, command-line wrappers, configuration records, and available training/seed datasets. The prepared archive is named `SPIN-JEvo_GitHub_no_weights.zip`.
- **Tests and analysis scripts** — software tests and archived benchmark, analysis, and plotting scripts, distributed separately as `SPIN-JEvo_test_scripts.zip`.
- **Study data** — sequence collections, measurement tables, and analysis-ready results, distributed separately as `SPIN-JEvo_website_data.zip` for the [laboratory download page](https://zhouyq-lab.szbl.ac.cn/download/).
- **LoRA adapter weights** — TadA and CcdA model files hosted on [Figshare](https://doi.org/10.6084/m9.figshare.31101862).

**The website data archive does not include the GitHub code archive or the test-script archive.** Some training and reference-seed files are intentionally shared between the code and data distributions. Model weights are not duplicated in any of the three archives. Manuscript documents and reviewer-response letters are not included.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── environment.yml
├── configs/
│   └── paper_protocol.json
├── src/
│   ├── train_lora_classifier.py
│   ├── evolve_with_ga.py
│   └── ga_utils.py
├── scripts/
│   ├── train_tada_lora.sh
│   └── evolve_tada_segmasker.sh
├── data/
│   ├── README.md
│   ├── training/
│   └── reference_seeds/
└── docs/
    └── weights_figshare.json
```

The archive contains a top-level `github_ready/` directory. Place its **contents** at the repository root when updating GitHub.

## Installation

Use Python 3.10 or later in a dedicated environment:

```bash
git clone https://github.com/chenzh-hash/SPIN-JEvo.git
cd SPIN-JEvo
python -m pip install -r requirements.txt
```

The base encoder is `facebook/esm2_t33_650M_UR50D` (ESM-2 650M). A compatible local checkpoint can also be supplied. GPU execution requires an appropriate PyTorch/CUDA installation. The search implementation uses NCBI `segmasker` by default; this is a separate dependency.

Dependency files specify software requirements; they are not a complete lockfile of the historical computing environment.

## Computational entry points

- `src/train_lora_classifier.py` trains the LoRA-adapted encoder and classification head from a labeled CSV.
- `src/evolve_with_ga.py` performs offline, sequence-ranking-guided genetic search using an adapter directory and an input CSV.
- `src/ga_utils.py` implements the genetic operators.
- `scripts/` contains wrappers for the TadA workflow.

Inspect the available arguments before running:

```bash
python src/train_lora_classifier.py --help
python src/evolve_with_ga.py --help
```

Training CSVs use `Sequence` and `label` columns by default. Synthetic label-0 contrastive sequences are not experimentally confirmed inactive sequences. The positive-only seed files are not complete binary training sets.

The training wrapper defaults to `data/training/tada_round1_train_historical.csv`. The search wrapper requires an explicit `ADAPTER_PATH`; no adapter is bundled. Download the relevant files from Figshare and retain their configuration files together, or supply a compatible locally trained adapter. Consult the Figshare record for model identification rather than inferring it from repeated filenames.

All sequence and model outputs are intended for offline computational research and evaluation. No experimental procedures are provided.

## Training inputs and configuration records

See [data/README.md](data/README.md) for file-level provenance and [configs/paper_protocol.json](configs/paper_protocol.json) for recorded settings.

- `tada_round1_train_historical.csv` contains the archived 20-record TadA Round I training input.
- `tada_positive_10.csv` and `ccda_positive_22.csv` contain the positive seed records from the final Supporting Information.
- `tada_round2_train.csv` is a preserved 111-record historical snapshot, not a verified reconstruction of the manuscript's 20+60-record cumulative update.
- `data/tadA_lora_set.csv` is a legacy 40-record example, not the historical 20-record Round I input.
- An exact CcdA binary training snapshot is not included; its positive seed records must not be treated as the complete training set.

The original TadA searches used populations of 100 and retained 100 final records per run. Controlled supplementary searches used populations of 10. These configurations are distinct. The release preserves available historical inputs and does not claim exact reconstruction of every published model or run.

## Study data archive

The separate website data archive contains:

- Natural TadA references from UniRef100 2021_04 and 2026_03: 39,214 and 35,964 sequences.
- Natural CcdA C-terminal 36-residue references: 2,655 and 1,985 sequences, respectively, plus the corresponding full-length references.
- Existing TadA Round I/II generated populations (997/845 sequences) and experimentally active sets (23/31 variants).
- The existing CcdA sampled library (3,041 sequences) and high-fitness set (65 sequences).
- Supporting Information Tables S1–S8 in CSV format.
- Analysis-ready tables for search trajectories, structural scores, sequence identities, projections, and phylogenetic analyses.

Within that archive, start with `README.md` and `DATASET_INDEX.json`. Current primary tables are under `analysis_ready/`; older results remain in dated directories for provenance. The archive does not bundle full UniRef databases, raw structural-model files, sequencing reads, or model weights.

The reconstructed 2021 reference sets are historical family references, not a verified copy of the exact ESM-2 pretraining corpus. UMAP and t-SNE are descriptive visualizations; sequence-identity statistics provide separate quantitative comparisons.

## Tests and analysis scripts

Extract `SPIN-JEvo_test_scripts.zip` separately. From its `test_scripts/` directory, run the lightweight genetic-operator tests:

```bash
python -m unittest discover -s tests -p test_ga_utils.py -v
```

The archive also contains analysis and plotting implementations. See its README for input and dependency requirements. Some archived scripts retain historical filesystem paths and require adaptation to a local installation. Passing software tests does not establish biological predictive accuracy or complete end-to-end reproduction.

## Model weights and citation

**TadA and CcdA LoRA adapter weights:** [Figshare, DOI: 10.6084/m9.figshare.31101862](https://doi.org/10.6084/m9.figshare.31101862).

**Earlier preprint:** Chen, Z.; Tang, J.; Zhang, T.; Zhang, X.; Nie, Q.; Zhan, J.; Zhou, Y. *SPIN-dvEvo: Exploration of vast functional sequence space by directed virtual evolution from a local sequence cluster* (2026). [DOI: 10.65215/LTSpreprints.2026.01.29.000103](https://doi.org/10.65215/LTSpreprints.2026.01.29.000103).

Please cite the relevant study and Figshare record when using these resources. The earlier preprint retains its original title. Base-model weights and third-party components remain subject to their respective licenses and terms.
