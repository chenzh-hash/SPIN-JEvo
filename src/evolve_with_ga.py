"""Run SPIN-JEvo sequence-ranking-guided genetic search.

The model output is treated as a task-specific ranking score.  It is not a
calibrated probability of experimental activity.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoTokenizer, EsmForSequenceClassification

from ga_utils import AMINO_ACIDS, mutate_crossover


os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"

MODEL_POS_INDEX = 1
ESM_MAX_LEN = 1022


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LoRA-ranking-guided GA sequence sampling."
    )
    parser.add_argument("--model-name", required=True, help="Base ESM-2 model name or local path.")
    parser.add_argument("--adapter-path", required=True, help="Path to the LoRA adapter directory.")
    parser.add_argument("--input-csv", required=True, help="Seed CSV containing Sequence and label columns.")
    parser.add_argument("--output-dir", required=True, help="Directory for generated populations and logs.")
    parser.add_argument("--run-tag", default="1", help="Identifier used in output filenames.")
    parser.add_argument("--sequence-column", default="Sequence")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--positive-label", type=int, default=1)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--initial-substitution-fraction", type=float, default=0.20)
    parser.add_argument("--mutation-rate", type=float, default=0.02)
    parser.add_argument(
        "--mating-pool-fraction",
        type=float,
        default=0.75,
        help=(
            "Highest-scoring population fraction eligible as parents. "
            "The archived implementation discarded the bottom quartile, "
            "therefore retaining 0.75."
        ),
    )
    parser.add_argument("--acceptance-scale", type=float, default=0.125)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=1000)
    parser.add_argument("--segmasker-cutoff", type=int, default=5)
    parser.add_argument("--disable-segmasker", action="store_true")
    parser.add_argument("--seed", type=int, default=8893)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ["initial_substitution_fraction", "mutation_rate", "mating_pool_fraction"]:
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 1")
    if args.generations < 1:
        raise ValueError("--generations must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.acceptance_scale < 0.0:
        raise ValueError("--acceptance-scale must be non-negative")


def has_long_low_complexity_segment(sequence: str, cutoff: int) -> bool:
    """Return whether NCBI segmasker reports a segment longer than ``cutoff``."""
    with tempfile.TemporaryDirectory(prefix="spin_jevo_segmasker_") as tmp:
        fasta_path = Path(tmp) / "sequence.fasta"
        interval_path = Path(tmp) / "sequence.interval"
        fasta_path.write_text(f">query\n{sequence}\n")
        subprocess.run(
            ["segmasker", "-in", str(fasta_path), "-out", str(interval_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for line in interval_path.read_text().splitlines():
            if " - " not in line:
                continue
            start, end = map(int, line.strip().split(" - "))
            if end - start + 1 > cutoff:
                return True
    return False


def mutate_initial_sequence(sequence: str, fraction: float, rng: np.random.Generator) -> str:
    child = list(sequence)
    count = int(round(len(child) * fraction))
    if count == 0:
        return sequence
    positions = rng.choice(len(child), size=count, replace=False)
    for position in positions:
        choices = [aa for aa in AMINO_ACIDS if aa != child[position]]
        child[position] = str(rng.choice(choices))
    return "".join(child)


@torch.inference_mode()
def scores_for_sequences(model, tokenizer, sequences, device, batch_size, max_length):
    values = []
    for start in range(0, len(sequences), batch_size):
        chunk = sequences[start : start + batch_size]
        tokens = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=min(max_length, ESM_MAX_LEN),
        ).to(device)
        logits = model(**tokens).logits
        scores = torch.softmax(logits.float(), dim=-1)[:, MODEL_POS_INDEX]
        values.append(scores.cpu().numpy())
    return np.concatenate(values) if values else np.zeros(0, dtype=np.float32)


def main() -> None:
    args = parse_args()
    validate_args(args)
    py_rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    base_model = EsmForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PeftModel.from_pretrained(base_model, args.adapter_path).to(device).eval()

    frame = pd.read_csv(args.input_csv)
    missing = {args.sequence_column, args.label_column} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required seed columns: {sorted(missing)}")
    frame = frame[frame[args.label_column].eq(args.positive_label)].copy()
    if len(frame) < 2:
        raise ValueError("At least two positive seed sequences are required")
    frame = frame.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    frame["Sequence"] = frame[args.sequence_column].astype(str).str.slice(stop=args.max_length)
    frame["Sequence"] = [
        mutate_initial_sequence(seq, args.initial_substitution_fraction, np_rng)
        for seq in frame["Sequence"]
    ]
    frame["score"] = scores_for_sequences(
        model,
        tokenizer,
        frame["Sequence"].tolist(),
        device,
        args.batch_size,
        args.max_length,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{Path(args.input_csv).stem}_run{args.run_tag}"
    csv_path = output_dir / f"{stem}.csv"
    log_path = output_dir / f"{stem}.txt"
    config_path = output_dir / f"{stem}.config.json"
    config_path.write_text(json.dumps(vars(args), indent=2, default=str) + "\n")
    log_path.write_text(
        f"generation:0,mean_score:{frame['score'].mean():.8g},"
        f"median_score:{frame['score'].median():.8g}\n"
    )

    use_segmasker = not args.disable_segmasker and shutil.which("segmasker") is not None
    if not args.disable_segmasker and not use_segmasker:
        print("[Warn] segmasker not found; low-complexity filtering is disabled")

    mating_pool = frame["Sequence"].tolist()
    population_size = len(frame)
    for generation in range(1, args.generations + 1):
        children = []
        observed = set()
        attempts = 0
        max_attempts = population_size * 1000
        while len(children) < population_size and attempts < max_attempts:
            attempts += 1
            parent1 = py_rng.choice(mating_pool)
            parent2 = py_rng.choice(mating_pool)
            child = mutate_crossover(
                parent1,
                parent2,
                mutation_rate=args.mutation_rate,
                rng=py_rng,
            )
            if child in observed:
                continue
            if use_segmasker and has_long_low_complexity_segment(
                child, args.segmasker_cutoff
            ):
                continue
            observed.add(child)
            children.append(child)
        if len(children) != population_size:
            raise RuntimeError(
                f"Could create only {len(children)} unique children after "
                f"{attempts} attempts"
            )

        child_scores = scores_for_sequences(
            model,
            tokenizer,
            children,
            device,
            args.batch_size,
            args.max_length,
        )
        for index, (child, child_score) in enumerate(zip(children, child_scores)):
            current_score = max(float(frame.at[index, "score"]), 1e-12)
            ratio = float(child_score) / current_score
            acceptance_probability = min(1.0, ratio * args.acceptance_scale)
            if ratio >= 1.0 or py_rng.random() < acceptance_probability:
                frame.at[index, "Sequence"] = child
                frame.at[index, "score"] = child_score

        keep = max(2, int(np.ceil(population_size * args.mating_pool_fraction)))
        keep = min(population_size, keep)
        mating_pool = (
            frame.sort_values("score", ascending=False)["Sequence"].head(keep).tolist()
        )
        with log_path.open("a") as handle:
            handle.write(
                f"generation:{generation},mean_score:{frame['score'].mean():.8g},"
                f"median_score:{frame['score'].median():.8g}\n"
            )
        frame[["Sequence", "score"]].to_csv(csv_path, index=False)

    print(f"Saved final population to: {csv_path}")
    print(f"Saved generation log to: {log_path}")
    print(f"Saved run configuration to: {config_path}")


if __name__ == "__main__":
    main()
