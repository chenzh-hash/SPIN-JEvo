import argparse
import os
import random
import shutil
import subprocess

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoTokenizer, EsmForSequenceClassification

from ga_utils import mutate_crossover

os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"

MODEL_POS_INDEX = 1
ESM_MAX_LEN = 1022
PROB_BS = 32
MUT_RATE = 0.20
EC_PICK = 4
EPOCHS = 100
AA_STD = list("ACDEFGHIKLMNPQRSTVWY")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run probability-only GA sampling with segmasker filtering.")
    parser.add_argument("--model-name", required=True, help="Base ESM-2 model name or local path.")
    parser.add_argument("--adapter-path", required=True, help="Path to the LoRA adapter directory.")
    parser.add_argument("--input-csv", required=True, help="Initial seed pool CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory for output CSV and TXT files.")
    parser.add_argument("--run-tag", default="1", help="Suffix tag used in the original output filenames.")
    return parser.parse_args()


def segmasker_available() -> bool:
    return shutil.which("segmasker") is not None


_warned_segmasker = False


def run_segmasker(input_fasta: str, output_interval: str) -> None:
    subprocess.run(f"segmasker -in {input_fasta} -out {output_interval}", shell=True, check=True)


def parse_interval_file(interval_file: str, cutoff: int = 5) -> bool:
    with open(interval_file, "r") as handle:
        for line in handle:
            if " - " in line:
                start, end = map(int, line.strip().split(" - "))
                if (end - start + 1) > cutoff:
                    return True
    return False


def check_disorder_in_sequence(sequence: str, sequence_id: int) -> bool:
    global _warned_segmasker
    if not segmasker_available():
        if not _warned_segmasker:
            print("[Warn] segmasker not found. Disorder filtering is skipped.")
            _warned_segmasker = True
        return False

    fasta_path = f"temp_seqcrtbstr_{sequence_id}.fasta"
    interval_path = f"temp_seqcrtbstr_{sequence_id}.interval"
    with open(fasta_path, "w") as handle:
        handle.write(f">seq_{sequence_id}\n{sequence}\n")
    run_segmasker(fasta_path, interval_path)
    flag = parse_interval_file(interval_path)
    os.remove(fasta_path)
    os.remove(interval_path)
    return flag


def probs_for_sequences(model, tokenizer, seqs, device):
    out_all = []
    with torch.inference_mode():
        for i in range(0, len(seqs), PROB_BS):
            chunk = seqs[i:i + PROB_BS]
            toks = tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=ESM_MAX_LEN,
            ).to(device)
            out = model(**toks)
            prob = torch.softmax(out.logits, dim=-1)[:, MODEL_POS_INDEX]
            out_all.append(prob.detach().cpu().numpy())
            del toks, out, prob
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return np.concatenate(out_all, axis=0) if out_all else np.zeros((0,), dtype=np.float32)


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    base_model = EsmForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PeftModel.from_pretrained(base_model, args.adapter_path).to(device).eval()

    origin_name = os.path.basename(args.input_csv)
    output_stem = os.path.splitext(origin_name)[0] + f"_prob_id20_tada10_{EC_PICK}_run{args.run_tag}"
    os.makedirs(args.output_dir, exist_ok=True)
    txt_path = os.path.join(args.output_dir, output_stem + ".txt")
    csv_path = os.path.join(args.output_dir, output_stem + ".csv")

    df = pd.read_csv(args.input_csv)
    X = df[df["label"] == 1].sample(frac=1).reset_index(drop=True)
    X["Sequence"] = X["Sequence"].astype(str).str.slice(stop=1000)

    for idx in range(len(X)):
        seq = list(X.at[idx, "Sequence"])
        for pos in range(len(seq)):
            if np.random.rand() < MUT_RATE:
                choices = list(set(AA_STD) - {seq[pos]})
                seq[pos] = np.random.choice(list(choices))
        X.at[idx, "Sequence"] = "".join(seq)

    X["prob"] = probs_for_sequences(model, tokenizer, X["Sequence"].tolist(), device)
    X["score"] = X["prob"]

    print("str_generation:0")
    print(f"gen:0, mean_prob:{float(X['prob'].mean()):.6e}")

    fittest_sequences = X["Sequence"].tolist()

    for generation in range(1, EPOCHS + 1):
        print(f"generation:{generation}")

        new_population = []
        while len(new_population) < X.shape[0]:
            p1 = random.choice(fittest_sequences)
            p2 = random.choice(fittest_sequences)
            child = mutate_crossover(p1, p2)
            if not check_disorder_in_sequence(child, len(new_population)):
                new_population.append(child)
                new_population = list(set(new_population))

        X["child_seq"] = new_population
        X["child_prob"] = probs_for_sequences(model, tokenizer, X["child_seq"].tolist(), device)
        X["child_score"] = X["child_prob"]

        for i in range(len(X)):
            cur_prob = float(X.iloc[i]["prob"])
            nxt_prob = float(X.iloc[i]["child_prob"])
            acceptance_ratio = (torch.tensor(nxt_prob) / torch.tensor(cur_prob)).item()
            if (acceptance_ratio >= 1.0) or (torch.rand(1).item() < acceptance_ratio * 0.125):
                X.at[i, "Sequence"] = X.iloc[i]["child_seq"]
                X.at[i, "prob"] = X.iloc[i]["child_prob"]
                X.at[i, "score"] = X.iloc[i]["child_prob"]

        idx_sorted = np.argsort(np.array(X["prob"].to_list()))
        sorted_pop = [X["Sequence"].to_list()[i] for i in idx_sorted]
        fittest_sequences = sorted_pop[len(sorted_pop) // EC_PICK:]

        mean_prob = float(X["prob"].mean())
        print(f"gen:{generation}, mean_prob:{mean_prob:.6e}")
        with open(txt_path, "a") as handle:
            handle.write(f"gen:{generation}, mean_prob:{mean_prob:.6e}\n")
        X[["Sequence", "prob", "score"]].to_csv(csv_path, index=False)

    print(f"Saved final population to: {csv_path}")
    print(f"Saved generation log to: {txt_path}")


if __name__ == "__main__":
    main()
