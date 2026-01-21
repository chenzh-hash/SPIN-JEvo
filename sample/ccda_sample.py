# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"

import sys, subprocess, random, shutil
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, EsmForSequenceClassification, EsmModel
from peft import PeftModel, PeftConfig
from Genetic_algorithm_generation import mutate_crossover  # 你已有

# ================= 配置 =================
MODEL_NAME   = "/mnt/hdd/chenzh/esm2_t33_650M_UR50D"
ADAPTER_PATH = "/models/ccda_model/"
ESM_MAX_LEN  = 1022
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# —— 主开关
ENABLE_COS = False                # False: 完全回到“只用概率”的旧版
APPLY_ID_WITHOUT_COS = False      # True: 纯概率模式也加入 identity（默认按你要求为 False）

# —— 二分类：负类=0、正类=1（严格按你要求）
POS_INDEX = 1

# —— 权重与调度
LAMBDA = 1.0                      # 概率权重
NU     = 1.0                      # cos 权重上限（你要 1:10 可设 10）
PROB_WARMUP = 1e-4
PROB_TARGET = 0.10

# —— identity 目标与形状（新增）
ENABLE_ID = False
ID_TARGET   = 0.60
ID_SIGMA    = 0.30                # σ 放宽，远离 0.8 时也有非零奖励（避免无梯度）
ID_WEIGHT   = 2.0
ID_SHAPE    = "gauss"             # "gauss" 或 "quad"（二次型：-((id-tgt)/σ)^2）

# —— 初始化与注入（关键）
USE_TARGET_SEEDS   = False        # 初代从 TARGET_FASTAS 突变 20%（≈0.8 identity）
INJECT_EVERY       = 10           # 每 N 代注入一次
INJECT_FRAC        = 0.25         # 注入覆盖最差的比例
GUIDED_TARGET_ID   = 0.80         # 注入个体目标 identity
GUIDED_JITTER      = 0.05         # 注入时围绕 0.8 ± 抖动，避免同质化

# —— cos 质心数据
TARGET_FASTAS  = "ccdA_seed.fasta"

# —— 批大小 / 变异率 / 代数
PROB_BS  = 32
EMB_BS   = 16
MUT_RATE = 0.20
origin   = 'CcdA_seed.csv'
ec_pick  = 4
epoch    = 100

AA_STD = list("ACDEFGHIKLMNPQRSTVWY")

# ================= 工具 =================
def segmasker_available():
    return shutil.which("segmasker") is not None

_WARNED = {"segmasker": False}

def run_segmasker(input_fasta, output_interval):
    subprocess.run(f"segmasker -in {input_fasta} -out {output_interval}", shell=True, check=True)

def parse_interval_file(interval_file, cutoff=5):
    with open(interval_file,'r') as f:
        for line in f:
            if ' - ' in line:
                s,e = map(int, line.strip().split(' - '))
                if (e-s+1) > cutoff: return True
    return False

def check_disorder_in_sequence(sequence, sequence_id):
    if not segmasker_available():
        if not _WARNED["segmasker"]:
            print("[Warn] 未找到 segmasker，跳过无序区过滤。")
            _WARNED["segmasker"] = True
        return False
    fa  = f"temp_seqccda0_{sequence_id}.fasta"
    itv = f"temp_seqccda0_{sequence_id}.interval"
    with open(fa,'w') as f: f.write(f">seq_{sequence_id}\n{sequence}\n")
    run_segmasker(fa, itv)
    flag = parse_interval_file(itv)
    os.remove(fa); os.remove(itv)
    return flag

def read_fasta_simple(path):
    seqs, buf = [], []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('>'):
                if buf:
                    seqs.append(''.join(buf).strip()); buf=[]
            else:
                buf.append(line.strip())
    if buf: seqs.append(''.join(buf).strip())
    return seqs

def normalize_rows(M):
    M = np.asarray(M, dtype=np.float32)
    n = np.linalg.norm(M, axis=1, keepdims=True) + 1e-8
    return M / n

# ================= 模型 =================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 分类器 + LoRA（输出 logits -> softmax -> 概率）
zs_base_model = EsmForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
_ = PeftConfig.from_pretrained(ADAPTER_PATH)
ec_model = PeftModel.from_pretrained(zs_base_model, ADAPTER_PATH).to(DEVICE).eval()

# 用于 cos/medoid 的 backbone
need_backbone = ENABLE_COS or ENABLE_ID or USE_TARGET_SEEDS
esm_backbone = EsmModel.from_pretrained(MODEL_NAME).to(DEVICE).eval() if need_backbone else None

@torch.inference_mode()
def probs_for_sequences(seqs):
    out_all = []
    for i in range(0, len(seqs), PROB_BS):
        chunk = seqs[i:i+PROB_BS]
        toks = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True,
                         max_length=ESM_MAX_LEN).to(DEVICE)
        out  = ec_model(**toks)
        p    = torch.softmax(out.logits, dim=-1)[:, POS_INDEX]
        out_all.append(p.detach().cpu().numpy())
        del toks, out, p
        if DEVICE.type == "cuda": torch.cuda.empty_cache()
    return np.concatenate(out_all, axis=0) if out_all else np.zeros((0,), dtype=np.float32)

@torch.inference_mode()
def embed_batch(seqs):
    embs = []
    for i in range(0, len(seqs), EMB_BS):
        chunk = seqs[i:i+EMB_BS]
        toks = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True,
                         max_length=ESM_MAX_LEN).to(DEVICE)
        out  = esm_backbone(**toks, output_hidden_states=True)
        last = out.last_hidden_state
        attn = toks["attention_mask"].unsqueeze(-1)
        ids  = toks["input_ids"]
        valid = attn.bool()
        cls_id = getattr(tokenizer, "cls_token_id", None)
        sep_id = getattr(tokenizer, "sep_token_id", None)
        if cls_id is not None: valid = valid & (ids != cls_id).unsqueeze(-1)
        if sep_id is not None: valid = valid & (ids != sep_id).unsqueeze(-1)
        masked = last * valid
        lens = valid.sum(dim=1).clamp(min=1)
        pooled = masked.sum(dim=1) / lens
        embs.append(pooled.detach().cpu().numpy())
        del toks, out, last, attn, ids, valid, masked, lens, pooled
        if DEVICE.type == "cuda": torch.cuda.empty_cache()
    return np.concatenate(embs, axis=0) if embs else np.zeros((0,1), dtype=np.float32)

def compute_target_centroid_and_medoid(specs=TARGET_FASTAS, max_per=400):
    pool = []
    for path, w in specs:
        seqs = read_fasta_simple(path)
        if max_per and len(seqs) > max_per:
            seqs = random.sample(seqs, max_per)
        pool.extend(seqs)
    assert len(pool) > 0, "目标 FASTA 为空"

    E = embed_batch(pool)
    C = normalize_rows(E).mean(axis=0, keepdims=True)
    C = normalize_rows(C)[0]  # 单位化质心
    cos = normalize_rows(E) @ C.reshape(-1,1)
    j = int(np.argmax(cos))
    medoid_seq = pool[j]
    return C, medoid_seq

# 初始化 cos/medoid
if need_backbone:
    C_TARGET, MEDOID_SEQ = compute_target_centroid_and_medoid()
else:
    C_TARGET, MEDOID_SEQ = None, None

COS_CACHE = {}
def get_cos_for_seqs(seqs):
    need = [s for s in seqs if s not in COS_CACHE]
    if need:
        E = embed_batch(need)
        E = normalize_rows(E)
        c = C_TARGET.reshape(1,-1)
        cos = (E @ c.T).squeeze(-1)
        for s, cs in zip(need, cos):
            COS_CACHE[s] = float(cs)
    return np.array([COS_CACHE[s] for s in seqs], dtype=np.float32)

# ============== identity 相关 ==============
def hamming_identity(a: str, b: str) -> float:
    L = min(len(a), len(b))
    if L <= 0: return 0.0
    same = sum(1 for i in range(L) if a[i] == b[i])
    return same / float(L)

ID_CACHE = {}
def get_identity_for_seqs(seqs):
    need = [s for s in seqs if s not in ID_CACHE]
    if need:
        vals = [hamming_identity(s, MEDOID_SEQ) for s in need]
        for s, v in zip(need, vals):
            ID_CACHE[s] = float(v)
    return np.array([ID_CACHE[s] for s in seqs], dtype=np.float32)

def identity_reward(id_val, tgt=ID_TARGET, sigma=ID_SIGMA, shape=ID_SHAPE):
    if shape == "quad":
        # 二次型：范围宽，始终有“往 tgt 靠近”的动力
        return 1.0 - ((id_val - tgt) / (sigma + 1e-12))**2
    # 默认高斯（σ 放宽到 0.30）
    return float(np.exp(- ((id_val - tgt) ** 2) / (2.0 * sigma * sigma + 1e-12)))

def clamp01(x):
    return max(0.0, min(1.0, float(x)))

# —— 调度
def nu_schedule(mean_prob):
    if not ENABLE_COS:
        return 0.0
    if mean_prob < PROB_WARMUP:  return 0.0
    if mean_prob >= PROB_TARGET: return NU
    frac = (mean_prob - PROB_WARMUP) / max(1e-12, (PROB_TARGET - PROB_WARMUP))
    return NU * clamp01(frac)

def id_weight_schedule(mean_prob):
    # 身份项也可热身；如希望更晚介入，可把阈值设得更高
    if mean_prob < PROB_WARMUP:  return 0.0
    if mean_prob >= PROB_TARGET: return ID_WEIGHT
    frac = (mean_prob - PROB_WARMUP) / max(1e-12, (PROB_TARGET - PROB_WARMUP))
    return ID_WEIGHT * clamp01(frac)

# ============== 初始化/注入 ==============
def mutate_random_20pct(seq):
    s = list(seq)
    for i in range(len(s)):
        if random.random() < 0.20:
            aa = s[i]
            cand = random.choice([x for x in AA_STD if x != aa])
            s[i] = cand
    return ''.join(s)

def seed_from_targets(n):
    # 从 TARGET_FASTAS 采样，按 20% 随机突变（平均 identity ≈ 0.8）
    bank = []
    for path, w in TARGET_FASTAS:
        bank.extend(read_fasta_simple(path))
    if len(bank) == 0:
        raise RuntimeError("TARGET_FASTAS 为空，无法初始化种群。")
    out = []
    while len(out) < n:
        base = random.choice(bank)
        out.append(mutate_random_20pct(base))
    return out

def guided_to_identity(seq, target_seq, target_id=0.80):
    """把 seq 有指导地改向 target_seq，使 identity 接近 target_id"""
    s = list(seq)
    L = min(len(s), len(target_seq))
    # 当前匹配位点
    matches = [i for i in range(L) if s[i] == target_seq[i]]
    mism    = [i for i in range(L) if s[i] != target_seq[i]]
    cur_id  = len(matches) / max(1, L)
    need = int(round(target_id * L)) - len(matches)
    if need > 0:
        # 提升 identity：把若干不匹配位点替换为 target 的氨基酸
        k = min(need, len(mism))
        pick = random.sample(mism, k) if k > 0 else []
        for i in pick:
            s[i] = target_seq[i]
    elif need < 0:
        # 降低到指定 identity：从匹配位点里打乱几位
        k = min(-need, len(matches))
        pick = random.sample(matches, k) if k > 0 else []
        for i in pick:
            orig = s[i]
            s[i] = random.choice([x for x in AA_STD if x != orig])
    return ''.join(s)

def inject_guided(X, frac=INJECT_FRAC, target_id=GUIDED_TARGET_ID, jitter=GUIDED_JITTER):
    """将最差的 frac 个体替换为『medoid 引导到 target_id±jitter』的新个体"""
    n = len(X)
    k = max(1, int(frac * n))
    # 以当前排序（按 score 或 prob）找最差
    if 'score' in X:
        order = np.argsort(np.array(X['score'].to_list()))
    else:
        order = np.argsort(np.array(X['prob'].to_list()))
    worst_idx = order[:k]
    new_seqs = []
    for _ in worst_idx:
        tid = clamp01(np.random.normal(target_id, jitter))
        new_seqs.append(guided_to_identity(MEDOID_SEQ, MEDOID_SEQ, tid))  # 从 medoid 复制并调到目标 id
    # 替换，并补全字段
    for idx, new_s in zip(worst_idx, new_seqs):
        X.at[idx, 'Sequence'] = new_s
    # 重新计算 prob/cos/ident/score
    X['prob'] = probs_for_sequences(X['Sequence'].tolist())
    if ENABLE_COS:
        X['cos'] = get_cos_for_seqs(X['Sequence'].tolist())
    if ENABLE_ID and (ENABLE_COS or APPLY_ID_WITHOUT_COS):
        X['ident'] = get_identity_for_seqs(X['Sequence'].tolist())
    mean_prob = float(X['prob'].mean())
    X['score'] = None  # 占位

    return X

# ================= 数据：初代 =================
if USE_TARGET_SEEDS:
    # 用目标集合作初代（≈ 80% identity）
    df_seed = pd.read_csv(origin)
    n0 = df_seed[df_seed['label']==1].shape[0]
    init_seqs = seed_from_targets(n0)
    X = pd.DataFrame({'Sequence': init_seqs})
else:
    df = pd.read_csv(origin)
    X = df[df['label']==1].sample(frac=1).reset_index(drop=True)
    X['Sequence'] = X['Sequence'].str.slice(stop=1000)
    # 初代：每位 20% 概率突变（你的老做法）
    for idx in range(len(X)):
        seq = list(X.at[idx,'Sequence'])
        for pos in range(len(seq)):
            if np.random.rand() < MUT_RATE:
                choices = list(set(AA_STD) - {seq[pos]})
                seq[pos] = np.random.choice(list(choices))
        X.at[idx,'Sequence'] = ''.join(seq)

# 初代概率 / cos / identity / 评分
X['prob'] = probs_for_sequences(X['Sequence'].tolist())
if ENABLE_COS:
    X['cos'] = get_cos_for_seqs(X['Sequence'].tolist())
if ENABLE_ID and (ENABLE_COS or APPLY_ID_WITHOUT_COS):
    X['ident'] = get_identity_for_seqs(X['Sequence'].tolist())

def compute_score(prob_arr, cos_arr=None, id_arr=None, mean_prob_for_sched=None):
    # 纯概率模式
    if not ENABLE_COS and not (APPLY_ID_WITHOUT_COS and ENABLE_ID):
        return prob_arr.copy()

    lam = LAMBDA
    mean_p = mean_prob_for_sched if mean_prob_for_sched is not None else float(np.mean(prob_arr))
    nu_eff = nu_schedule(mean_p)
    id_eff = id_weight_schedule(mean_p)

    score = lam * prob_arr
    if ENABLE_COS and cos_arr is not None:
        score = score + nu_eff * cos_arr
    if ENABLE_ID and (ENABLE_COS or APPLY_ID_WITHOUT_COS) and id_arr is not None:
        id_reward = np.array([identity_reward(v) for v in id_arr], dtype=np.float32)
        # quad 形状需要截断到 [0,1]，避免负值强拉分
        if ID_SHAPE == "quad":
            id_reward = np.clip(id_reward, 0.0, 1.0)
        score = score + id_eff * id_reward
    return score

X['score'] = compute_score(
    prob_arr=X['prob'].values,
    cos_arr=(X['cos'].values if 'cos' in X else None),
    id_arr=(X['ident'].values if 'ident' in X else None),
    mean_prob_for_sched=float(X['prob'].mean())
)

print("str_generation:0")
if ENABLE_COS:
    mean_prob  = float(X['prob'].mean())
    line = f"gen:0, mean_prob:{mean_prob:.6e}"
    if 'cos' in X:   line += f", mean_cos:{float(X['cos'].mean()):.6f}"
    if 'ident' in X:
        mean_id = float(X['ident'].mean())
        mean_idr= float(np.mean([identity_reward(v) for v in X['ident'].values]))
        line += f", mean_ident:{mean_id:.4f}, mean_id_reward:{mean_idr:.4f}"
    line += f", mean_total:{float(X['score'].mean()):.6f}"
    print(line)
    print(f"[Sched] NU_eff={nu_schedule(mean_prob):.6f}  ID_eff={id_weight_schedule(mean_prob):.6f}  "
          f"(LAMBDA={LAMBDA:.3f}, ID_WEIGHT={ID_WEIGHT:.3f}, ID_TARGET={ID_TARGET:.2f}, SHAPE={ID_SHAPE})")
else:
    print(f"gen:0, mean_prob:{float(X['prob'].mean()):.6e}")

# ================= GA 主循环 =================
fittest_sequences = X['Sequence'].tolist()
m = sys.argv[1] if len(sys.argv)>1 else "0"

for i1 in range(1, epoch+1):
    print(f"generation:{i1}")

    # —— 周期性注入（解决 identity 长期过低） —— #
    if ENABLE_ID and (ENABLE_COS or APPLY_ID_WITHOUT_COS) and (i1 % INJECT_EVERY == 0):
        X = inject_guided(X, frac=INJECT_FRAC, target_id=GUIDED_TARGET_ID, jitter=GUIDED_JITTER)
        # 注入后立即刷新 score
        mean_p = float(X['prob'].mean())
        X['score'] = compute_score(
            prob_arr=X['prob'].values,
            cos_arr=(X['cos'].values if 'cos' in X else None),
            id_arr=(X['ident'].values if 'ident' in X else None),
            mean_prob_for_sched=mean_p
        )
        print(f"[Inject] guided medoid variants injected (frac={INJECT_FRAC:.2f}, target_id≈{GUIDED_TARGET_ID}±{GUIDED_JITTER})")

    # 产生子代
    new_population = []
    while len(new_population) < X.shape[0]:
        p1, p2 = random.choice(fittest_sequences), random.choice(fittest_sequences)
        child  = mutate_crossover(p1, p2)
        if not check_disorder_in_sequence(child, len(new_population)):
            # 轻微“向 medoid 拉一把”（只在 identity 过低时启用）
            if ENABLE_ID and (ENABLE_COS or APPLY_ID_WITHOUT_COS):
                id_now = hamming_identity(child, MEDOID_SEQ)
                if id_now < 0.5 and random.random() < 0.25:
                    target_id = clamp01(np.random.normal(GUIDED_TARGET_ID, GUIDED_JITTER))
                    child = guided_to_identity(child, MEDOID_SEQ, target_id=target_id)
            new_population.append(child)
            new_population = list(set(new_population))  # 去重

    X['child_seq']  = new_population
    X['child_prob'] = probs_for_sequences(X['child_seq'].tolist())

    # 需要的话计算 cos/ident
    if ENABLE_COS:
        X['child_cos']  = get_cos_for_seqs(X['child_seq'].tolist())
    if ENABLE_ID and (ENABLE_COS or APPLY_ID_WITHOUT_COS):
        X['child_ident'] = get_identity_for_seqs(X['child_seq'].tolist())

    # 本代有效权重
    mean_prob_for_sched = float(X['prob'].mean())
    X['child_score'] = compute_score(
        prob_arr=X['child_prob'].values,
        cos_arr=(X['child_cos'].values if 'child_cos' in X else None),
        id_arr=(X['child_ident'].values if 'child_ident' in X else None),
        mean_prob_for_sched=mean_prob_for_sched
    )

    # —— 接受/拒绝 —— #
    if ENABLE_COS or (APPLY_ID_WITHOUT_COS and ENABLE_ID):
        # 将父代 score 用同一调度重算，保持一致的比值
        X['score'] = compute_score(
            prob_arr=X['prob'].values,
            cos_arr=(X['cos'].values if 'cos' in X else None),
            id_arr=(X['ident'].values if 'ident' in X else None),
            mean_prob_for_sched=mean_prob_for_sched
        )
        for i in range(len(X)):
            cur_s = float(X.iloc[i]['score'])
            nxt_s = float(X.iloc[i]['child_score'])
            ratio = nxt_s / (cur_s if cur_s != 0 else 1e-12)
            if (ratio >= 1.0) or (random.random() < ratio * 0.125):
                X.at[i, 'Sequence']   = X.iloc[i]['child_seq']
                X.at[i, 'prob']       = X.iloc[i]['child_prob']
                if ENABLE_COS:
                    X.at[i, 'cos']    = X.iloc[i].get('child_cos', np.nan)
                if ENABLE_ID and (ENABLE_COS or APPLY_ID_WITHOUT_COS):
                    X.at[i, 'ident']  = X.iloc[i].get('child_ident', np.nan)
                X.at[i, 'score']      = X.iloc[i]['child_score']
    else:
        # —— 纯概率模式（与老版本一致）—— #
        for i in range(len(X)):
            cur_p = float(X.iloc[i]['prob'])
            nxt_p = float(X.iloc[i]['child_prob'])
            acceptance_ratio = (torch.tensor(nxt_p) / torch.tensor(cur_p)).item()
            if (acceptance_ratio >= 1.0) or (torch.rand(1).item() < acceptance_ratio * 0.125):
                X.at[i, 'Sequence'] = X.iloc[i]['child_seq']
                X.at[i, 'prob']     = X.iloc[i]['child_prob']
                X.at[i, 'score']    = X.iloc[i]['child_prob']  # 纯概率

    # 父本池更新
    if ENABLE_COS or (APPLY_ID_WITHOUT_COS and ENABLE_ID):
        idx_sorted = np.argsort(np.array(X['score'].to_list()))
    else:
        idx_sorted = np.argsort(np.array(X['prob'].to_list()))
    sorted_pop = [X['Sequence'].to_list()[i] for i in idx_sorted]
    fittest_sequences = sorted_pop[len(sorted_pop)//ec_pick:]

    # —— 打印 & 保存 —— #
    if ENABLE_COS or (APPLY_ID_WITHOUT_COS and ENABLE_ID):
        mean_prob  = float(X['prob'].mean())
        line = f"gen:{i1}, mean_prob:{mean_prob:.6e}"
        if 'cos' in X:   line += f", mean_cos:{float(X['cos'].mean()):.6f}"
        if 'ident' in X:
            mean_id = float(X['ident'].mean())
            mean_idr= float(np.mean([identity_reward(v) for v in X['ident'].values]))
            line += f", mean_ident:{mean_id:.4f}, mean_id_reward:{mean_idr:.4f}"
        line += f", mean_total:{float(X['score'].mean()):.6f}"
        print(line)
        print(f"[Sched] NU_eff={nu_schedule(mean_prob):.6f} (LAMBDA={LAMBDA:.3f})  "
              f"ID_eff={id_weight_schedule(mean_prob):.6f} (ID_WEIGHT={ID_WEIGHT:.3f}, ID_TARGET={ID_TARGET:.2f}, SHAPE={ID_SHAPE})")
        txt_path = origin.split('.')[0]+f'_{ec_pick}_ec{m}.txt'
        cols = ['Sequence','prob']
        if ENABLE_COS: cols.append('cos')
        if ENABLE_ID:  cols.append('ident')
        cols.append('score')
        csv_path = origin.split('.')[0]+f'_{ec_pick}_ec{m}.csv'
        with open(txt_path, 'a') as f:
            f.write(line+"\n")
        X[cols].to_csv(csv_path, index=False)
    else:
        mean_prob = float(X['prob'].mean())
        print(f"gen:{i1}, mean_prob:{mean_prob:.6e}")
        txt_path = origin.split('.')[0]+f'_probonly_{ec_pick}_ec{m}.txt'
        csv_path = origin.split('.')[0]+f'_probonly_{ec_pick}_ec{m}.csv'
        with open(txt_path, 'a') as f:
            f.write(f"gen:{i1}, mean_prob:{mean_prob:.6e}\n")
        X[['Sequence','prob','score']].to_csv(csv_path, index=False)



