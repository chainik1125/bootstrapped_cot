# sentence_transition_probe.py
# A minimal, modular smoke test for signal in sentence/step transitions.
# - Loads HF + TransformerLens models
# - Generates short CoTs with "Step k:" boundaries (PRM-style) or sentence splits
# - Extracts residual states (late layer) via run_with_cache
# - Builds matched within-step vs cross-step pairs (k=1..3 ahead)
# - Fits a single ridge map and reports ΔLoss = Loss(cross) - Loss(within)
#
# Refs: TL run_with_cache / ActivationCache; PRM newline step format; Future Lens token-ahead predictability.
# Docs: TL (https://transformerlensorg.github.io/TransformerLens/), ActivationCache & run_with_cache,
# PRM "Let's Verify Step by Step" (newline-delimited steps), Future Lens (1–3 tokens ahead from single state).
# (citations in the chat body)

import os, re, math, random, argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

# ----------------------------
# 1) LOADING: model & tokenizer
# ----------------------------

def load_transformer(cfg):
    """
    Load:
      - TransformerLens HookedTransformer for activation access
      - HF AutoModelForCausalLM for generation
      - HF tokenizer
    """
    from transformer_lens import HookedTransformer
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_name = cfg["model"]["name"]
    device_cfg = cfg["model"]["device"]
    dtype_str = cfg["model"]["dtype"]

    device = "cuda" if device_cfg == "auto" and torch.cuda.is_available() else device_cfg
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(dtype_str, torch.float32)

    print(f"[load_transformer] Loading TL model: {model_name} (device={device})")
    tl_model = HookedTransformer.from_pretrained(model_name, device=device)

    print("[load_transformer] Loading HF model & tokenizer for generation")
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype if device != "cpu" else None,
        device_map="auto" if device == "cuda" else None,
    )
    return tl_model, hf_model, tok


# ----------------------------
# 2) LOADING: dataset / questions
# ----------------------------

BUILTIN_QUESTIONS = [
    "If a rectangle has length 8 and width 5, what is its area?",
    "A store sells pencils at 3 for $2. How much do 12 pencils cost?",
    "Sarah ran 5 km each day for 4 days, then 8 km on the fifth day. What is her total distance?",
    "A tank holds 60 liters. It loses 15 liters. What percent remains?",
    "The sum of two numbers is 21 and their difference is 5. What are the numbers?",
    "Simplify: (3/4) + (5/8).",
    "If 6 workers finish a job in 10 days, how many days for 4 workers (same rate)?",
    "A number increased by 25% is 45. What was the number?",
    "Solve for x: 2x + 7 = 31.",
    "What is the probability of rolling an even number on a fair six-sided die?",
    "A car travels 120 km in 2 hours. What is its average speed?",
    "The average of 4, 6, 10, and x is 8. Find x.",
    "Compute 18% of 250.",
    "If y = 2x + 3 and x = 7, find y.",
    "Find the LCM of 6 and 14.",
    "If a price is discounted by 20% from $50, what is the new price?",
    "Simplify: 5^2 - 3^2.",
    "What is the perimeter of a square with side length 9?",
    "Solve: 3(x - 2) = 15.",
    "Convert 2 hours and 30 minutes to minutes.",
]

def load_questions(cfg) -> List[str]:
    source = cfg["data"]["question_source"]
    n = cfg["experiment"]["num_questions"]
    if source == "builtin":
        return BUILTIN_QUESTIONS[:n]
    elif source == "file":
        path = cfg["data"]["questions_file"]
        assert path and os.path.exists(path), f"questions_file not found: {path}"
        with open(path, "r") as f:
            qs = [line.strip() for line in f if line.strip()]
        return qs[:n]
    else:
        raise ValueError(f"Unknown question_source: {source}")


# ----------------------------
# 3) PREVIEW: generation & segmentation
# ----------------------------

def generate_cot(hf_model, tok, question: str, cfg) -> str:
    tpl = cfg["prompt"]["template"]
    prompt = tpl.format(q=question)
    gen_cfg = cfg["generation"]
    max_new = cfg["experiment"]["max_new_tokens"]

    inputs = tok(prompt, return_tensors="pt").to(hf_model.device)
    with torch.no_grad():
        out = hf_model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=gen_cfg["do_sample"],
            temperature=gen_cfg["temperature"],
            top_p=gen_cfg["top_p"],
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
    return tok.decode(out[0], skip_special_tokens=True)

STEP_HDR_RE = re.compile(r"(?m)^(?:Step\s*\d+\s*:)\s*")

def split_steps(text: str, mode: str = "numbered") -> List[Tuple[int,int,int,int]]:
    """
    Return list of (step_start_char, header_end_char, content_start_char, step_end_char).
    - "numbered": splits on 'Step k:' headers, ending at next header or 'Answer:' or text end.
    - "sentence": naive sentence split (period/question/exclamation).
    """
    spans = []
    if mode == "numbered":
        ms = list(STEP_HDR_RE.finditer(text))
        if ms:
            for i, m in enumerate(ms):
                start = m.start()
                hdr_end = m.end()
                if i+1 < len(ms):
                    end = ms[i+1].start()
                else:
                    ans = re.search(r"(?m)^Answer\s*:\s*", text[m.end():])
                    end = (m.end() + ans.start()) if ans else len(text)
                content_s = hdr_end
                while content_s < end and text[content_s].isspace():
                    content_s += 1
                if content_s < end:
                    spans.append((start, hdr_end, content_s, end))
        if not spans:  # fallback to sentence
            mode = "sentence"

    if mode == "sentence":
        sent_re = re.compile(r"([^.!?]+[.!?])", re.S)
        for m in sent_re.finditer(text):
            s, e = m.start(), m.end()
            spans.append((s, s, s, e))
        if not spans:
            spans = [(0,0,0,len(text))]

    return spans

def preview_examples(texts: List[str], cfg, max_show: int = 2):
    print("\n[preview_examples] Showing a couple of segmented traces:")
    mode = cfg["experiment"]["step_format"]
    for t in texts[:max_show]:
        print("="*80)
        print(t)
        spans = split_steps(t, mode)
        print("\n-- Segments --")
        for i,(s,h,c,e) in enumerate(spans[:8]):
            frag = t[c:e].strip().replace("\n"," ")
            print(f"[{i}] {frag[:120]}{'...' if len(frag)>120 else ''}")


# ----------------------------
# 4) PAIR BUILDING (states & windows)
# ----------------------------

@dataclass
class Pair:
    x: np.ndarray       # source pooled state
    y: np.ndarray       # target pooled state
    kind: str           # "cross" or "within"
    k: int              # lookahead window size
    layer: int

def chars_to_token_ranges(tok, text: str, spans):
    enc = tok(text, return_offsets_mapping=True, return_tensors="pt", add_special_tokens=False)
    offsets = enc["offset_mapping"][0].tolist()
    ranges = []
    for (s,h,c,e) in spans:
        # first token overlapping content start
        t_start = next((i for i,(a,b) in enumerate(offsets) if b>c), None)
        # last token with start < end
        t_end = next((len(offsets)-1 - i for i,(a,b) in enumerate(reversed(offsets)) if a<e), None)
        if t_start is None or t_end is None or t_start>t_end:
            continue
        ranges.append((t_start, t_end))
    return enc["input_ids"][0], ranges

def select_layer_index(tl_model, layer_spec) -> int:
    L = tl_model.cfg.n_layers
    if isinstance(layer_spec, int):
        return layer_spec if layer_spec >= 0 else (L + layer_spec)
    if isinstance(layer_spec, str) and layer_spec.startswith("late-"):
        n = int(layer_spec.split("-")[1])
        return max(0, L - n)
    if layer_spec == "last":
        return L - 1
    return max(0, L - 2)

def pool_last(H: torch.Tensor, idx: int, n: int) -> torch.Tensor:
    start = max(0, idx - (n-1))
    return H[start:idx+1].mean(dim=0)

def pool_first(H: torch.Tensor, idx: int, k: int) -> torch.Tensor:
    end = min(H.shape[0], idx + k)
    return H[idx:end].mean(dim=0)

def build_pairs_from_text(tl_model, tok, text: str, cfg, layer_idx: int) -> List[Pair]:
    from transformer_lens import HookedTransformer

    mode = cfg["experiment"]["step_format"]
    k_max = cfg["experiment"]["k_max"]
    last_n = cfg["experiment"]["last_n_src_tok"]
    first_n_override = cfg["experiment"]["first_n_tgt_tok"]

    spans = split_steps(text, mode)
    if len(spans) < 1:
        return []

    input_ids, step_ranges = chars_to_token_ranges(tok, text, spans)
    if len(step_ranges) < 1:
        return []

    tokens = input_ids.unsqueeze(0).to(tl_model.cfg.device)
    with torch.no_grad():
        logits, cache = tl_model.run_with_cache(tokens, remove_batch_dim=False)

    H = cache["resid_post", layer_idx][0].detach().cpu()  # [seq, d_model]
    pairs: List[Pair] = []

    # cross-step pairs
    for i in range(len(step_ranges) - 1):
        s_start, s_end = step_ranges[i]
        t_start, t_end = step_ranges[i+1]
        p = s_end  # last token of current step content
        for k in range(1, k_max+1):
            tgt_k = first_n_override or k
            x = pool_last(H, p, last_n).numpy()
            y = pool_first(H, t_start, tgt_k).numpy()
            pairs.append(Pair(x=x, y=y, kind="cross", k=k, layer=layer_idx))

    # within-step pairs (distance-matched)
    for (s_start, s_end) in step_ranges:
        for k in range(1, k_max+1):
            # choose a few interior positions p so that p+k <= s_end
            candidates = list(range(s_start + last_n, s_end - k + 1))
            if not candidates:
                continue
            for p in random.sample(candidates, k=min(3, len(candidates))):
                tgt_k = first_n_override or k
                x = pool_last(H, p, last_n).numpy()
                y = pool_first(H, p+1, tgt_k).numpy()
                pairs.append(Pair(x=x, y=y, kind="within", k=k, layer=layer_idx))

    return pairs


# ----------------------------
# 5) FIT & EVALUATE (ridge)
# ----------------------------

def fit_ridge_and_eval(pairs: List[Pair], alpha: float = 1.0, seed: int = 42, run_shuffle=True, tl_model=None) -> Dict:
    X = np.stack([p.x for p in pairs], axis=0)
    Y = np.stack([p.y for p in pairs], axis=0)
    kinds = np.array([p.kind for p in pairs])
    ks = np.array([p.k for p in pairs])

    X_tr, X_te, Y_tr, Y_te, kinds_te, ks_te = train_test_split(X, Y, kinds, ks, test_size=0.3, random_state=seed)
    reg = Ridge(alpha=alpha, fit_intercept=False, random_state=seed).fit(X_tr, Y_tr)
    Y_hat = reg.predict(X_te)

    def group(mask):
        idx = np.where(mask)[0]
        if len(idx)==0:
            return {"n":0, "mse": float("nan"), "cos": float("nan")}
        mse = float(np.mean(np.sum((Y_te[idx]-Y_hat[idx])**2, axis=1)))
        cos = float(np.mean(np.sum(Y_te[idx]*Y_hat[idx], axis=1) /
                            (np.linalg.norm(Y_te[idx],axis=1)*np.linalg.norm(Y_hat[idx],axis=1) + 1e-8)))
        return {"n": int(len(idx)), "mse": mse, "cos": cos}

    results = {}
    for k_val in sorted(set(ks_te.tolist())):
        m_within = group((kinds_te=="within") & (ks_te==k_val))
        m_cross  = group((kinds_te=="cross")  & (ks_te==k_val))
        results[k_val] = {
            "within": m_within,
            "cross": m_cross,
            "delta_mse": (m_cross["mse"] - m_within["mse"]) if not (math.isnan(m_cross["mse"]) or math.isnan(m_within["mse"])) else float("nan"),
            "delta_cos": (m_cross["cos"] - m_within["cos"]) if not (math.isnan(m_cross["cos"]) or math.isnan(m_within["cos"])) else float("nan"),
        }

    if run_shuffle:
        cross_mask = (kinds_te=="cross")
        if cross_mask.sum()>1:
            Y_cross = Y_te[cross_mask]
            Y_cross_hat = Y_hat[cross_mask]
            perm = np.random.permutation(Y_cross.shape[0])
            results["shuffle_cross_mse"] = float(np.mean(np.sum((Y_cross[perm]-Y_cross_hat)**2, axis=1)))

    # Optional token-space CE (logit-lens style)
    if tl_model is not None:
        results["token_space_ce"] = token_space_ce(tl_model, Y_te, Y_hat)

    results["coef_norm"] = float(np.linalg.norm(reg.coef_))
    results["alpha"] = alpha
    return results, reg, (X_te, Y_te, Y_hat, kinds_te, ks_te)

def token_space_ce(tl_model, Y_true: np.ndarray, Y_hat: np.ndarray) -> float:
    """
    Approximate token-space comparison via final layer-norm + unembed (Logit Lens-style).
    """
    with torch.no_grad():
        h_true = torch.from_numpy(Y_true).to(tl_model.cfg.device)
        h_pred = torch.from_numpy(Y_hat).to(tl_model.cfg.device)
        z_true = tl_model.ln_final(h_true)
        z_pred = tl_model.ln_final(h_pred)
        logits_true = z_true @ tl_model.W_U + tl_model.b_U
        logits_pred = z_pred @ tl_model.W_U + tl_model.b_U
        logp_true = F.log_softmax(logits_true, dim=-1)
        logp_pred = F.log_softmax(logits_pred, dim=-1)
        # H(P_true, P_pred) averaged
        ce = float((-logp_pred.exp() * logp_true).sum(dim=-1).mean().item())
        return ce


# ----------------------------
# 6) INSPECT: a single datapoint behavior
# ----------------------------

def inspect_single_point(text: str, tok, tl_model, layer_idx: int, cfg, reg: Ridge):
    """
    Show how the probe behaves on one boundary:
      - print the step end, the next step start
      - compute cos/MSE for k=1..3 on this single example
    """
    spans = split_steps(text, cfg["experiment"]["step_format"])
    if len(spans) < 2:
        print("[inspect_single_point] Need at least 2 steps.")
        return
    input_ids, ranges = chars_to_token_ranges(tok, text, spans)
    tokens = input_ids.unsqueeze(0).to(tl_model.cfg.device)
    with torch.no_grad():
        _, cache = tl_model.run_with_cache(tokens, remove_batch_dim=False)
    H = cache["resid_post", layer_idx][0].detach().cpu()

    (s_start, s_end), (t_start, t_end) = ranges[0], ranges[1]  # first boundary
    last_n = cfg["experiment"]["last_n_src_tok"]
    first_n_override = cfg["experiment"]["first_n_tgt_tok"]
    k_max = cfg["experiment"]["k_max"]

    print("\n[inspect_single_point] --- boundary between step 0 and 1 ---")
    print("Current step (tail):", tok.decode(input_ids[s_end-10:s_end+1]))
    print("Next step (head):   ", tok.decode(input_ids[t_start:t_start+10]))

    for k in range(1, k_max+1):
        tgt_k = first_n_override or k
        x = pool_last(H, s_end, last_n).unsqueeze(0).numpy()
        y = pool_first(H, t_start, tgt_k).unsqueeze(0).numpy()
        y_hat = reg.predict(x)
        mse = float(np.sum((y - y_hat)**2))
        cos = float(np.sum(y*y_hat) / (np.linalg.norm(y)*np.linalg.norm(y_hat) + 1e-8))
        print(f"k={k}: single-example mse={mse:.4f}  cos={cos:.4f}")


# ----------------------------
# 7) ORCHESTRATION (main)
# ----------------------------

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def main(cfg_path: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["experiment"]["random_seed"])
    tl_model, hf_model, tok = load_transformer(cfg)

    # choose layer
    layer_idx = select_layer_index(tl_model, cfg["experiment"]["layer_to_use"])
    print(f"[main] Using layer index {layer_idx} / {tl_model.cfg.n_layers-1}")

    # questions + quick generation
    questions = load_questions(cfg)
    texts = [generate_cot(hf_model, tok, q, cfg) for q in questions]

    # preview a couple of examples & the test design
    preview_examples(texts, cfg, max_show=2)

    # build all pairs
    all_pairs: List[Pair] = []
    for t in texts:
        all_pairs.extend(build_pairs_from_text(tl_model, tok, t, cfg, layer_idx))
    print(f"\n[main] Built {len(all_pairs)} pairs "
          f"(within={sum(p.kind=='within' for p in all_pairs)}, cross={sum(p.kind=='cross' for p in all_pairs)})")

    if len(all_pairs) < 50:
        print("[main] WARNING: very few pairs; consider increasing num_questions or max_new_tokens.")

    # fit & evaluate
    results, reg, (X_te, Y_te, Y_hat, kinds_te, ks_te) = fit_ridge_and_eval(
        all_pairs,
        alpha=1.0,
        seed=cfg["experiment"]["random_seed"],
        run_shuffle=cfg["experiment"]["run_shuffle_control"],
        tl_model=tl_model if cfg["experiment"]["compute_token_ce"] else None
    )

    print("\n=== State-space metrics (Δ = cross - within) ===")
    for k in sorted(k for k in results.keys() if isinstance(k, int)):
        r = results[k]
        print(f"k={k}: within(n={r['within']['n']}) mse={r['within']['mse']:.4f} cos={r['within']['cos']:.4f}  |  "
              f"cross(n={r['cross']['n']}) mse={r['cross']['mse']:.4f} cos={r['cross']['cos']:.4f}  |  "
              f"Δmse={r['delta_mse']:.4f} Δcos={r['delta_cos']:.4f}")

    if "shuffle_cross_mse" in results:
        print(f"Shuffle control (cross-step mismatched) mse={results['shuffle_cross_mse']:.4f}")

    if "token_space_ce" in results:
        print(f"Token-space CE (logit-lens style, mixed test preds): {results['token_space_ce']:.4f}")

    # inspect a single boundary behavior
    inspect_single_point(texts[0], tok, tl_model, layer_idx, cfg, reg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
