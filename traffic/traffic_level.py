import pandas as pd
import numpy as np
import random

# ================= CONFIG =================
INPUT_CSV = "MAWI-WIDE_2023-2025.csv"
OUTPUT_CSV = "TL_MAWI-WIDE_2023-2025.csv"

# Modes: "random", "topk", "proportional", "firstk"
FLOW_SELECTION_MODE = "random"

# TL5 Metric Modes:
#   "concentrate"  - Keep original Total/T, concentrate into 1-2 flows (elephant flow)
#   "top_flows"    - Use top 1-2 existing flows as-is (preserves original characteristics)
#   "combine_top3" - Sum top-3 flows into 1-2 synthetic flows
TL5_METRIC_MODE = "concentrate"

# For reproducibility (optional)
random.seed(0)
np.random.seed(0)
# ==========================================

# Load data
df = pd.read_csv(INPUT_CSV)

# Column groups
flows_T_cols = [f"F{i}/T" for i in range(1, 11)]
flows_P_cols = [f"F{i}/P" for i in range(1, 11)]
flows_avg_cols = [f"F{i}/Avg_packet_size" for i in range(1, 11)]
flows_npk_cols = [f"F{i}/n_packets" for i in range(1, 11)]

# ---------- 1) traffic_level based on Total/T ----------


def traffic_level(total_t: float) -> int:
    if total_t <= 1000:
        return 1      # L1: low traffic
    elif total_t <= 1500:
        return 2      # L2: medium-low
    elif total_t <= 2000:
        return 3      # L3: medium-high
    elif total_t <= 3000:
        return 4      # L4: high traffic (8-10 flows)
    else:
        return 5      # L5: heavy concentrated (1-2 elephant flows)


df["traffic_level"] = df["Total/T"].apply(traffic_level)

# ---------- 2) mapping: traffic_level -> k candidates ----------
level_to_ks = {
    1: [1, 2],        # TL1: low traffic
    2: [3, 4, 5],     # TL2: medium-low
    3: [6, 7],        # TL3: medium-high
    4: [8, 9, 10],    # TL4: high traffic
    5: [1, 2],        # TL5: heavy concentrated (elephant flows)
}

# ---------- 3) helper: choose which flows to keep ----------


def choose_keep_mask(traffic: np.ndarray,
                     nonzero_idx: np.ndarray,
                     k_eff: int,
                     mode: str) -> np.ndarray:
    """
    Return a boolean mask of length 10 indicating which flows to keep.
    traffic: array of F*/T values (length 10)
    nonzero_idx: indices of flows with traffic > 0
    k_eff: number of flows to keep (<= len(nonzero_idx))
    mode: one of "random", "topk", "proportional", "firstk"
    """
    if len(nonzero_idx) == 0 or k_eff <= 0:
        return np.zeros(10, dtype=bool)

    if mode == "random":
        chosen_idx = np.random.choice(nonzero_idx, size=k_eff, replace=False)

    elif mode == "topk":
        # Keep k flows with the largest traffic among non-zero ones
        nz_traffic = traffic[nonzero_idx]
        top_local = np.argsort(-nz_traffic)[:k_eff]
        chosen_idx = nonzero_idx[top_local]

    elif mode == "proportional":
        # Probability proportional to traffic (among non-zero)
        nz_traffic = traffic[nonzero_idx]
        # In case of numerical weirdness, fall back to uniform
        if nz_traffic.sum() <= 0:
            chosen_idx = np.random.choice(
                nonzero_idx, size=k_eff, replace=False)
        else:
            probs = nz_traffic / nz_traffic.sum()
            chosen_idx = np.random.choice(
                nonzero_idx, size=k_eff, replace=False, p=probs)

    elif mode == "firstk":
        # Just take the first k non-zero flows in index order
        chosen_idx = nonzero_idx[:k_eff]

    else:
        raise ValueError(f"Unknown FLOW_SELECTION_MODE: {mode}")

    keep_mask = np.zeros(10, dtype=bool)
    keep_mask[chosen_idx] = True
    return keep_mask


# ---------- 4) TL5 special handling ----------

def adjust_row_tl5(row: pd.Series, k: int, mode: str) -> pd.Series:
    """
    Special adjustment for TL5 (heavy concentrated flows).

    Modes:
    - "concentrate": Keep original Total/T, distribute into k flows
    - "top_flows": Keep top k flows as-is
    - "combine_top3": Combine top 3 flows into k synthetic flows
    """
    traffic = np.array([row[c] for c in flows_T_cols], dtype=float)
    packets = np.array([row[c] for c in flows_P_cols], dtype=float)
    avg_sizes = np.array([row[c] for c in flows_avg_cols], dtype=float)
    n_packets = np.array([row[c] for c in flows_npk_cols], dtype=float)

    nonzero_idx = np.where(traffic > 0)[0]
    if len(nonzero_idx) == 0:
        return row

    # Sort by traffic descending
    sorted_idx = nonzero_idx[np.argsort(-traffic[nonzero_idx])]

    # Zero out all flows first
    for i in range(10):
        row[flows_T_cols[i]] = 0.0
        row[flows_P_cols[i]] = 0.0
        row[flows_avg_cols[i]] = 0.0
        row[flows_npk_cols[i]] = 0.0

    if mode == "concentrate":
        # Distribute original Total/T and Total/P into k flows
        original_total_t = traffic.sum()
        original_total_p = packets.sum()
        original_total_npk = n_packets.sum()

        # Weighted average of packet sizes (by n_packets)
        if original_total_npk > 0:
            weighted_avg_size = np.sum(avg_sizes * n_packets) / original_total_npk
        else:
            weighted_avg_size = avg_sizes[sorted_idx[0]] if len(sorted_idx) > 0 else 0.0

        # Distribute evenly across k flows
        for i in range(k):
            row[flows_T_cols[i]] = original_total_t / k
            row[flows_P_cols[i]] = original_total_p / k
            row[flows_avg_cols[i]] = weighted_avg_size
            row[flows_npk_cols[i]] = original_total_npk / k

    elif mode == "top_flows":
        # Keep top k flows as-is
        k_eff = min(k, len(sorted_idx))
        for i, orig_idx in enumerate(sorted_idx[:k_eff]):
            row[flows_T_cols[i]] = traffic[orig_idx]
            row[flows_P_cols[i]] = packets[orig_idx]
            row[flows_avg_cols[i]] = avg_sizes[orig_idx]
            row[flows_npk_cols[i]] = n_packets[orig_idx]

    elif mode == "combine_top3":
        # Combine top 3 flows into k synthetic flows
        top3_idx = sorted_idx[:min(3, len(sorted_idx))]
        combined_t = traffic[top3_idx].sum()
        combined_p = packets[top3_idx].sum()
        combined_npk = n_packets[top3_idx].sum()

        if combined_npk > 0:
            combined_avg_size = np.sum(avg_sizes[top3_idx] * n_packets[top3_idx]) / combined_npk
        else:
            combined_avg_size = avg_sizes[top3_idx].mean()

        # Distribute combined metrics into k flows
        for i in range(k):
            row[flows_T_cols[i]] = combined_t / k
            row[flows_P_cols[i]] = combined_p / k
            row[flows_avg_cols[i]] = combined_avg_size
            row[flows_npk_cols[i]] = combined_npk / k

    else:
        raise ValueError(f"Unknown TL5_METRIC_MODE: {mode}")

    # Recompute totals
    row["Total/T"] = float(sum(row[c] for c in flows_T_cols))
    row["Total/P"] = float(sum(row[c] for c in flows_P_cols))

    return row


# ---------- 5) row-wise adjustment ----------


def adjust_row(row: pd.Series) -> pd.Series:
    level = int(row["traffic_level"])
    ks = level_to_ks[level]
    k = random.choice(ks)

    # TL5 gets special handling
    if level == 5:
        return adjust_row_tl5(row, k, TL5_METRIC_MODE)

    # current per-flow traffic
    traffic = np.array([row[c] for c in flows_T_cols], dtype=float)

    # indices of flows that actually have traffic
    nonzero_idx = np.where(traffic > 0)[0]

    # if no traffic at all, nothing to change
    if len(nonzero_idx) == 0:
        return row

    # effective k cannot exceed number of non-zero flows
    k_eff = min(k, len(nonzero_idx))

    # choose which flows to keep according to the selected mode
    keep_mask = choose_keep_mask(
        traffic, nonzero_idx, k_eff, FLOW_SELECTION_MODE)

    # Zero out numeric fields for dropped flows
    for i in range(10):
        if not keep_mask[i]:
            row[flows_T_cols[i]] = 0.0
            row[flows_P_cols[i]] = 0.0
            row[flows_avg_cols[i]] = 0.0
            row[flows_npk_cols[i]] = 0.0
            # q_type and *_ips are left as-is; traffic is zero so they won't matter

    # Recompute totals from the remaining flows
    row["Total/T"] = float(sum(row[c] for c in flows_T_cols))
    row["Total/P"] = float(sum(row[c] for c in flows_P_cols))

    return row


# ---------- 5) apply to all rows ----------
df_transformed = df.apply(adjust_row, axis=1)

# ---------- 6) save ----------
df_transformed.to_csv(OUTPUT_CSV, index=False)
print(f"Saved: {OUTPUT_CSV}")
print(f"Flow selection mode: {FLOW_SELECTION_MODE}")
print(f"TL5 metric mode: {TL5_METRIC_MODE}")
