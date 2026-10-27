"""
v5 one-to-one greedy matching (same geometry as generate_hns_bar_chart.py /
generate_pr_curve_doublestrand_one_to_one.py). Duplicated here so this workflow
does not modify the bar_chart package.
"""
from __future__ import annotations

import numpy as np

UPSTREAM_MIN = 6
UPSTREAM_MAX = 39


def _build_promoter_regions(ref_pos, ref_plus, tolerance: int):
    regions = {}
    for strand_val, mask in [("+", ref_plus), ("-", ~ref_plus)]:
        pos = ref_pos[mask]
        if strand_val == "+":
            rs = pos - UPSTREAM_MAX - tolerance
            re = pos - UPSTREAM_MIN + tolerance
        else:
            rs = pos + UPSTREAM_MIN - tolerance
            re = pos + UPSTREAM_MAX + tolerance
        order = np.argsort(rs)
        rs, re = rs[order], re[order]
        prefix_max_end = np.maximum.accumulate(re)
        regions[strand_val] = (rs, re, prefix_max_end)
    return regions


def _vectorized_is_tp(w_starts, w_ends, reg_starts, reg_ends, prefix_max_end):
    n = len(w_starts)
    if n == 0 or len(reg_starts) == 0:
        return np.zeros(n, dtype=np.bool_)
    idx = np.searchsorted(reg_starts, w_starts, side="right") - 1
    valid = idx >= 0
    result = np.zeros(n, dtype=np.bool_)
    result[valid] = prefix_max_end[idx[valid]] >= w_ends[valid]
    return result


def compute_geom_ok(starts, ends, is_plus, ref_pos, ref_plus, tolerance: int) -> np.ndarray:
    regions = _build_promoter_regions(ref_pos, ref_plus, tolerance)
    out = np.zeros(len(starts), dtype=np.bool_)
    pm = is_plus
    out[pm] = _vectorized_is_tp(starts[pm], ends[pm], *regions["+"])
    out[~pm] = _vectorized_is_tp(starts[~pm], ends[~pm], *regions["-"])
    return out


def greedy_o2o_topk_with_matched_ref(
    starts: np.ndarray,
    ends: np.ndarray,
    is_plus: np.ndarray,
    scores: np.ndarray,
    ref_pos: np.ndarray,
    ref_plus: np.ndarray,
    tolerance: int,
    geom_ok: np.ndarray,
    k: int,
):
    """
    First k predictions in global descending score order (mergesort).
    Returns tp_centers, fp_centers, fn_ref_positions, matched_ref (bool, len n_ref).
    """
    n_ref = len(ref_pos)
    n = len(starts)
    k = min(int(k), n)
    desc_order = np.argsort(-scores, kind="mergesort")

    plus_ref_idx = np.where(ref_plus)[0]
    minus_ref_idx = np.where(~ref_plus)[0]
    plus_pos = ref_pos[plus_ref_idx]
    minus_pos = ref_pos[minus_ref_idx]
    o_p = np.argsort(plus_pos)
    o_m = np.argsort(minus_pos)
    plus_pos_s = plus_pos[o_p]
    minus_pos_s = minus_pos[o_m]
    plus_idx_s = plus_ref_idx[o_p]
    minus_idx_s = minus_ref_idx[o_m]

    matched_ref = np.zeros(n_ref, dtype=np.bool_)
    tp_centers: list[int] = []
    fp_centers: list[int] = []

    for step in range(k):
        idx = int(desc_order[step])
        ws = int(starts[idx])
        we = int(ends[idx])
        center = (ws + we) // 2

        if not geom_ok[idx]:
            fp_centers.append(center)
            continue

        if is_plus[idx]:
            lo_t = we + UPSTREAM_MIN - tolerance
            hi_t = ws + UPSTREAM_MAX + tolerance
            if lo_t > hi_t:
                fp_centers.append(center)
                continue
            lo = int(np.searchsorted(plus_pos_s, lo_t, side="left"))
            hi = int(np.searchsorted(plus_pos_s, hi_t, side="right"))
            pos_s = plus_pos_s
            idx_s = plus_idx_s
        else:
            lo_t = we - UPSTREAM_MAX - tolerance
            hi_t = ws - UPSTREAM_MIN + tolerance
            if lo_t > hi_t:
                fp_centers.append(center)
                continue
            lo = int(np.searchsorted(minus_pos_s, lo_t, side="left"))
            hi = int(np.searchsorted(minus_pos_s, hi_t, side="right"))
            pos_s = minus_pos_s
            idx_s = minus_idx_s

        best_j = -1
        best_d = 10**18
        for j in range(lo, hi):
            ridx = int(idx_s[j])
            if matched_ref[ridx]:
                continue
            T = int(pos_s[j])
            if is_plus[idx]:
                reg_s = T - UPSTREAM_MAX - tolerance
                reg_e = T - UPSTREAM_MIN + tolerance
            else:
                reg_s = T + UPSTREAM_MIN - tolerance
                reg_e = T + UPSTREAM_MAX + tolerance
            if ws >= reg_s and we <= reg_e:
                d = abs(center - T)
                if d < best_d:
                    best_d = d
                    best_j = j

        if best_j >= 0:
            matched_ref[int(idx_s[best_j])] = True
            tp_centers.append(center)
        else:
            fp_centers.append(center)

    fn_positions = [int(ref_pos[i]) for i in range(n_ref) if not matched_ref[i]]
    return tp_centers, fp_centers, fn_positions, matched_ref
