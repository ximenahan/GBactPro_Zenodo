"""
PAS Dis: prefer the official PAS Dis column (tss_m10_range=[0,14]).

Fallback rebuilds Dis with Promoter_Architecture_Scanner getDis() when Dis is absent.
"""

import math

MAX_M10_TSS_DIST = 50


def get_dis(m10_end, tss_loc, minus10, down) -> str:
    dist = abs(int(m10_end) - int(tss_loc))
    down = str(down)
    if dist > MAX_M10_TSS_DIST:
        return ""
    if dist == 1:
        return "*"
    end = dist - 2
    if end <= 0:
        return ""
    return down[:end]


def dis_for_sequence(row) -> str:
    """Dis bases to append after Minus35+Spacer+Minus10."""
    if "Dis" in getattr(row, "index", []):
        raw = row["Dis"]
        if raw is None:
            return ""
        try:
            if isinstance(raw, float) and math.isnan(raw):
                return ""
        except (TypeError, ValueError):
            pass
        dis = str(raw)
        if dis in ("", "*", "nan", "None"):
            return ""
        return dis
    rebuilt = get_dis(row["loc_m10_end"], row["loc"], row["Minus10"], row["DOWN"])
    return "" if rebuilt == "*" else rebuilt
