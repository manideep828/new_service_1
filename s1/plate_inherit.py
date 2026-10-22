MIN_FIRST_IOU = 0.15
MAX_EDITS = 2


def edit_distance(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            prev, dp[j] = dp[j], (prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1]))
    return dp[n]


def find_plate_donor(recipient, cand_text, vehicles, by_track, iou_fn,
                     min_iou=MIN_FIRST_IOU, max_edits=MAX_EDITS):
    """Return the single donor vehicle whose confirmed plate may be inherited, else None."""
    if not cand_text or getattr(recipient, "first_bbox", None) is None:
        return None
    hits = []
    for d in vehicles:
        if d is recipient:
            continue
        if not getattr(d, "plate_final", False) or d.slot_id is not None:
            continue
        if by_track.get(d.track_id) is d:
            continue
        if d.last_bbox is None:
            continue
        if float(d.first_ts) >= float(recipient.first_ts):
            continue
        if float(d.last_seen_ts) < float(recipient.first_ts):
            continue
        dc = d.voter.consensus()
        if dc is None:
            continue
        if edit_distance(cand_text, dc.text) > max_edits:
            continue
        if iou_fn(recipient.first_bbox, d.last_bbox) < min_iou:
            continue
        hits.append(d)
    return hits[0] if len(hits) == 1 else None
