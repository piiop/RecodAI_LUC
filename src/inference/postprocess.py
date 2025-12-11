import json
import numpy as np

def rle_encode(mask: np.ndarray) -> str:
    """
    Convert a 2D binary mask into competition-format RLE.
    Returns a JSON string like "[start,length,...]".
    """
    mask = mask.astype(bool)
    flat = mask.T.flatten()
    dots = np.where(flat)[0]

    if len(dots) == 0:
        return json.dumps([])

    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend([b + 1, 0])  # 1-based start
        run_lengths[-1] += 1
        prev = b

    return json.dumps([int(x) for x in run_lengths])
