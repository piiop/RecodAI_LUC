# src/utils/cls_collapse_logger.py
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional


def _jsonify(v: Any) -> Any:
    """Convert torch / numpy scalars + misc to JSON-friendly."""
    try:
        import torch
        if isinstance(v, torch.Tensor):
            if v.numel() == 1:
                return v.detach().cpu().item()
            return v.detach().cpu().tolist()
    except Exception:
        pass

    try:
        import numpy as np
        if isinstance(v, (np.generic,)):
            return v.item()
    except Exception:
        pass

    if isinstance(v, (Path,)):
        return str(v)

    return v


class ClsCollapseLogger:
    """
    Logs:
      - run meta (json)
      - optimizer debug (json)
      - per-step losses (csv)
      - per-epoch summaries (csv)
      - one-time debug dumps (jsonl): target mask stats, logits/probs stats
    """
    def __init__(self, out_dir: str, run_name: str):
        self.out_dir = Path(out_dir) / run_name
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.paths = {
            "meta": self.out_dir / "meta.json",
            "optimizer": self.out_dir / "optimizer.json",
            "step_losses": self.out_dir / "step_losses.csv",
            "epoch_summary": self.out_dir / "epoch_summary.csv",
            "debug_jsonl": self.out_dir / "debug.jsonl",
        }

        self._step_writer: Optional[csv.DictWriter] = None
        self._epoch_writer: Optional[csv.DictWriter] = None

    # ---------------------
    # JSON / JSONL
    # ---------------------
    def write_meta(self, meta: Dict[str, Any]) -> None:
        meta = {k: _jsonify(v) for k, v in meta.items()}
        self.paths["meta"].write_text(json.dumps(meta, indent=2))

    def write_optimizer_debug(self, info: Dict[str, Any]) -> None:
        info = {k: _jsonify(v) for k, v in info.items()}
        self.paths["optimizer"].write_text(json.dumps(info, indent=2))

    def debug_event(self, tag: str, payload: Dict[str, Any]) -> None:
        rec = {"tag": tag, **payload}
        rec = {k: _jsonify(v) for k, v in rec.items()}
        with self.paths["debug_jsonl"].open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    # ---------------------
    # CSV writers
    # ---------------------
    def log_step_losses(self, row: Dict[str, Any]) -> None:
        row = {k: _jsonify(v) for k, v in row.items()}
        write_header = not self.paths["step_losses"].exists()
        with self.paths["step_losses"].open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)

    def log_epoch_summary(self, row: Dict[str, Any]) -> None:
        row = {k: _jsonify(v) for k, v in row.items()}
        write_header = not self.paths["epoch_summary"].exists()
        with self.paths["epoch_summary"].open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)
