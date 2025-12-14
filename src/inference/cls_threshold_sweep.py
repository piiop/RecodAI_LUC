# inference/cls_threshold_sweep.py

import torch
from pathlib import Path
from torch.utils.data import DataLoader

from src.data.dataloader import (
    ForgeryDataset,
    get_val_transform,
    detection_collate_fn,
)
from src.models.mask2former_v1 import Mask2FormerForgeryModel
from src.inference.infer_multi_configs import run_threshold_sweep
from src.utils.config_utils import load_yaml, sanitize_model_kwargs


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # config
    # ------------------------------------------------------------------
    CFG_PATH = "base.yaml"
    cfg = load_yaml(CFG_PATH)

    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    trainer_cfg = cfg.get("trainer", {})

    # ------------------------------------------------------------------
    # paths
    # ------------------------------------------------------------------
    weights_path = trainer_cfg.get(
        "save_path",
        "weights/full_train/model_full_data_baseline.pth",
    )

    out_dir = Path("experiments/cls_threshold_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # dataset / loader
    # ------------------------------------------------------------------
    dataset = ForgeryDataset(
        transform=get_val_transform(img_size=data_cfg.get("img_size", 256)),
        is_train=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=trainer_cfg.get("batch_size", 4),
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=detection_collate_fn,
        persistent_workers=True,
    )

    # ------------------------------------------------------------------
    # model (config-driven)
    # ------------------------------------------------------------------
    mk = sanitize_model_kwargs(model_cfg)
    model = Mask2FormerForgeryModel(
        **mk,
    ).to(device)

    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # ------------------------------------------------------------------
    # sweep definition
    # ------------------------------------------------------------------
    cls_thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    run_threshold_sweep(
        model=model,
        loader=loader,
        device=device,
        gates=[-1.0],
        cls_thresholds=cls_thresholds,
        mask_thresholds=[model_cfg.get("default_mask_threshold", 0.5)],
        out_csv_path=out_dir / "cls_threshold_sweep.csv",
    )

    print("Saved:")
    print(out_dir / "cls_threshold_sweep.csv")
    print(out_dir / "cls_threshold_sweep_agg.csv")


if __name__ == "__main__":
    main()
