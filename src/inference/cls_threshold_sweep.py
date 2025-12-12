# inference/cls_threshold_sweep.py

import torch
from pathlib import Path
from torch.utils.data import DataLoader
import os
from src.data.dataloader import (
    ForgeryDataset,
    get_val_transform,
    detection_collate_fn,
)
from src.models.mask2former_v1 import Mask2FormerForgeryModel
from src.inference.infer_multi_configs import run_threshold_sweep


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------
    # paths
    # --------------------
    WEIGHTS_PATH = "weights/full_train/model_full_data_baseline.pth"
    OUT_DIR = Path("experiments/cls_threshold_sweep")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ROOT_DIR = os.environ.get("DATA_ROOT", "data")
    # --------------------
    # dataset / loader (correct ForgeryDataset usage)
    # --------------------
    authentic_path = os.path.join(ROOT_DIR, "train_images", "authentic")
    forged_path = os.path.join(ROOT_DIR, "train_images", "forged")
    masks_path = os.path.join(ROOT_DIR, "train_masks")
    supp_forged_path = os.path.join(ROOT_DIR, "supplemental_images")
    supp_masks_path = os.path.join(ROOT_DIR, "supplemental_masks")

    dataset = ForgeryDataset(
        authentic_path=authentic_path,
        forged_path=forged_path,
        masks_path=masks_path,
        supp_forged_path=supp_forged_path if os.path.isdir(supp_forged_path) else None,
        supp_masks_path=supp_masks_path if os.path.isdir(supp_masks_path) else None,
        transform=get_val_transform(img_size=256),
        is_train=False,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=detection_collate_fn,  # IMPORTANT: list-of-images/list-of-targets
        persistent_workers=True,
    )

    # --------------------
    # model
    # --------------------
    model = Mask2FormerForgeryModel(
        num_queries=15,
        d_model=256,
        backbone_trainable=False,
        authenticity_penalty_weight=5.0,
        auth_gate_forged_threshold=-1.0,  # disabled
        default_mask_threshold=0.5,
        default_cls_threshold=0.5,
    ).to(device)

    state = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # --------------------
    # sweep config
    # --------------------
    cls_thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # gate fixed to disabled, mask threshold fixed
    df_rows, df_agg = run_threshold_sweep(
        model=model,
        loader=loader,
        device=device,
        gates=[-1.0],
        cls_thresholds=cls_thresholds,
        mask_thresholds=[0.5],
        out_csv_path=OUT_DIR / "cls_threshold_sweep.csv",
    )

    print("Saved:")
    print(OUT_DIR / "cls_threshold_sweep.csv")
    print(OUT_DIR / "cls_threshold_sweep_agg.csv")


if __name__ == "__main__":
    main()
