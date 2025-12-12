# inference/cls_threshold_sweep.py

import torch
from pathlib import Path
from torch.utils.data import DataLoader

from src.data.dataloader import ForgeryDataset
from src.models.mask2former_v1 import Mask2FormerForgeryModel
from src.infer_multi_configs import run_threshold_sweep


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------
    # paths
    # --------------------
    WEIGHTS_PATH = "weights/full_train/model_full_data_baseline.pth"
    OUT_DIR = Path("analysis/cls_threshold_sweep")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------
    # dataset / loader
    # --------------------
    dataset = ForgeryDataset(
        split="train",
        return_masks=True,
        transforms=None,   # no aug at inference
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
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
