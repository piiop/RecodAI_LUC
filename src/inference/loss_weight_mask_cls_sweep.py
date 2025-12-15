# inference/sweep_loss_weight_mask_cls_full_train.py
import subprocess

WEIGHTS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
BASE_CFG = "base.yaml"

for w in WEIGHTS:
    cmd = [
        "python", "-m", "src.cli", "full-train",
        "-c", BASE_CFG,
        "-o", f"model.loss_weight_mask_cls={w}",
        "-o", f"trainer.name=full_cls_w{w}",
        "-o", f"trainer.save_path=weights/full_train/full_cls_w{w}.pth",
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

