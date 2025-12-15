# inference/sweep_loss_weight_mask_cls.py
import subprocess

WEIGHTS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
BASE_CFG = "base.yaml"

for w in WEIGHTS:
    cmd = [
        "python", "-m", "src.cli", "cv",
        "-c", BASE_CFG,
        "-o", f"model.loss_weight_mask_cls={w}",
        "-o", f"trainer.name=cls_w{w}",
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
