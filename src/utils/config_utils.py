import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple, Union

import yaml


PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Paths / project layout helpers
# ---------------------------------------------------------------------------

# project_root/src/utils/config_utils.py -> project_root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


# ---------------------------------------------------------------------------
# Core YAML loading / merging
# ---------------------------------------------------------------------------

def _deep_update(base: MutableMapping[str, Any], other: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """
    Recursively update mapping `base` with values from `other`.

    - Dicts are merged recursively.
    - Non-dict values are overwritten.
    """
    for k, v in other.items():
        if (
            k in base
            and isinstance(base[k], MutableMapping)
            and isinstance(v, MutableMapping)
        ):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_yaml(path: PathLike) -> Dict[str, Any]:
    """Load a YAML file into a dict (empty dict if file is empty)."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _ensure_yaml_suffix(path: Path) -> Path:
    """Add `.yaml` if path has no suffix."""
    if path.suffix:
        return path
    return path.with_suffix(".yaml")


def resolve_config_path(name_or_path: PathLike) -> Path:
    """
    Resolve a config identifier to a concrete file path.

    Rules:
    - If the given path exists as-is, use it.
    - Otherwise, treat it as relative to CONFIG_DIR.
    - If no extension is given, `.yaml` is assumed.
    """
    p = Path(name_or_path)

    # Direct path (absolute or relative) that exists
    if p.is_file():
        return p.resolve()

    # Try under CONFIG_DIR
    candidate = CONFIG_DIR / p
    candidate = _ensure_yaml_suffix(candidate)
    if candidate.is_file():
        return candidate.resolve()

    # Fallback: assume user passed a path that just doesn't exist yet
    # (fail later on actual load attempt)
    return _ensure_yaml_suffix(p).resolve()


def load_and_merge_configs(
    config_ids: Sequence[PathLike],
    base_config: Optional[PathLike] = None,
) -> Dict[str, Any]:
    """
    Load and deep-merge multiple YAML configs.

    - If base_config is not None, it is loaded first.
    - Then each config in `config_ids` is loaded in order.
    - Later configs override earlier ones (deep merge).
    """
    result: Dict[str, Any] = {}

    def _load_one(cfg_id: PathLike) -> Dict[str, Any]:
        cfg_path = resolve_config_path(cfg_id)
        return load_yaml(cfg_path)

    if base_config is not None:
        base_path = resolve_config_path(base_config)
        base_cfg = load_yaml(base_path)
        _deep_update(result, base_cfg)

    for cfg_id in config_ids:
        cfg = _load_one(cfg_id)
        _deep_update(result, cfg)

    return result


# ---------------------------------------------------------------------------
# CLI override handling
# ---------------------------------------------------------------------------

def _parse_scalar(value: str) -> Any:
    """
    Parse a scalar override string into a Python object.

    Uses yaml.safe_load so that:
        "true" -> True
        "3" -> 3
        "1.5" -> 1.5
        "[1,2]" -> [1, 2]
    falls back to raw string if parsing fails.
    """
    try:
        parsed = yaml.safe_load(value)
    except Exception:
        return value
    return parsed


def _set_by_dotted_key(d: MutableMapping[str, Any], dotted_key: str, value: Any) -> None:
    """
    Set `value` in nested dict `d` given a dotted key, e.g.:

        dotted_key="trainer.lr", value=1e-4
        -> d["trainer"]["lr"] = 1e-4 (creating intermediate dicts as needed)
    """
    parts = dotted_key.split(".")
    cur = d
    for k in parts[:-1]:
        if k not in cur or not isinstance(cur[k], MutableMapping):
            cur[k] = {}
        cur = cur[k]
    cur[parts[-1]] = value


def parse_overrides(overrides: Iterable[str]) -> List[Tuple[str, Any]]:
    """
    Parse a list of "KEY=VALUE" strings into (key, parsed_value) tuples.

    KEY can be dotted for nested config fields, e.g. "model.num_queries=20".
    """
    parsed: List[Tuple[str, Any]] = []
    for item in overrides:
        if "=" not in item:
            # treat bare flag as boolean True
            key = item.strip()
            if not key:
                continue
            parsed.append((key, True))
            continue

        key, raw_val = item.split("=", 1)
        key = key.strip()
        raw_val = raw_val.strip()
        if not key:
            continue

        value = _parse_scalar(raw_val)
        parsed.append((key, value))
    return parsed


def apply_overrides(cfg: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    """
    Apply a sequence of override strings ("KEY=VALUE") to `cfg` in-place and return it.
    """
    for key, value in parse_overrides(overrides):
        _set_by_dotted_key(cfg, key, value)
    return cfg

# ---------------------------------------------------------------------------
# Config wrapper / argparse integration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """
    Lightweight wrapper around a nested dict.

    Provides:
        - attribute-style access: cfg.train.lr
        - dict-style access: cfg["train"]["lr"]
        - .to_dict() for raw dict.
    """

    data: Dict[str, Any]

    def __getattr__(self, item: str) -> Any:
        try:
            return self.data[item]
        except KeyError as e:
            raise AttributeError(item) from e

    def __getitem__(self, item: str) -> Any:
        return self.data[item]

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.data)


def add_config_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Attach common config-related CLI arguments to an argparse parser.

    Adds:
        --base-config:  name/path of base config (default: base_v2.yaml under config/)
        --config, -c:   additional config(s), later override earlier
        --override, -o: KEY=VALUE overrides (can be dotted, can repeat)
    """
    parser.add_argument(
        "--base-config",
        type=str,
        default="base_v2.yaml",
        help="Base YAML config (name or path, default: base_v2.yaml).",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        action="append",
        default=None,
        help=(
            "Additional YAML config(s), name or path. "
            "Later configs override earlier configs."
        ),
    )
    parser.add_argument(
        "--override",
        "-o",
        type=str,
        action="append",
        default=None,
        help=(
            "Override config fields: KEY=VALUE. "
            "Use dotted keys for nested fields, e.g. trainer.lr=1e-4. "
            "Can be specified multiple times."
        ),
    )
    return parser


def build_config_from_args(args: argparse.Namespace) -> Config:
    """
    Build a Config object from argparse args that used `add_config_arguments`.

    - Loads base config (if not None).
    - Loads any extra configs from --config in order.
    - Applies overrides from --override.
    """
    base_cfg_name = getattr(args, "base_config", None)
    extra_cfgs = getattr(args, "config", None) or []
    overrides = getattr(args, "override", None) or []

    cfg_dict = load_and_merge_configs(extra_cfgs, base_config=base_cfg_name)
    apply_overrides(cfg_dict, overrides)

    return Config(cfg_dict)

def get_default_model_cfg():
    return dict(
        num_queries=15,
        dropout=0.1,
        authenticity_penalty_weight=5.0,
        auth_gate_forged_threshold=0.5,
        default_mask_threshold=0.5,
        default_cls_threshold=0.5,
        cost_bce=1.0,
        cost_dice=1.0,
        loss_weight_mask_bce=1.0,
        loss_weight_mask_dice=1.0,
        loss_weight_mask_cls=1.0,
        loss_weight_auth_penalty=1.0,
    )


def build_model_cfg(user_cfg: dict | None):
    cfg = get_default_model_cfg()
    if user_cfg:
        cfg.update(user_cfg)
    return cfg

def sanitize_model_kwargs(model_cfg: dict) -> dict:
    cfg = dict(model_cfg)

    # drop non-ctor / meta keys
    cfg.pop("type", None)
    cfg.pop("loss_weights", None)

    # drop unset knobs so they don't leak into ctor
    cfg = {k: v for k, v in cfg.items() if v is not None}

    return cfg
