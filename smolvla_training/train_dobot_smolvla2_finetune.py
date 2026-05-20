#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dobot E6 SmolVLA2 fine-tuning wrapper.

This file is not a pretraining script.
- The VLM backbone is SmolVLM-2 and is frozen during fine-tuning.
- The Action Expert + projectors + action head are trained on Dobot 13D data.
- chunk_size=20 is chosen for short Dobot zone-approach episodes.
- The LeRobot v2.1-style dataset is produced by convert_dobot_to_lerobot_v21.py.

Usage:
    python smolvla_training/train_dobot_smolvla2_finetune.py --check
    python smolvla_training/train_dobot_smolvla2_finetune.py --smoke --yes
    python smolvla_training/train_dobot_smolvla2_finetune.py --train --yes
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "smolvla_training" / "configs" / "dobot_smolvla2_finetune.yaml"

OFFICIAL_DEBUG_COMMANDS = [
    "which lerobot-train",
    "lerobot-train --help",
    'python -c "import lerobot, os; print(os.path.dirname(lerobot.__file__))"',
    'find $(python -c "import lerobot, os; print(os.path.dirname(lerobot.__file__))") -iname "*smol*"',
]


def _print_usage() -> None:
    print("Usage:")
    print("  python smolvla_training/train_dobot_smolvla2_finetune.py --check")
    print("  python smolvla_training/train_dobot_smolvla2_finetune.py --smoke --yes")
    print("  python smolvla_training/train_dobot_smolvla2_finetune.py --train --yes")
    print("")
    print(f"Default config: {DEFAULT_CONFIG}")


def _print_lerobot_debug_commands() -> None:
    print("[LeRobot] 확인 명령어:")
    for command in OFFICIAL_DEBUG_COMMANDS:
        print(f"  {command}")


def _import_attr(candidates: list[tuple[str, str]], required: bool = True) -> Any:
    errors = []
    for module_name, attr_name in candidates:
        try:
            module = importlib.import_module(module_name)
            return getattr(module, attr_name)
        except Exception as exc:
            errors.append(f"{module_name}.{attr_name}: {exc}")
    if required:
        _print_lerobot_debug_commands()
        raise RuntimeError("Failed to import required LeRobot symbol:\n" + "\n".join(errors))
    return None


def _as_namespace(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _as_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_as_namespace(v) for v in obj]
    return obj


def _shape_of(value: Any) -> Any:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return list(shape)
    if isinstance(value, (list, tuple)):
        if value and isinstance(value[0], (list, tuple)):
            return [len(value), len(value[0])]
        return [len(value)]
    return type(value).__name__


def _to_device(batch: Any, device: Any) -> Any:
    try:
        import torch
    except Exception:
        return batch
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: _to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [_to_device(v, device) for v in batch]
    if isinstance(batch, tuple):
        return tuple(_to_device(v, device) for v in batch)
    return batch


def _call_with_supported_kwargs(fn: Any, kwargs: dict[str, Any]) -> Any:
    signature = inspect.signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return fn(**kwargs)
    supported = {k: v for k, v in kwargs.items() if k in signature.parameters}
    return fn(**supported)


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"Config file does not exist: {path}")
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required: pip install pyyaml") from exc
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_config_path"] = str(path)
    return cfg


def check_torch_cuda() -> Any:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(f"PyTorch import failed: {exc}") from exc
    print(f"[Torch] version={torch.__version__}")
    print(f"[Torch] cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[Torch] cuda_device={torch.cuda.get_device_name(0)}")
    else:
        print("[Torch] WARNING: CUDA is not available. Training will be very slow or impossible.")
    return torch


def check_lerobot_imports() -> dict[str, Any]:
    imports = {
        "make_dataset": _import_attr([
            ("lerobot.common.datasets.factory", "make_dataset"),
            ("lerobot.common.datasets.utils", "make_dataset"),
            ("lerobot.scripts.train", "make_dataset"),
        ]),
        "make_policy": _import_attr([
            ("lerobot.common.policies.factory", "make_policy"),
            ("lerobot.common.policies.utils", "make_policy"),
            ("lerobot.scripts.train", "make_policy"),
        ]),
        "make_optimizer_and_scheduler": _import_attr([
            ("lerobot.common.optim.factory", "make_optimizer_and_scheduler"),
            ("lerobot.common.optim.optimizers", "make_optimizer_and_scheduler"),
            ("lerobot.scripts.train", "make_optimizer_and_scheduler"),
        ]),
        "save_checkpoint": _import_attr([
            ("lerobot.common.utils.train_utils", "save_checkpoint"),
            ("lerobot.common.utils.utils", "save_checkpoint"),
            ("lerobot.scripts.train", "save_checkpoint"),
        ]),
        "LeRobotDataset": _import_attr([
            ("lerobot.common.datasets.lerobot_dataset", "LeRobotDataset"),
        ]),
        "SmolVLA2Config": _import_attr([
            ("lerobot.common.policies.smolvla2.configuration_smolvla2", "SmolVLA2Config"),
            ("lerobot.common.policies.smol_vla2.configuration_smol_vla2", "SmolVLA2Config"),
            ("lerobot.common.policies.smolvla.configuration_smolvla", "SmolVLAConfig"),
        ], required=False),
        "SmolVLA2Policy": _import_attr([
            ("lerobot.common.policies.smolvla2.modeling_smolvla2", "SmolVLA2Policy"),
            ("lerobot.common.policies.smol_vla2.modeling_smol_vla2", "SmolVLA2Policy"),
            ("lerobot.common.policies.smolvla.modeling_smolvla", "SmolVLAPolicy"),
        ], required=False),
    }
    print("[LeRobot] required factory imports OK")
    if imports["SmolVLA2Config"] is None or imports["SmolVLA2Policy"] is None:
        print("[LeRobot] WARNING: SmolVLA2Config/SmolVLA2Policy 후보 import 실패")
        _print_lerobot_debug_commands()
    else:
        print("[LeRobot] SmolVLA2Config/SmolVLA2Policy import OK")
    return imports


def check_dataset_files(cfg: dict[str, Any]) -> Path:
    root = Path(cfg["dataset_root"])
    required = [
        root / "meta" / "info.json",
        root / "meta" / "episodes.jsonl",
        root / "meta" / "tasks.jsonl",
        root / "meta" / "stats.json",
        root / "data",
        root / "videos",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("Dataset files missing:\n" + "\n".join(missing))
    print(f"[Dataset] files OK: {root}")
    return root


def load_info_json(dataset_root: str | Path) -> dict[str, Any]:
    path = Path(dataset_root) / "meta" / "info.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_info_json(info: dict[str, Any], cfg: dict[str, Any]) -> None:
    features = info.get("features", {})
    if info.get("fps") != 20:
        raise RuntimeError(f"info.json fps must be 20, got {info.get('fps')}")

    state_key = cfg["state_key"]
    action_key = cfg["action_key"]
    if features.get(state_key, {}).get("shape") != [13]:
        raise RuntimeError(f"{state_key} shape must be [13], got {features.get(state_key)}")
    if features.get(action_key, {}).get("shape") != [13]:
        raise RuntimeError(f"{action_key} shape must be [13], got {features.get(action_key)}")

    for image_key in ["observation.images.OBS_IMAGE_1", "observation.images.OBS_IMAGE_2"]:
        feature = features.get(image_key)
        if feature is None:
            raise RuntimeError(f"info.json missing image feature: {image_key}")
        if feature.get("shape") != [512, 512, 3]:
            raise RuntimeError(f"{image_key} shape must be [512, 512, 3], got {feature.get('shape')}")
    print("[Dataset] info.json OK: fps=20, state/action=13D, images=512x512x3")


def validate_episode_lengths_for_chunk(cfg: dict[str, Any]) -> None:
    dataset_root = Path(cfg["dataset_root"])
    chunk_size = int(cfg["chunk_size"])
    min_required = chunk_size + 1
    path = dataset_root / "meta" / "episodes.jsonl"
    lengths = []
    short = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            length = int(row.get("length", 0))
            episode_index = int(row.get("episode_index", len(lengths)))
            lengths.append(length)
            if length < min_required:
                short.append((episode_index, length))

    if not lengths:
        raise RuntimeError("episodes.jsonl has no episodes")
    mean_len = sum(lengths) / len(lengths)
    print(f"[Dataset] episode length min/mean/max = {min(lengths)}/{mean_len:.1f}/{max(lengths)}")
    if short:
        print(f"[Dataset] WARNING: {len(short)}/{len(lengths)} episodes shorter than chunk_size+1 ({min_required})")
        for episode_index, length in short[:10]:
            print(f"  short episode_{episode_index:06d}: length={length}")
    if len(short) / len(lengths) >= 0.20:
        raise RuntimeError(
            f"Too many short episodes for chunk_size={chunk_size}: "
            f"{len(short)}/{len(lengths)} >= 20%"
        )


def test_lerobot_dataset_load(dataset_root: str | Path, imports: dict[str, Any] | None = None) -> Any:
    if imports is None:
        imports = check_lerobot_imports()
    dataset_cls = imports["LeRobotDataset"]
    try:
        dataset = dataset_cls(str(dataset_root))
    except TypeError:
        dataset = dataset_cls(root=str(dataset_root))
    print(f"[LeRobotDataset] loaded: len={len(dataset)}")
    if len(dataset) == 0:
        raise RuntimeError("LeRobotDataset loaded but len(dataset) == 0")
    return dataset


def print_sample_summary(dataset: Any, cfg: dict[str, Any]) -> dict[str, Any]:
    sample = dataset[0]
    print("[Sample] keys:")
    for key in sorted(sample.keys()):
        print(f"  {key}: shape={_shape_of(sample[key])}")

    for key in [cfg["state_key"], cfg["action_key"], *cfg["image_keys"]]:
        if key not in sample:
            raise RuntimeError(f"Sample missing required key: {key}")
    print(f"[Sample] state shape={_shape_of(sample[cfg['state_key']])}")
    print(f"[Sample] action shape={_shape_of(sample[cfg['action_key']])}")
    return sample


def find_smolvla2_files() -> list[Path]:
    try:
        import lerobot
    except Exception as exc:
        print(f"[SmolVLA2 Search] SKIP: import lerobot failed: {exc}")
        _print_lerobot_debug_commands()
        return []
    root = Path(lerobot.__file__).resolve().parent
    matches = sorted(path for path in root.rglob("*smol*") if path.is_file())
    print(f"[SmolVLA2 Search] root={root}")
    for path in matches[:40]:
        print(f"  {path}")
    if len(matches) > 40:
        print(f"  ... {len(matches) - 40} more")
    return matches


def resolve_checkpoint_config(cfg: dict[str, Any], require_checkpoint: bool = False) -> dict[str, Any]:
    checkpoint = str(cfg.get("pretrained_checkpoint") or "").strip()
    if not checkpoint:
        msg = "pretrained_checkpoint is empty"
        if require_checkpoint:
            raise RuntimeError(msg)
        print(f"[Checkpoint] WARNING: {msg}")
        return {}

    checkpoint_path = Path(checkpoint).expanduser()
    if not checkpoint_path.exists():
        raise RuntimeError(f"pretrained_checkpoint does not exist: {checkpoint_path}")

    config_data = {}
    if checkpoint_path.is_dir():
        candidates = [
            "config.json",
            "policy_config.json",
            "preprocessor_config.json",
            "training_config.json",
            "adapter_config.json",
        ]
        for name in candidates:
            path = checkpoint_path / name
            if path.exists():
                try:
                    with path.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        config_data.update(data)
                        print(f"[Checkpoint] config loaded: {path}")
                except Exception as exc:
                    print(f"[Checkpoint] WARNING: failed to read {path}: {exc}")
    else:
        print(f"[Checkpoint] file exists: {checkpoint_path}")

    for key in ("max_state_dim", "max_action_dim"):
        if config_data.get(key) is not None:
            print(f"[Checkpoint] using {key}={config_data[key]} from checkpoint config")
        elif cfg.get(key) is not None:
            config_data[key] = cfg[key]
            print(f"[Checkpoint] using {key}={cfg[key]} from YAML")
    return config_data


def build_smolvla2_config(
    cfg: dict[str, Any],
    imports: dict[str, Any],
    checkpoint_config: dict[str, Any] | None = None,
) -> Any:
    config_cls = imports.get("SmolVLA2Config")
    checkpoint_config = checkpoint_config or {}
    if config_cls is None:
        print("[Policy Config] SmolVLA2Config class unavailable; using SimpleNamespace config")
        policy_cfg = _as_namespace(dict(cfg))
    else:
        checkpoint = str(cfg.get("pretrained_checkpoint") or "").strip()
        if checkpoint and hasattr(config_cls, "from_pretrained"):
            try:
                policy_cfg = config_cls.from_pretrained(checkpoint)
            except Exception:
                policy_cfg = config_cls()
        else:
            policy_cfg = config_cls()

    updates = {
        "policy_type": cfg["policy_type"],
        "robot_type": cfg["robot_type"],
        "fps": cfg["fps"],
        "chunk_size": cfg["chunk_size"],
        "state_dim": cfg["state_dim"],
        "action_dim": cfg["action_dim"],
        "max_state_dim": checkpoint_config.get("max_state_dim", cfg.get("max_state_dim")),
        "max_action_dim": checkpoint_config.get("max_action_dim", cfg.get("max_action_dim")),
        "image_keys": list(cfg["image_keys"]),
        "state_key": cfg["state_key"],
        "action_key": cfg["action_key"],
        "train_expert_only": bool(cfg["train_expert_only"]),
        "freeze_vision_encoder": bool(cfg["freeze_vision_encoder"]),
    }
    for key, value in updates.items():
        if value is not None:
            try:
                setattr(policy_cfg, key, value)
            except Exception:
                pass

    print(
        "[Policy Config] Dobot dims: "
        f"state_dim={cfg['state_dim']}, action_dim={cfg['action_dim']}, chunk_size={cfg['chunk_size']}"
    )
    print(
        "[Policy Config] model max dims: "
        f"max_state_dim={getattr(policy_cfg, 'max_state_dim', None)}, "
        f"max_action_dim={getattr(policy_cfg, 'max_action_dim', None)}"
    )
    return policy_cfg


def _make_dataset(cfg: dict[str, Any], imports: dict[str, Any], policy_cfg: Any | None = None) -> Any:
    make_dataset = imports["make_dataset"]
    dataset_root = str(cfg["dataset_root"])
    dataset_cfg = SimpleNamespace(
        repo_id=dataset_root,
        root=dataset_root,
        episodes=None,
        image_transforms=None,
        revision=None,
    )
    train_cfg = SimpleNamespace(
        dataset=dataset_cfg,
        policy=policy_cfg,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )
    attempts = [
        lambda: make_dataset(train_cfg),
        lambda: make_dataset(dataset_cfg, policy_cfg),
        lambda: _call_with_supported_kwargs(make_dataset, {"cfg": train_cfg, "policy_cfg": policy_cfg}),
        lambda: _call_with_supported_kwargs(make_dataset, {"dataset_cfg": dataset_cfg, "policy_cfg": policy_cfg}),
    ]
    errors = []
    for attempt in attempts:
        try:
            dataset = attempt()
            print("[Dataset] make_dataset OK")
            return dataset
        except Exception as exc:
            errors.append(str(exc))

    print("[Dataset] WARNING: make_dataset failed; falling back to LeRobotDataset for diagnostics")
    for error in errors[:4]:
        print(f"  make_dataset error: {error}")
    return test_lerobot_dataset_load(dataset_root, imports)


def load_pretrained_policy_or_fail(cfg: dict[str, Any], imports: dict[str, Any], dataset: Any | None = None) -> Any:
    checkpoint_config = resolve_checkpoint_config(cfg, require_checkpoint=True)
    policy_cfg = build_smolvla2_config(cfg, imports, checkpoint_config)
    checkpoint = str(cfg["pretrained_checkpoint"]).strip()
    policy_cls = imports.get("SmolVLA2Policy")

    if policy_cls is not None and hasattr(policy_cls, "from_pretrained"):
        try:
            policy = policy_cls.from_pretrained(checkpoint, config=policy_cfg)
            print("[Policy] loaded with SmolVLA2Policy.from_pretrained(checkpoint, config=...)")
            return policy, policy_cfg
        except TypeError:
            policy = policy_cls.from_pretrained(checkpoint)
            print("[Policy] loaded with SmolVLA2Policy.from_pretrained(checkpoint)")
            return policy, policy_cfg
        except Exception as exc:
            print(f"[Policy] WARNING: SmolVLA2Policy.from_pretrained failed: {exc}")

    make_policy = imports["make_policy"]
    attempts = [
        lambda: make_policy(policy_cfg, dataset_stats=getattr(dataset, "stats", None)),
        lambda: make_policy(policy_cfg, dataset),
        lambda: _call_with_supported_kwargs(
            make_policy,
            {"cfg": policy_cfg, "dataset_stats": getattr(dataset, "stats", None), "dataset": dataset},
        ),
    ]
    errors = []
    for attempt in attempts:
        try:
            policy = attempt()
            print("[Policy] loaded with LeRobot make_policy")
            return policy, policy_cfg
        except Exception as exc:
            errors.append(str(exc))
    _print_lerobot_debug_commands()
    raise RuntimeError("Failed to load pretrained SmolVLA2 policy:\n" + "\n".join(errors))


def apply_smolvla2_freeze_config(policy: Any, policy_cfg: Any, cfg: dict[str, Any]) -> None:
    for key in ("train_expert_only", "freeze_vision_encoder"):
        try:
            setattr(policy_cfg, key, bool(cfg[key]))
        except Exception:
            pass

    freeze_terms = (
        "smolvlm",
        "smol_vlm",
        "siglip",
        "smollm",
        "vision_model",
        "language_model",
        "vlm",
        "backbone",
    )
    train_terms = []
    if cfg["train_action_expert"]:
        train_terms.extend(["action_expert", "flow", "flow_matching"])
    if cfg["train_state_projector"]:
        train_terms.extend(["state_proj", "state_projector"])
    if cfg["train_action_in_projector"]:
        train_terms.extend(["action_in_proj", "action_in_projector"])
    if cfg["train_action_out_projector"]:
        train_terms.extend(["action_out_proj", "action_out_projector"])
    if cfg["train_action_head"]:
        train_terms.extend(["action_head", "action_decoder"])
    if cfg["train_feature_projector"]:
        train_terms.extend(["feature_proj", "feature_projector"])

    for name, param in policy.named_parameters():
        lname = name.lower()
        should_train = any(term in lname for term in train_terms)
        should_freeze = any(term in lname for term in freeze_terms)
        param.requires_grad = bool(should_train and not should_freeze)
    print("[Freeze] applied: VLM/backbone frozen, expert/projector/head selected for training")


def print_trainable_parameters(policy: Any) -> dict[str, int]:
    total = 0
    trainable = 0
    trainable_names = []
    for name, param in policy.named_parameters():
        n = int(param.numel())
        total += n
        if param.requires_grad:
            trainable += n
            trainable_names.append(name)
    pct = 100.0 * trainable / max(total, 1)
    print(f"[Trainable] {trainable:,}/{total:,} parameters ({pct:.3f}%)")
    print("[Trainable] trainable parameter names:")
    for name in trainable_names[:120]:
        print(f"  {name}")
    if len(trainable_names) > 120:
        print(f"  ... {len(trainable_names) - 120} more")
    return {"total": total, "trainable": trainable, "count": len(trainable_names)}


def validate_trainable_scope(policy: Any) -> None:
    frozen_warning_terms = ("smolvlm", "smol_vlm", "siglip", "smollm", "vision_model", "language_model", "vlm", "backbone")
    train_terms = (
        "action_expert",
        "flow",
        "flow_matching",
        "state_proj",
        "state_projector",
        "action_in_proj",
        "action_out_proj",
        "action_head",
        "feature_proj",
        "feature_projector",
    )
    trainable_names = [name for name, param in policy.named_parameters() if param.requires_grad]
    for name in trainable_names:
        if any(term in name.lower() for term in frozen_warning_terms):
            print(f"[Trainable] WARNING: VLM/backbone parameter is trainable: {name}")
    if not any(any(term in name.lower() for term in train_terms) for name in trainable_names):
        raise RuntimeError("No expert/projector/head parameters are trainable")


def _extract_loss(output: Any) -> Any:
    if isinstance(output, dict):
        if "loss" in output:
            return output["loss"]
        for value in output.values():
            if hasattr(value, "ndim") and getattr(value, "ndim", 1) == 0:
                return value
    if hasattr(output, "loss"):
        return output.loss
    return output


def _policy_forward(policy: Any, batch: dict[str, Any]) -> Any:
    try:
        return policy(batch)
    except TypeError:
        try:
            return policy.forward(batch)
        except TypeError:
            return policy(**batch)


def _make_loader(dataset: Any, cfg: dict[str, Any], batch_size: int | None = None) -> Any:
    import torch
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=batch_size or int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )


def run_smoke_forward(cfg: dict[str, Any], imports: dict[str, Any], policy: Any, dataset: Any) -> None:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    policy.train()
    loader = _make_loader(dataset, cfg, batch_size=1)
    batch = next(iter(loader))
    batch = _to_device(batch, device)
    with torch.autocast(device_type="cuda", enabled=bool(cfg["use_amp"]) and device.type == "cuda"):
        output = _policy_forward(policy, batch)
        loss = _extract_loss(output)
    if not torch.is_tensor(loss):
        raise RuntimeError(f"Smoke forward did not return a tensor loss/output: {type(loss)}")
    if not torch.isfinite(loss.detach()).all():
        raise RuntimeError(f"Smoke forward produced non-finite loss: {loss}")
    print("[Smoke] 1 batch forward OK")


def _make_optimizer_scheduler(cfg: dict[str, Any], imports: dict[str, Any], policy: Any) -> tuple[Any, Any]:
    make_optimizer_and_scheduler = imports["make_optimizer_and_scheduler"]
    optim_cfg = SimpleNamespace(
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    train_cfg = SimpleNamespace(optimizer=optim_cfg, scheduler=None, policy=None)
    attempts = [
        lambda: make_optimizer_and_scheduler(train_cfg, policy),
        lambda: make_optimizer_and_scheduler(policy, train_cfg),
        lambda: _call_with_supported_kwargs(
            make_optimizer_and_scheduler,
            {"cfg": train_cfg, "policy": policy, "params": [p for p in policy.parameters() if p.requires_grad]},
        ),
    ]
    errors = []
    for attempt in attempts:
        try:
            result = attempt()
            if isinstance(result, tuple):
                if len(result) == 2:
                    return result
                if len(result) >= 1:
                    return result[0], None
            return result, None
        except Exception as exc:
            errors.append(str(exc))
    raise RuntimeError("make_optimizer_and_scheduler failed:\n" + "\n".join(errors))


def _save_checkpoint(imports: dict[str, Any], output_dir: Path, step: int, policy: Any, optimizer: Any, scheduler: Any, cfg: dict[str, Any]) -> None:
    save_checkpoint = imports["save_checkpoint"]
    ckpt_dir = output_dir / "checkpoints" / f"step_{step:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    attempts = [
        lambda: save_checkpoint(ckpt_dir, step, policy, optimizer, scheduler),
        lambda: save_checkpoint(ckpt_dir, policy, optimizer, scheduler, step),
        lambda: _call_with_supported_kwargs(
            save_checkpoint,
            {
                "checkpoint_dir": ckpt_dir,
                "output_dir": ckpt_dir,
                "step": step,
                "policy": policy,
                "model": policy,
                "optimizer": optimizer,
                "scheduler": scheduler,
                "cfg": _as_namespace(cfg),
            },
        ),
    ]
    errors = []
    for attempt in attempts:
        try:
            attempt()
            print(f"[Checkpoint] saved: {ckpt_dir}")
            return
        except Exception as exc:
            errors.append(str(exc))
    raise RuntimeError("save_checkpoint failed:\n" + "\n".join(errors))


def run_finetune(cfg: dict[str, Any], imports: dict[str, Any], policy: Any, dataset: Any) -> None:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as f:
        json.dump({k: v for k, v in cfg.items() if not k.startswith("_")}, f, ensure_ascii=False, indent=2)

    policy.to(device)
    policy.train()
    loader = _make_loader(dataset, cfg)
    iterator = iter(loader)
    optimizer, scheduler = _make_optimizer_scheduler(cfg, imports, policy)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["use_amp"]) and device.type == "cuda")
    num_steps = int(cfg["num_train_steps"])
    save_every = int(cfg["save_every"])

    for step in range(1, num_steps + 1):
        step_start = time.time()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = _to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=bool(cfg["use_amp"]) and device.type == "cuda"):
            output = _policy_forward(policy, batch)
            loss = _extract_loss(output)
        if not torch.is_tensor(loss):
            raise RuntimeError(f"Training forward did not return a tensor loss/output: {type(loss)}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in policy.parameters() if p.requires_grad and p.grad is not None],
            max_norm=10.0,
        )
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - step_start
        print(
            f"[Train] step={step}/{num_steps} loss={float(loss.detach().cpu()):.6f} "
            f"lr={lr:.3e} grad_norm={float(grad_norm):.4f} step_time={elapsed:.3f}s"
        )
        if step % save_every == 0:
            _save_checkpoint(imports, output_dir, step, policy, optimizer, scheduler, cfg)

    _save_checkpoint(imports, output_dir, num_steps, policy, optimizer, scheduler, cfg)
    print(f"[Train] final checkpoint saved under: {output_dir / 'checkpoints'}")


def _run_check(cfg: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    random.seed(int(cfg["seed"]))
    check_torch_cuda()
    imports = check_lerobot_imports()
    dataset_root = check_dataset_files(cfg)
    info = load_info_json(dataset_root)
    validate_info_json(info, cfg)
    validate_episode_lengths_for_chunk(cfg)
    dataset = test_lerobot_dataset_load(dataset_root, imports)
    print_sample_summary(dataset, cfg)
    find_smolvla2_files()
    resolve_checkpoint_config(cfg, require_checkpoint=False)
    print("[Check] OK")
    return imports, dataset


def main() -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    if not (args.check or args.smoke or args.train):
        _print_usage()
        return 0
    if (args.smoke or args.train) and not args.yes:
        raise RuntimeError("--smoke/--train requires --yes")

    cfg = load_yaml_config(args.config)
    if cfg.get("policy_type") != "smolvla2":
        raise RuntimeError(f"policy_type must be 'smolvla2', got {cfg.get('policy_type')}")
    if int(cfg.get("state_dim", -1)) != 13 or int(cfg.get("action_dim", -1)) != 13:
        raise RuntimeError("Dobot E6 fine-tuning requires state_dim=13 and action_dim=13")
    if int(cfg.get("chunk_size", -1)) != 20:
        raise RuntimeError("This Dobot setup requires chunk_size=20")

    imports, dataset = _run_check(cfg)
    if args.check and not (args.smoke or args.train):
        return 0

    policy, policy_cfg = load_pretrained_policy_or_fail(cfg, imports, dataset)
    apply_smolvla2_freeze_config(policy, policy_cfg, cfg)
    print_trainable_parameters(policy)
    validate_trainable_scope(policy)

    if args.smoke:
        run_smoke_forward(cfg, imports, policy, dataset)
    if args.train:
        run_finetune(cfg, imports, policy, dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
