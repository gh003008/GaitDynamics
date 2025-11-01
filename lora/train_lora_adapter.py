#!/usr/bin/env python3
import os
import sys
import json
import math
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Local imports from repo
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)
from args import parse_opt
from model.dance_decoder import DanceDecoder
import model.model as model_mod
from model.model import GaussianDiffusion
from model.utils import maybe_wrap
from data.preprocess import Normalizer
from data.scaler import MinMaxScaler
from importlib.machinery import SourceFileLoader
lora_mod = SourceFileLoader("lora_mod", os.path.join(REPO_ROOT, "lora", "lora.py")).load_module()
inject_lora = lora_mod.inject_lora
lora_parameters = lora_mod.lora_parameters
count_trainable_parameters = lora_mod.count_trainable_parameters


@dataclass
class TrialMeta:
    file_path: str
    T: int
    C: int
    height_m: float
    weight_kg: float
    sampling_rate: int
    columns: List[str]
    pos_vec: np.ndarray
    available_win_start: List[int]
    dset_name: str
    sub_and_trial_name: str


class CrouchNPZDataset(Dataset):
    """
    Loads crouch gait trials from an NPZ tree and yields random clean windows normalized to the base model's normalizer.

    Returns a tuple compatible with MotionModel training:
      (converted_pose_norm [W,C], model_offsets [4,4,20], i_trial, slice_index, height_m, cond[6])
    """
    def __init__(self, root: str, opt, window_len: int, normalizer: Normalizer, dset_keyword: str = "crouch"):
        super().__init__()
        self.root = root
        self.window_len = window_len
        self.opt = opt
        self.normalizer = normalizer
        self.cond = torch.zeros(6).float()

        # Discover files
        self.files = sorted(glob.glob(os.path.join(root, "**", "*.npz"), recursive=True))
        if not self.files:
            raise FileNotFoundError(f"No NPZ files found under {root}")

        # Scan metadata and build index
        self.trials: List[TrialMeta] = []
        self.total_windows = 0
        for f in self.files:
            npz = np.load(f, allow_pickle=True)
            states = npz["model_states"]  # (T, C)
            columns = [str(x) for x in npz["model_states_columns"]]
            height_m = float(npz["height_m"]) if npz["height_m"].shape == () else float(npz["height_m"][()])
            weight_kg = float(npz["weight_kg"]) if npz["weight_kg"].shape == () else float(npz["weight_kg"][()])
            sampling_rate = int(npz["sampling_rate"]) if npz["sampling_rate"].shape == () else int(npz["sampling_rate"][()])
            pos_vec = np.array(npz["pos_vec"]).astype(np.float32)
            probably_missing = np.array(npz["probably_missing"]).astype(bool)
            T, C = states.shape

            # Verify / map columns to model order
            model_cols = self.opt.model_states_column_names
            if len(columns) != len(model_cols):
                raise ValueError(f"Column count mismatch for {f}: npz {len(columns)} vs opt {len(model_cols)}")
            # Build mapping indices from npz order to model order
            map_idx = [columns.index(c) for c in model_cols]

            # Compute available window starts where GRF not missing (optional but sensible)
            # Follow similar logic to TrialData.set_available_win_start
            available = []
            # run-length encode probably_missing
            run_val = None
            run_len = 0
            runs = []
            for v in probably_missing:
                if run_val is None:
                    run_val = v
                    run_len = 1
                elif v == run_val:
                    run_len += 1
                else:
                    runs.append((run_val, run_len))
                    run_val = v
                    run_len = 1
            if run_val is not None:
                runs.append((run_val, run_len))
            # compute starts
            cursor = 0
            for val, count in runs:
                if (not val) and count >= window_len:
                    available.extend(list(range(cursor, cursor + count - window_len + 1)))
                cursor += count

            # Fallback: if no clean window, allow any window
            if len(available) == 0 and T >= window_len:
                available = list(range(0, T - window_len + 1))

            sub_dir = Path(f).parent
            dset_name = dset_keyword
            # subject and trial naming
            parts = list(Path(f).parts)
            try:
                subj = parts[-3]
                cond = parts[-2]
                trial = Path(f).stem
                sub_and_trial_name = f"{subj}_{cond}_{trial}"
            except Exception:
                sub_and_trial_name = Path(f).stem

            meta = TrialMeta(
                file_path=f,
                T=T,
                C=C,
                height_m=height_m,
                weight_kg=weight_kg,
                sampling_rate=sampling_rate,
                columns=columns,
                pos_vec=pos_vec,
                available_win_start=available,
                dset_name=dset_name,
                sub_and_trial_name=sub_and_trial_name,
            )
            meta.map_idx = map_idx  # attach lazily
            if T >= window_len and len(available) > 0:
                self.trials.append(meta)
                self.total_windows += len(available)

        if len(self.trials) == 0:
            raise RuntimeError("No valid trials/windows found for training.")

        # Pre-build identity model offsets [4,4,20]
        # The FK losses are disabled in training, but the call still executes; so provide a valid tensor.
        eye = torch.eye(4)
        offsets = torch.stack([eye.clone() for _ in range(20)], dim=2)
        self.model_offsets = offsets.float()

        # Bookkeeping mapping for __getitem__ sampling
        # Build trial index to (file_path, starts)
        self.trial_starts: List[List[int]] = [t.available_win_start for t in self.trials]

    def __len__(self) -> int:
        # Like MotionDataset, provide a pseudo length for random sampling
        return self.opt.pseudo_dataset_len

    def __getitem__(self, idx: int):
        # Random trial and random window within it
        i_trial = np.random.randint(0, len(self.trials))
        trial = self.trials[i_trial]
        starts = trial.available_win_start
        sidx = int(starts[np.random.randint(0, len(starts))])

        npz = np.load(trial.file_path, allow_pickle=True)
        states = np.array(npz["model_states"], dtype=np.float32)  # (T, C)
        # Reorder to match model
        states = states[:, trial.map_idx]
        # Slice window
        win = states[sidx:sidx + self.window_len, :]
        # Normalize to base scaler
        win_t = torch.from_numpy(win)
        win_t = self.normalizer.normalize(win_t)

        # Outputs
        model_offsets = self.model_offsets  # [4,4,20]
        height_m = torch.tensor(float(trial.height_m)).float()
        cond = self.cond
        return (win_t, model_offsets, i_trial, sidx, height_m, cond)

    # For logging in train loop
    def get_attributes_of_trials(self) -> Tuple[List[str], List[str]]:
        sub_and_trial_names = [t.sub_and_trial_name for t in self.trials]
        dset_names = [t.dset_name for t in self.trials]
        return sub_and_trial_names, dset_names


def save_lora_adapter(model: torch.nn.Module, save_path: str, extra: Dict = None):
    state = {}
    # Collect only LoRA submodule parameters
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            if module.lora_A is not None:
                state[f"{name}.lora_A.weight"] = module.lora_A.weight.detach().cpu()
            if module.lora_B is not None:
                state[f"{name}.lora_B.weight"] = module.lora_B.weight.detach().cpu()
    package = {"lora_state_dict": state}
    if extra:
        package.update(extra)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(package, save_path)


def main():
    opt = parse_opt()

    # CLI additions via env vars or defaults
    # Expect envs or default paths
    exp_name = os.environ.get('LORA_EXP_NAME', 'lora_crouch')
    data_root = os.environ.get('LORA_DATA_ROOT', 'gdp_ld_crouch')
    save_root = os.environ.get('LORA_SAVE_DIR', 'runs/adapter_train')
    lora_r = int(os.environ.get('LORA_R', '16'))
    lora_alpha = int(os.environ.get('LORA_ALPHA', '32'))
    lora_dropout = float(os.environ.get('LORA_DROPOUT', '0.05'))
    epochs = int(os.environ.get('LORA_EPOCHS', '5'))
    batch_size = int(os.environ.get('LORA_BATCH', str(max(1, opt.batch_size // 2))))
    lr = float(os.environ.get('LORA_LR', '1e-3'))

    assert opt.checkpoint != "", "Please pass --checkpoint to use the pretrained base model."

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build base model and diffusion (mirror MotionModel defaults)
    repr_dim = len(opt.model_states_column_names)
    horizon = opt.window_len
    base_model = DanceDecoder(
        nfeats=repr_dim,
        seq_len=horizon,
        latent_dim=256,
        ff_size=1024,
        num_layers=4,
        num_heads=4,
        dropout=0.1,
        activation=F.gelu,
    )
    diffusion = GaussianDiffusion(
        base_model,
        horizon,
        repr_dim,
        opt,
        n_timestep=1000,
        predict_epsilon=False,
        loss_type="l2",
        use_p2=False,
        cond_drop_prob=0.,
        guidance_weight=2,
    ).to(device)

    # Optionally disable FK inside loss to avoid NaNs from random preds during LoRA warmup
    # Default: disabled (env 'LORA_DISABLE_FK' = '1')
    if os.environ.get('LORA_DISABLE_FK', '1') == '1':
        def _safe_fk(states, offsets):
            # states: [B, T, C]
            b, t, _ = states.shape
            foot = torch.zeros((4, b, t, 3), device=states.device, dtype=states.dtype)
            joints = torch.zeros((11, b, t, 3), device=states.device, dtype=states.dtype)
            names = ['pelvis','hip_r','knee_r','ankle_r','calcn_r','mtp_r','hip_l','knee_l','ankle_l','calcn_l','mtp_l']
            segs = torch.zeros((10, b, t, 3, 3), device=states.device, dtype=states.dtype)
            return foot, joints, names, segs
        # Monkey-patch the symbol used in model.model
        model_mod.forward_kinematics = _safe_fk

    # Load pretrained weights + normalizer
    # Torch >=2.6 defaults weights_only=True; we need full pickle for embedded Normalizer
    try:
        from torch.serialization import add_safe_globals
        add_safe_globals([Normalizer, MinMaxScaler])
    except Exception:
        pass
    ckpt = torch.load(opt.checkpoint, map_location=device, weights_only=False)
    normalizer = ckpt.get("normalizer", None)
    if normalizer is None:
        raise RuntimeError("Checkpoint missing embedded normalizer; required for consistent scaling.")
    diffusion.model.load_state_dict(maybe_wrap(ckpt["ema_state_dict"], 1))
    diffusion.set_normalizer(normalizer)

    # Inject LoRA and freeze base weights
    replaced, _ = inject_lora(diffusion.model, r=lora_r, alpha=lora_alpha, dropout=lora_dropout, include_mha_out_proj=True)
    # Freeze all, then re-enable LoRA params
    for p in diffusion.model.parameters():
        p.requires_grad = False
    for p in lora_parameters(diffusion.model):
        p.requires_grad = True

    trainable_count = count_trainable_parameters(diffusion.model)
    print(f"LoRA injected. Replaced modules: {replaced}. Trainable parameters: {trainable_count}")

    # Dataset & loader
    dataset = CrouchNPZDataset(root=data_root, opt=opt, window_len=opt.window_len, normalizer=normalizer)
    num_workers = int(os.environ.get('LORA_WORKERS', '0'))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda')
    )

    # Optimizer
    optimizer = torch.optim.AdamW(lora_parameters(diffusion.model), lr=lr)

    # Save dir
    save_dir = Path(save_root) / f"{exp_name}_{torch.randint(0, 10**8, (1,)).item()}"
    (save_dir / 'weights').mkdir(parents=True, exist_ok=True)
    config_path = save_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump({
            "checkpoint": opt.checkpoint,
            "data_root": data_root,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "save_dir": str(save_dir),
            "wandb": False,
            "wandb_project": "GaitDynamics-LoRA",
            "wandb_name": exp_name,
        }, f, indent=2)

    diffusion.train()
    for epoch in range(1, epochs + 1):
        running = 0.0
        n = 0
        for batch in loader:
            x_start, model_offsets, i_trial, slice_index, height_m, cond = batch
            # Move tensors
            x_start = x_start.to(device)
            model_offsets = model_offsets.to(device)
            height_m = height_m.to(device)
            cond = cond.to(device)

            # diffusion.loss expects x tuple and cond
            total_loss, losses = diffusion((x_start, model_offsets, i_trial, slice_index, height_m, cond), cond, t_override=None)
            # Skip NaN batches safely
            if torch.isnan(total_loss):
                continue
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            running += float(total_loss.detach().cpu().item())
            n += 1

        avg = running / max(1, n)
        print(f"Epoch {epoch}/{epochs} - loss: {avg:.6f}")

        # Save every epoch
        save_lora_adapter(diffusion.model, str(save_dir / 'weights' / f'epoch-{epoch}-lora.pt'),
                          extra={"normalizer": normalizer})

    print(f"[DONE] Saved LoRA adapters under: {save_dir}")


if __name__ == '__main__':
    main()
