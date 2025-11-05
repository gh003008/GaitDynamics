#!/usr/bin/env python3
import os
import sys
import json
import math
import glob
import time
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

# Optional: Weights & Biases
_WANDB_AVAILABLE = False
try:
    import wandb  # type: ignore
    _WANDB_AVAILABLE = True
except Exception:
    _WANDB_AVAILABLE = False

# Optional: tqdm progress bar
_TQDM_AVAILABLE = False
try:
    from tqdm import tqdm  # type: ignore
    _TQDM_AVAILABLE = True
except Exception:
    _TQDM_AVAILABLE = False
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
        # If root contains multiple conditions, restrict to the requested keyword (default: 'crouch')
        if dset_keyword:
            kw = f"{os.sep}{dset_keyword}{os.sep}"
            self.files = [f for f in self.files if kw in f]
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
    weight_decay = float(os.environ.get('LORA_WD', '0.0'))
    grad_clip = float(os.environ.get('LORA_CLIP', '1.0'))
    num_workers = int(os.environ.get('LORA_WORKERS', '0'))
    log_every = int(os.environ.get('LORA_LOG_EVERY', '50'))
    use_wandb = os.environ.get('WANDB_ENABLE', '0') == '1' and _WANDB_AVAILABLE
    wandb_project = os.environ.get('WANDB_PROJECT', 'GaitDynamics-LoRA')
    wandb_entity = os.environ.get('WANDB_ENTITY', None)
    wandb_name = os.environ.get('WANDB_NAME', exp_name)
    only_simple_loss = os.environ.get('LORA_ONLY_SIMPLE_LOSS', '1') == '1'

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

    # Optionally force a "simple loss only" training step to minimize numerical issues during warmup
    # This replaces GaussianDiffusion.p_losses at runtime to compute only the reconstruction loss (with p2 weighting)
    if only_simple_loss:
        def _simple_p_losses(self, x, cond, t):
            x_start, model_offsets, _, _, height_m, _ = x
            noise = torch.randn_like(x_start)
            x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
            x_recon = self.model(x_noisy, cond, t)
            model_out = x_recon
            target = x_start if not self.predict_epsilon else noise
            loss_simple = self.loss_fn(model_out, target, reduction="none")
            # p2 weighting without external helper
            try:
                b = x_start.shape[0]
                weight = self.p2_loss_weight[t].view(b, 1, 1)
                loss_simple = loss_simple * weight
            except Exception:
                pass
            ls = loss_simple.mean()
            zeros = torch.tensor(0., device=ls.device)
            return ls, [ls, zeros, zeros, zeros, zeros, loss_simple]
        # Bind the method to the diffusion instance
        import types as _types
        diffusion.p_losses = _types.MethodType(_simple_p_losses, diffusion)

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
    # Ensure normalizer/scaler tensors live on CPU to avoid CUDA init in DataLoader workers
    def _force_normalizer_cpu(norm: Normalizer):
        try:
            scaler = norm.scaler
            for name in [
                'scale_', 'min_', 'data_min_', 'data_max_', 'data_range_', 'data_mean_'
            ]:
                if hasattr(scaler, name):
                    t = getattr(scaler, name)
                    if isinstance(t, torch.Tensor):
                        setattr(scaler, name, t.detach().cpu())
        except Exception:
            pass
        return norm
    normalizer = _force_normalizer_cpu(normalizer)
    diffusion.model.load_state_dict(maybe_wrap(ckpt["ema_state_dict"], 1))
    diffusion.set_normalizer(normalizer)

    # W&B init (optional) BEFORE dataset creation to allow sweep overrides
    use_wandb = use_wandb and _WANDB_AVAILABLE
    run = None
    if use_wandb:
        base_config = {
            "checkpoint": opt.checkpoint,
            "data_root": data_root,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "grad_clip": grad_clip,
            "num_workers": num_workers,
            "window_len": int(opt.window_len),
            "target_sampling_rate": int(opt.target_sampling_rate),
            "pseudo_dataset_len": int(opt.pseudo_dataset_len),
            "only_simple_loss": bool(only_simple_loss),
        }
        run = wandb.init(project=wandb_project, entity=wandb_entity, name=wandb_name, config=base_config)
        cfg = wandb.config
        # Allow sweep to override some hparams and dataset params
        def _ov(key, cur, cast=lambda x: x):
            try:
                if key in cfg:
                    return cast(cfg[key])
            except Exception:
                pass
            return cur
        data_root = _ov('data_root', data_root, str)
        lora_r = _ov('lora_r', lora_r, int)
        lora_alpha = _ov('lora_alpha', lora_alpha, int)
        lora_dropout = _ov('lora_dropout', lora_dropout, float)
        epochs = _ov('epochs', epochs, int)
        batch_size = _ov('batch_size', batch_size, int)
        lr = _ov('lr', lr, float)
        weight_decay = _ov('weight_decay', weight_decay, float)
        grad_clip = _ov('grad_clip', grad_clip, float)
        log_every = _ov('log_every', log_every, int)
        only_simple_loss = _ov('only_simple_loss', only_simple_loss, bool)
        # Override model options if provided
        new_win = _ov('window_len', int(opt.window_len), int)
        if new_win != int(opt.window_len):
            opt.window_len = int(new_win)
        new_sr = _ov('target_sampling_rate', int(opt.target_sampling_rate), int)
        if new_sr != int(opt.target_sampling_rate):
            opt.target_sampling_rate = int(new_sr)
        new_pseudo = _ov('pseudo_dataset_len', int(opt.pseudo_dataset_len), int)
        if new_pseudo != int(opt.pseudo_dataset_len):
            opt.pseudo_dataset_len = int(new_pseudo)

        # If the sweep toggled only_simple_loss after diffusion creation, rebind
        if only_simple_loss:
            def _simple_p_losses(self, x, cond, t):
                x_start, model_offsets, _, _, height_m, _ = x
                noise = torch.randn_like(x_start)
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                x_recon = self.model(x_noisy, cond, t)
                model_out = x_recon
                target = x_start if not self.predict_epsilon else noise
                loss_simple = self.loss_fn(model_out, target, reduction="none")
                try:
                    b = x_start.shape[0]
                    weight = self.p2_loss_weight[t].view(b, 1, 1)
                    loss_simple = loss_simple * weight
                except Exception:
                    pass
                ls = loss_simple.mean()
                zeros = torch.tensor(0., device=ls.device)
                return ls, [ls, zeros, zeros, zeros, zeros, loss_simple]
            import types as _types
            diffusion.p_losses = _types.MethodType(_simple_p_losses, diffusion)

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
    trial_names, dset_names = dataset.get_attributes_of_trials()
    total_trials = len(dataset.trials)
    total_windows = dataset.total_windows
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda')
    )
    # If drop_last would zero out all batches, relax it
    try:
        if len(loader) == 0:
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=num_workers,
                pin_memory=(device.type == 'cuda')
            )
    except Exception:
        pass

    # Optimizer
    optimizer = torch.optim.AdamW(lora_parameters(diffusion.model), lr=lr, weight_decay=weight_decay)
    if use_wandb:
        # Log dataset summary table (truncated)
        try:
            table = wandb.Table(columns=["idx","trial","dset"])
            for i, (n, d) in enumerate(zip(trial_names[:50], dset_names[:50])):
                table.add_data(i, n, d)
            wandb.log({"dataset_preview": table})
        except Exception:
            pass

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
            "weight_decay": weight_decay,
            "grad_clip": grad_clip,
            "num_workers": num_workers,
            "save_dir": str(save_dir),
            "wandb": bool(use_wandb),
            "wandb_project": wandb_project,
            "wandb_name": wandb_name,
            "dataset_trials": int(total_trials),
            "dataset_total_windows": int(total_windows),
            "pseudo_dataset_len": int(opt.pseudo_dataset_len),
            "only_simple_loss": bool(only_simple_loss),
        }, f, indent=2)

    diffusion.train()
    global_step = 0
    for epoch in range(1, epochs + 1):
        running = 0.0
        n = 0
        t0 = time.time()

        iterator = loader
        use_bar = _TQDM_AVAILABLE and (os.environ.get('LORA_TQDM', '1') == '1')
        if use_bar:
            try:
                total_batches = len(loader)
            except Exception:
                total_batches = None
            desc = f"Epoch {epoch}/{epochs}"
            iterator = tqdm(loader, total=total_batches, desc=desc, dynamic_ncols=True, leave=False)

        # Log loader/dataset sizes at epoch start
        if use_wandb and epoch == 1:
            try:
                wandb.log({
                    "dataset/num_trials": total_trials,
                    "dataset/total_windows": total_windows,
                    "dataset/pseudo_len": int(opt.pseudo_dataset_len),
                    "loader/num_batches_drop_last": len(loader),
                    "train/batch_size": int(batch_size),
                }, step=global_step)
            except Exception:
                pass

        nan_skipped = 0
        for batch in iterator:
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
                nan_skipped += 1
                continue
            optimizer.zero_grad()
            total_loss.backward()
            # Grad clip (LoRA params only)
            if grad_clip and grad_clip > 0:
                try:
                    torch.nn.utils.clip_grad_norm_(list(lora_parameters(diffusion.model)), max_norm=grad_clip)
                except Exception:
                    pass
            optimizer.step()
            step_loss = float(total_loss.detach().cpu().item())
            running += step_loss
            n += 1

            # Update progress bar postfix
            if _TQDM_AVAILABLE and use_bar:
                try:
                    iterator.set_postfix({
                        'loss': f"{step_loss:.4f}",
                        'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
                    })
                except Exception:
                    pass

            # Per-step logging (throttled)
            if use_wandb and (global_step % log_every == 0):
                log = {"train/step_loss": step_loss,
                       "train/epoch": epoch,
                       "optim/lr": float(optimizer.param_groups[0]['lr'])}
                # Log loss components if available
                if isinstance(losses, dict):
                    for k, v in losses.items():
                        try:
                            log[f"loss/{k}"] = float(v.detach().cpu().item())
                        except Exception:
                            pass
                # Grad norm (LoRA only)
                try:
                    g2 = 0.0
                    for p in lora_parameters(diffusion.model):
                        if p.grad is not None:
                            g2 += float(p.grad.detach().data.norm(2).item() ** 2)
                    log["optim/grad_norm"] = math.sqrt(g2)
                except Exception:
                    pass
                # GPU memory
                if device.type == 'cuda':
                    try:
                        log["sys/max_mem_mb"] = torch.cuda.max_memory_allocated(device) / (1024*1024)
                        log["sys/mem_mb"] = torch.cuda.memory_allocated(device) / (1024*1024)
                    except Exception:
                        pass
                wandb.log(log, step=global_step)
            global_step += 1

        avg = running / max(1, n)
        dt = time.time() - t0
        print(f"Epoch {epoch}/{epochs} - loss: {avg:.6f} ({n} steps, {dt:.1f}s)")
        if n == 0:
            print(f"[WARN] No training steps this epoch. pseudo_len={opt.pseudo_dataset_len}, batch_size={batch_size}, loader_len={len(loader) if hasattr(loader,'__len__') else 'NA'}, nan_skipped={nan_skipped}")
        if use_wandb:
            wandb.log({
                "train/epoch_loss": avg,
                "train/epoch_time_s": dt,
                "train/steps": n,
                "train/nan_skipped": int(nan_skipped),
                "epoch": epoch
            }, step=global_step)

        # Save every epoch
        save_lora_adapter(diffusion.model, str(save_dir / 'weights' / f'epoch-{epoch}-lora.pt'),
                          extra={"normalizer": normalizer})
        if use_wandb:
            try:
                wandb.save(str(save_dir / 'weights' / f'epoch-{epoch}-lora.pt'), base_path=str(save_dir))
            except Exception:
                pass

    print(f"[DONE] Saved LoRA adapters under: {save_dir}")
    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
