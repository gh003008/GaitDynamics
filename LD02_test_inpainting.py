"""
LD02: Inpainting Test

Tests the model's ability to reconstruct masked portions of gait data.
Different masking patterns test different aspects of the model's understanding.

Usage:
    python LD02_test_inpainting.py --exp_name LD_proper --mask_type mask_kinematics
    python LD02_test_inpainting.py --exp_name LD_proper --mask_type mask_kinetics
    python LD02_test_inpainting.py --exp_name LD_proper --mask_type mask_knee --num_samples 3
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import from existing scripts
sys.path.insert(0, os.path.dirname(__file__))
from args import parse_opt
from train_models_only_LD import LDGDPDataset, setup_ld_args
from model.model import MotionModel
from data.preprocess import Normalizer


# Channel indices for masking
CHANNEL_GROUPS = {
    'pelvis_pos': [0, 1, 2],  # pelvis_tx, ty, tz
    'knee': [3, 6],  # knee_angle_r, knee_angle_l
    'ankle': [4, 7],  # ankle_angle_r, ankle_angle_l
    'subtalar': [5, 8],  # subtalar_angle_r, subtalar_angle_l
    'kinematics': list(range(0, 9)),  # All joint angles and pelvis pos
    'kinetics': list(range(9, 21)),  # All GRF and COP (12 channels)
    '6d_poses': list(range(21, 45)),  # All 6D poses (24 channels)
}


def get_mask_pattern(mask_type: str, num_channels: int) -> np.ndarray:
    """
    Create masking pattern for inpainting.
    
    Args:
        mask_type: Type of masking ('mask_kinematics', 'mask_kinetics', 'mask_knee', etc.)
        num_channels: Total number of channels
        
    Returns:
        mask: Binary mask [C] where 1 = mask (inpaint), 0 = keep (condition)
    """
    mask = np.zeros(num_channels, dtype=np.float32)
    
    if mask_type == 'mask_kinematics':
        # Mask all kinematics → predict from kinetics
        mask[CHANNEL_GROUPS['kinematics']] = 1.0
        
    elif mask_type == 'mask_kinetics':
        # Mask all kinetics → predict from kinematics
        mask[CHANNEL_GROUPS['kinetics']] = 1.0
        
    elif mask_type == 'mask_knee':
        # Mask only knee angles
        mask[CHANNEL_GROUPS['knee']] = 1.0
        
    elif mask_type == 'mask_ankle':
        # Mask only ankle angles
        mask[CHANNEL_GROUPS['ankle']] = 1.0
        
    elif mask_type == 'mask_6d_poses':
        # Mask all 6D poses
        mask[CHANNEL_GROUPS['6d_poses']] = 1.0
        
    elif mask_type == 'mask_all_joints':
        # Mask all kinematics + 6D poses
        mask[CHANNEL_GROUPS['kinematics']] = 1.0
        mask[CHANNEL_GROUPS['6d_poses']] = 1.0
        
    else:
        raise ValueError(f"Unknown mask type: {mask_type}")
    
    return mask


def load_trained_model(exp_name: str, opt, normalizer):
    """Load trained model from checkpoint."""
    model_dir = Path(f"runs/train/{exp_name}/weights")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    checkpoints = list(model_dir.glob("*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found in {model_dir}")
    
    checkpoint_path = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]) if p.stem.split('_')[-1].isdigit() else 0)
    print(f"Loading checkpoint: {checkpoint_path}")
    
    model = MotionModel(opt)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.diffusion.master_model.load_state_dict(checkpoint['ema_state_dict'])
    model.normalizer = normalizer
    model.diffusion.set_normalizer(normalizer)
    model.eval()
    
    return model, checkpoint_path


def inpaint_trial(model, trial_data: torch.Tensor, mask_pattern: np.ndarray, 
                  normalizer: Normalizer, opt) -> torch.Tensor:
    """
    Inpaint masked portions of a trial using the diffusion model.
    
    Args:
        model: Trained MotionModel
        trial_data: [T, C] normalized trial data
        mask_pattern: [C] binary mask (1 = inpaint, 0 = keep)
        normalizer: Data normalizer
        opt: Options
        
    Returns:
        inpainted: [T, C] inpainted data (unnormalized)
    """
    device = next(model.diffusion.model.parameters()).device
    window_len = opt.window_len
    T, C = trial_data.shape
    
    # Convert mask to tensor
    mask_pattern_tensor = torch.from_numpy(mask_pattern).float()
    
    # Process trial in windows with overlap
    stride = window_len // 2
    inpainted = torch.zeros(T, C)
    counts = torch.zeros(T)
    
    for start_idx in range(0, T - window_len + 1, stride):
        end_idx = start_idx + window_len
        window = trial_data[start_idx:end_idx].unsqueeze(0).to(device)  # [1, window_len, C]
        
        # Create mask for this window [1, window_len, C]
        mask = mask_pattern_tensor.unsqueeze(0).unsqueeze(0).expand(1, window_len, C).to(device)
        
        # Inpaint
        with torch.no_grad():
            pred = model.eval_loop(
                opt,
                state_true=window,
                masks=mask,
                value_diff_thd=None,
                value_diff_weight=None,
                cond=torch.ones(6).to(device),
                num_of_generation_per_window=1,
                mode="inpaint"
            )  # [1, 1, window_len, C]
        
        pred = pred[0, 0].cpu()  # [window_len, C]
        
        # Accumulate
        inpainted[start_idx:end_idx] += pred
        counts[start_idx:end_idx] += 1
    
    # Handle last window
    if T > window_len:
        last_start = T - window_len
        last_window = trial_data[last_start:].unsqueeze(0).to(device)
        mask = mask_pattern_tensor.unsqueeze(0).unsqueeze(0).expand(1, window_len, C).to(device)
        
        with torch.no_grad():
            pred = model.eval_loop(
                opt,
                state_true=last_window,
                masks=mask,
                value_diff_thd=None,
                value_diff_weight=None,
                cond=torch.ones(6).to(device),
                num_of_generation_per_window=1,
                mode="inpaint"
            )
        
        pred = pred[0, 0].cpu()
        inpainted[last_start:] += pred
        counts[last_start:] += 1
    
    # Average overlapping predictions
    inpainted = inpainted / counts.unsqueeze(1).clamp(min=1)
    
    # NOTE: eval_loop already returns unnormalized data (via generate_samples)
    # So we don't need to unnormalize again here
    
    return inpainted


def compute_inpainting_metrics(true_data: np.ndarray, inpainted_data: np.ndarray, 
                                mask_pattern: np.ndarray, channel_names: List[str]) -> Dict:
    """Compute metrics only for masked (inpainted) channels."""
    metrics = {}
    
    # Get masked channel indices
    masked_indices = np.where(mask_pattern == 1.0)[0]
    
    # Overall metrics (only for masked channels)
    masked_true = true_data[:, masked_indices]
    masked_inpainted = inpainted_data[:, masked_indices]
    
    mse = np.mean((masked_true - masked_inpainted) ** 2)
    mae = np.mean(np.abs(masked_true - masked_inpainted))
    rmse = np.sqrt(mse)
    
    # Correlation (handle edge cases)
    try:
        corr = np.corrcoef(masked_true.flatten(), masked_inpainted.flatten())[0, 1]
    except:
        corr = float('nan')
    
    metrics['overall'] = {
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(rmse),
        'correlation': float(corr),
        'num_masked_channels': len(masked_indices)
    }
    
    # Per-channel metrics (only for masked channels)
    metrics['per_channel'] = {}
    for idx in masked_indices:
        ch_name = channel_names[idx]
        ch_true = true_data[:, idx]
        ch_inpainted = inpainted_data[:, idx]
        
        ch_mse = np.mean((ch_true - ch_inpainted) ** 2)
        ch_mae = np.mean(np.abs(ch_true - ch_inpainted))
        
        try:
            ch_corr = np.corrcoef(ch_true, ch_inpainted)[0, 1]
        except:
            ch_corr = float('nan')
        
        metrics['per_channel'][ch_name] = {
            'mse': float(ch_mse),
            'mae': float(ch_mae),
            'correlation': float(ch_corr)
        }
    
    return metrics


def plot_inpainting_comparison(true_data: np.ndarray, inpainted_data: np.ndarray,
                                mask_pattern: np.ndarray, channel_names: List[str],
                                trial_name: str, output_path: Path):
    """Plot comparison of true vs inpainted data for masked channels."""
    masked_indices = np.where(mask_pattern == 1.0)[0]
    num_masked = len(masked_indices)
    
    if num_masked == 0:
        print("No masked channels to plot")
        return
    
    # Plot up to 12 channels
    num_to_plot = min(12, num_masked)
    indices_to_plot = masked_indices[:num_to_plot]
    
    fig, axes = plt.subplots(num_to_plot, 1, figsize=(12, 2 * num_to_plot))
    if num_to_plot == 1:
        axes = [axes]
    
    for i, idx in enumerate(indices_to_plot):
        ax = axes[i]
        ax.plot(true_data[:, idx], 'b-', label='True', alpha=0.7, linewidth=1.5)
        ax.plot(inpainted_data[:, idx], 'r--', label='Inpainted', alpha=0.7, linewidth=1.5)
        ax.set_ylabel(channel_names[idx], fontsize=10)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc='upper right')
        if i == num_to_plot - 1:
            ax.set_xlabel('Frame')
    
    plt.suptitle(f'Inpainting Results: {trial_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved comparison plot: {output_path}")


def test_inpainting(exp_name: str, mask_type: str, num_samples: int = None):
    """Main testing function for inpainting."""
    # Setup
    opt = parse_opt()
    opt = setup_ld_args(opt)
    opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 60)
    print(f"LD02: Inpainting Test - {exp_name}")
    print(f"Mask type: {mask_type}")
    print("=" * 60)
    print()
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = LDGDPDataset(
        data_path=opt.data_path_train,
        train=True,
        opt=opt
    )
    normalizer = train_dataset.normalizer
    
    test_dataset = LDGDPDataset(
        data_path=opt.data_path_test,
        train=False,
        opt=opt,
        normalizer=normalizer
    )
    
    print(f"Test set: {len(test_dataset.trials)} trials")
    
    # Load model
    print("\nLoading trained model...")
    model, checkpoint_path = load_trained_model(exp_name, opt, normalizer)
    
    # Get mask pattern
    num_channels = len(opt.model_states_column_names)
    mask_pattern = get_mask_pattern(mask_type, num_channels)
    num_masked = int(mask_pattern.sum())
    print(f"Masking {num_masked}/{num_channels} channels")
    print(f"Masked channels: {[opt.model_states_column_names[i] for i, m in enumerate(mask_pattern) if m == 1.0]}")
    
    # Create output directory
    output_dir = Path(f"results/{exp_name}/2_inpainting/{mask_type}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model info
    model_info = {
        'checkpoint_path': str(checkpoint_path),
        'mask_type': mask_type,
        'num_masked_channels': num_masked,
        'masked_channels': [opt.model_states_column_names[i] for i, m in enumerate(mask_pattern) if m == 1.0],
        'test_trials': len(test_dataset.trials),
        'model_params': sum(p.numel() for p in model.diffusion.model.parameters()),
    }
    with open(output_dir / 'model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    # Test trials
    trials_to_test = test_dataset.trials[:num_samples] if num_samples else test_dataset.trials
    print(f"\nTesting {len(trials_to_test)} trials...")
    
    all_metrics = []
    
    for trial in tqdm(trials_to_test, desc="Inpainting"):
        trial_name = trial.sub_and_trial_name
        trial_dir = output_dir / trial_name
        trial_dir.mkdir(exist_ok=True)
        
        # Get true data (unnormalized)
        true_data = normalizer.unnormalize(trial.converted_pose.unsqueeze(0)).squeeze(0).cpu().numpy()
        
        # Inpaint
        inpainted_data = inpaint_trial(model, trial.converted_pose, mask_pattern, normalizer, opt).cpu().numpy()
        
        # Compute metrics (only for masked channels)
        metrics = compute_inpainting_metrics(true_data, inpainted_data, mask_pattern, opt.model_states_column_names)
        all_metrics.append({'trial': trial_name, **metrics['overall']})
        
        # Save results
        np.savez_compressed(trial_dir / 'true.npz', data=true_data)
        np.savez_compressed(trial_dir / 'inpainted.npz', data=inpainted_data)
        np.savez_compressed(trial_dir / 'mask.npz', mask=mask_pattern)
        
        with open(trial_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Plot
        plot_inpainting_comparison(true_data, inpainted_data, mask_pattern, 
                                   opt.model_states_column_names, trial_name,
                                   trial_dir / 'comparison.png')
    
    # Summary
    avg_mse = np.mean([m['mse'] for m in all_metrics])
    avg_mae = np.mean([m['mae'] for m in all_metrics])
    avg_rmse = np.mean([m['rmse'] for m in all_metrics])
    avg_corr = np.nanmean([m['correlation'] for m in all_metrics])
    
    summary = {
        'num_trials': len(all_metrics),
        'mask_type': mask_type,
        'num_masked_channels': num_masked,
        'avg_mse': float(avg_mse),
        'avg_mae': float(avg_mae),
        'avg_rmse': float(avg_rmse),
        'avg_correlation': float(avg_corr),
        'per_trial': all_metrics
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("INPAINTING RESULTS SUMMARY")
    print("=" * 60)
    print(f"Experiment: {exp_name}")
    print(f"Mask type: {mask_type}")
    print(f"Masked channels: {num_masked}/{num_channels}")
    print(f"Trials tested: {len(all_metrics)}")
    print(f"Average MSE (masked channels): {avg_mse:.6f}")
    print(f"Average MAE (masked channels): {avg_mae:.6f}")
    print(f"Average RMSE (masked channels): {avg_rmse:.6f}")
    print(f"Average Correlation: {avg_corr:.6f}")
    print(f"\nResults saved to: {output_dir}")
    print("=" * 60)


def main():
    # Custom argument parser to avoid conflict with args.py
    parser = argparse.ArgumentParser(description='LD02: Inpainting Test', add_help=False)
    parser.add_argument('--exp_name', type=str, required=True, help='Experiment name')
    parser.add_argument('--mask_type', type=str, required=True,
                       choices=['mask_kinematics', 'mask_kinetics', 'mask_knee', 
                               'mask_ankle', 'mask_6d_poses', 'mask_all_joints'],
                       help='Type of masking pattern')
    parser.add_argument('--num_samples', type=int, default=None, 
                       help='Number of trials to test (default: all)')
    
    our_args, remaining = parser.parse_known_args()
    
    # Remove our arguments from sys.argv so parse_opt() doesn't see them
    sys.argv = [sys.argv[0]] + remaining
    
    test_inpainting(our_args.exp_name, our_args.mask_type, our_args.num_samples)


if __name__ == "__main__":
    main()
