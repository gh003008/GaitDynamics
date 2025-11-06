#!/usr/bin/env python3
"""
LD01: Test Set Reconstruction
Test trained LD model on S011 test set and evaluate reconstruction performance.

Usage:
    python LD01_test_reconstruction.py --exp_name LD_proper
    python LD01_test_reconstruction.py --exp_name LD_proper --num_samples 3

Results saved to: results/{exp_name}/1_reconstruction/
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from args import parse_opt
from model.model import MotionModel
from data.preprocess import Normalizer
from train_models_only_LD import LDGDPDataset, setup_ld_args
from consts import *


def load_trained_model(exp_name: str, opt, normalizer):
    """Load trained model from checkpoint."""
    # Find model checkpoint
    model_dir = Path(f"runs/train/{exp_name}/weights")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Find latest checkpoint
    checkpoints = list(model_dir.glob("*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found in {model_dir}")
    
    # Use the one with highest epoch number
    checkpoint_path = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]) if p.stem.split('_')[-1].isdigit() else 0)
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Create model
    model = MotionModel(opt)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load EMA model (better quality than raw model)
    model.diffusion.master_model.load_state_dict(checkpoint['ema_state_dict'])
    
    # Set normalizer manually for inference
    model.normalizer = normalizer
    model.diffusion.set_normalizer(normalizer)
    
    model.eval()
    
    return model, checkpoint_path


def reconstruct_trial(model, trial_data: torch.Tensor, normalizer: Normalizer, opt) -> torch.Tensor:
    """Reconstruct a single trial using the model.
    
    Args:
        model: Trained MotionModel
        trial_data: [T, C] normalized trial data
        normalizer: Data normalizer
        opt: Options
        
    Returns:
        reconstructed: [T, C] reconstructed data (unnormalized)
    """
    device = next(model.diffusion.model.parameters()).device
    window_len = opt.window_len
    T, C = trial_data.shape
    
    # Process trial in windows with overlap
    stride = window_len // 2  # 50% overlap
    reconstructed = torch.zeros(T, C)
    counts = torch.zeros(T)
    
    for start_idx in range(0, T - window_len + 1, stride):
        end_idx = start_idx + window_len
        window = trial_data[start_idx:end_idx].unsqueeze(0).to(device)  # [1, window_len, C]
        
        # Prepare mask for reconstruction (no masking, full reconstruction)
        mask = torch.zeros(1, window_len, C).to(device)  # All zeros = reconstruct all
        
        # Generate reconstruction
        with torch.no_grad():
            pred = model.eval_loop(
                opt,
                state_true=window,
                masks=mask,
                value_diff_thd=None,
                value_diff_weight=None,
                cond=torch.ones(6).to(device),  # Pass tensor explicitly
                num_of_generation_per_window=1,
                mode="inpaint"
            )  # [1, 1, window_len, C]
        
        pred = pred[0, 0].cpu()  # [window_len, C]
        
        # Accumulate with overlap handling
        reconstructed[start_idx:end_idx] += pred
        counts[start_idx:end_idx] += 1
    
    # Handle last window if needed
    if T > window_len:
        last_start = T - window_len
        last_window = trial_data[last_start:].unsqueeze(0).to(device)
        mask = torch.zeros(1, window_len, C).to(device)
        
        with torch.no_grad():
            pred = model.eval_loop(
                opt,
                state_true=last_window,
                masks=mask,
                value_diff_thd=None,
                value_diff_weight=None,
                cond=torch.ones(6).to(device),  # Pass tensor explicitly
                num_of_generation_per_window=1,
                mode="inpaint"
            )
        
        pred = pred[0, 0].cpu()
        reconstructed[last_start:] += pred
        counts[last_start:] += 1
    
    # Average overlapping predictions
    reconstructed = reconstructed / counts.unsqueeze(1).clamp(min=1)
    
    # Unnormalize - add batch dimension
    reconstructed = normalizer.unnormalize(reconstructed.unsqueeze(0)).squeeze(0)
    
    return reconstructed


def compute_metrics(true: np.ndarray, pred: np.ndarray, column_names: List[str]) -> Dict:
    """Compute reconstruction metrics.
    
    Args:
        true: [T, C] ground truth
        pred: [T, C] predictions
        column_names: List of column names
        
    Returns:
        metrics: Dictionary of metrics
    """
    mse = np.mean((true - pred) ** 2)
    mae = np.mean(np.abs(true - pred))
    rmse = np.sqrt(mse)
    
    # Per-channel metrics
    channel_mse = np.mean((true - pred) ** 2, axis=0)
    channel_mae = np.mean(np.abs(true - pred), axis=0)
    
    # Correlation
    correlations = []
    for i in range(true.shape[1]):
        corr = np.corrcoef(true[:, i], pred[:, i])[0, 1]
        correlations.append(corr)
    
    avg_corr = np.mean(correlations)
    
    metrics = {
        'overall': {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'correlation': float(avg_corr)
        },
        'per_channel': {
            name: {
                'mse': float(channel_mse[i]),
                'mae': float(channel_mae[i]),
                'correlation': float(correlations[i])
            }
            for i, name in enumerate(column_names)
        }
    }
    
    return metrics


def plot_comparison(true: np.ndarray, pred: np.ndarray, column_names: List[str], 
                   save_path: Path, max_channels: int = 12):
    """Plot comparison between true and predicted data.
    
    Args:
        true: [T, C] ground truth
        pred: [T, C] predictions
        column_names: List of column names
        save_path: Path to save plot
        max_channels: Maximum number of channels to plot
    """
    n_channels = min(len(column_names), max_channels)
    fig, axes = plt.subplots(n_channels, 1, figsize=(12, 2 * n_channels))
    if n_channels == 1:
        axes = [axes]
    
    for i in range(n_channels):
        ax = axes[i]
        t = np.arange(true.shape[0]) / 100.0  # Assume 100 Hz
        
        ax.plot(t, true[:, i], 'b-', label='True', alpha=0.7, linewidth=1)
        ax.plot(t, pred[:, i], 'r--', label='Pred', alpha=0.7, linewidth=1)
        
        ax.set_ylabel(column_names[i], fontsize=8)
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend(loc='upper right')
        if i == n_channels - 1:
            ax.set_xlabel('Time (s)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved comparison plot: {save_path}")


def test_reconstruction(exp_name: str, num_samples: int = None):
    """Run reconstruction test on S011 test set.
    
    Args:
        exp_name: Experiment name (e.g., 'LD_proper')
        num_samples: Number of trials to test (None = all)
    """
    print("="*60)
    print(f"LD01: Test Set Reconstruction - {exp_name}")
    print("="*60)
    
    # Setup
    opt = parse_opt()
    opt = setup_ld_args(opt)
    
    # Load test dataset
    print("\nLoading test dataset (S011)...")
    
    # Load training dataset first to get normalizer
    print("Loading training dataset for normalizer...")
    train_dataset = LDGDPDataset(
        data_path=opt.data_path_train,
        train=True,
        opt=opt
    )
    normalizer = train_dataset.normalizer
    
    # Now load test dataset with the normalizer
    test_dataset = LDGDPDataset(
        data_path=opt.data_path_test,
        train=False,
        opt=opt,
        normalizer=normalizer  # Pass the normalizer
    )
    
    print(f"Test set: {len(test_dataset.trials)} trials")
    
    # Load model with normalizer
    print("\nLoading trained model...")
    model, checkpoint_path = load_trained_model(exp_name, opt, normalizer)
    
    # Create output directory
    output_dir = Path(f"results/{exp_name}/1_reconstruction")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model info
    model_info = {
        'checkpoint_path': str(checkpoint_path),
        'test_trials': len(test_dataset.trials),
        'model_params': sum(p.numel() for p in model.diffusion.model.parameters()),
    }
    with open(output_dir / 'model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    # Determine trials to test
    trials_to_test = test_dataset.trials[:num_samples] if num_samples else test_dataset.trials
    
    print(f"\nTesting {len(trials_to_test)} trials...")
    all_metrics = []
    
    # Test each trial
    for trial in tqdm(trials_to_test, desc="Reconstructing"):
        trial_name = trial.sub_and_trial_name
        trial_dir = output_dir / trial_name
        trial_dir.mkdir(exist_ok=True)
        
        # Get ground truth (unnormalized)
        # Add batch dimension for normalizer
        true_data = normalizer.unnormalize(trial.converted_pose.unsqueeze(0)).squeeze(0).cpu().numpy()
        
        # Reconstruct
        pred_data = reconstruct_trial(model, trial.converted_pose, normalizer, opt).cpu().numpy()
        
        # Compute metrics
        metrics = compute_metrics(true_data, pred_data, opt.model_states_column_names)
        all_metrics.append(metrics)
        
        # Save results
        np.savez_compressed(
            trial_dir / 'true.npz',
            data=true_data,
            columns=opt.model_states_column_names
        )
        np.savez_compressed(
            trial_dir / 'pred.npz',
            data=pred_data,
            columns=opt.model_states_column_names
        )
        
        with open(trial_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Plot comparison
        plot_comparison(true_data, pred_data, opt.model_states_column_names, 
                       trial_dir / 'comparison.png')
    
    # Compute summary statistics
    summary = {
        'num_trials': len(trials_to_test),
        'avg_mse': np.mean([m['overall']['mse'] for m in all_metrics]),
        'avg_mae': np.mean([m['overall']['mae'] for m in all_metrics]),
        'avg_rmse': np.mean([m['overall']['rmse'] for m in all_metrics]),
        'avg_correlation': np.mean([m['overall']['correlation'] for m in all_metrics]),
        'per_trial': [
            {
                'trial': trials_to_test[i].sub_and_trial_name,
                'mse': all_metrics[i]['overall']['mse'],
                'mae': all_metrics[i]['overall']['mae'],
                'correlation': all_metrics[i]['overall']['correlation']
            }
            for i in range(len(trials_to_test))
        ]
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("RECONSTRUCTION RESULTS SUMMARY")
    print("="*60)
    print(f"Experiment: {exp_name}")
    print(f"Trials tested: {summary['num_trials']}")
    print(f"Average MSE: {summary['avg_mse']:.6f}")
    print(f"Average MAE: {summary['avg_mae']:.6f}")
    print(f"Average RMSE: {summary['avg_rmse']:.6f}")
    print(f"Average Correlation: {summary['avg_correlation']:.4f}")
    print(f"\nResults saved to: {output_dir}")
    print("="*60)


def main():
    # Parse our arguments first (before parse_opt() interferes)
    parser = argparse.ArgumentParser(description='LD01: Test Set Reconstruction', add_help=False)
    parser.add_argument('--exp_name', type=str, default='LD_proper',
                       help='Experiment name (default: LD_proper)')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of trials to test (default: all)')
    
    our_args, remaining = parser.parse_known_args()
    
    # Remove our custom arguments from sys.argv so parse_opt() doesn't see them
    sys.argv = [sys.argv[0]] + remaining
    
    test_reconstruction(our_args.exp_name, our_args.num_samples)


if __name__ == '__main__':
    main()
