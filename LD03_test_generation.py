"""
LD03: Unconditional Generation Test

Tests the model's ability to generate realistic gait sequences from random noise.
Evaluates diversity and quality of generated samples.

Usage:
    python LD03_test_generation.py --exp_name LD_proper --num_samples 10
    python LD03_test_generation.py --exp_name LD_proper --num_samples 20 --seq_len 300
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Tuple
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


def generate_sample(model, opt, seq_len: int = 150) -> torch.Tensor:
    """
    Generate a single gait sequence from random noise.
    
    Args:
        model: Trained MotionModel
        opt: Options
        seq_len: Sequence length (default: 150, same as training)
        
    Returns:
        generated: [seq_len, C] unnormalized generated data
    """
    device = next(model.diffusion.model.parameters()).device
    num_channels = len(opt.model_states_column_names)
    
    # Create dummy input (will be ignored)
    dummy_input = torch.zeros(1, seq_len, num_channels).to(device)
    
    # Create empty mask (no conditioning) for unconditional generation
    empty_mask = torch.zeros(1, seq_len, num_channels).to(device)
    
    # Generate from noise using inpaint mode with empty mask
    with torch.no_grad():
        generated = model.eval_loop(
            opt,
            state_true=dummy_input,
            masks=empty_mask,  # Empty mask = unconditional generation
            value_diff_thd=None,
            value_diff_weight=None,
            cond=torch.ones(6).to(device),  # Default condition
            num_of_generation_per_window=1,
            mode="inpaint"
        )  # [1, 1, seq_len, C]
    
    generated = generated[0, 0].cpu()  # [seq_len, C]
    
    # NOTE: eval_loop already returns unnormalized data (via generate_samples)
    # So we don't need to unnormalize again here
    
    return generated


def compute_diversity_metrics(samples: List[np.ndarray]) -> dict:
    """
    Compute diversity metrics across generated samples.
    
    Args:
        samples: List of [T, C] arrays
        
    Returns:
        metrics: Dictionary of diversity metrics
    """
    # Stack samples [N, T, C]
    samples_array = np.stack(samples, axis=0)
    N, T, C = samples_array.shape
    
    # Per-frame variance across samples (measures diversity)
    frame_variance = np.var(samples_array, axis=0)  # [T, C]
    avg_variance = np.mean(frame_variance, axis=0)  # [C]
    
    # Pairwise distances between samples
    pairwise_distances = []
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.sqrt(np.mean((samples_array[i] - samples_array[j]) ** 2))
            pairwise_distances.append(dist)
    
    avg_pairwise_distance = np.mean(pairwise_distances) if pairwise_distances else 0.0
    
    # Per-channel statistics across samples
    channel_means = np.mean(samples_array, axis=(0, 1))  # [C]
    channel_stds = np.std(samples_array, axis=(0, 1))  # [C]
    
    metrics = {
        'num_samples': N,
        'avg_pairwise_distance': float(avg_pairwise_distance),
        'avg_variance_per_channel': {
            'mean': float(np.mean(avg_variance)),
            'std': float(np.std(avg_variance)),
            'min': float(np.min(avg_variance)),
            'max': float(np.max(avg_variance))
        },
        'channel_statistics': {
            'mean': channel_means.tolist(),
            'std': channel_stds.tolist()
        }
    }
    
    return metrics


def check_realism(sample: np.ndarray, channel_names: List[str]) -> dict:
    """
    Simple realism checks for generated samples.
    
    Checks:
    - Range validity (no extreme outliers)
    - Smoothness (no sudden jumps)
    - Periodicity (rough gait cycle detection)
    """
    T, C = sample.shape
    
    # Range check (using z-score)
    z_scores = np.abs((sample - np.mean(sample, axis=0)) / (np.std(sample, axis=0) + 1e-8))
    max_z_score = np.max(z_scores)
    outlier_ratio = np.mean(z_scores > 5.0)  # More than 5 std deviations
    
    # Smoothness check (average difference between consecutive frames)
    diffs = np.diff(sample, axis=0)
    avg_diff = np.mean(np.abs(diffs), axis=0)  # [C]
    max_jump = np.max(np.abs(diffs), axis=0)  # [C]
    
    # Periodicity check (for knee angle - index 3 or 6)
    knee_idx = 3  # knee_angle_r
    if C > knee_idx:
        knee_signal = sample[:, knee_idx]
        # Simple autocorrelation at expected gait cycle length (~100-120 frames for 1 Hz gait)
        autocorr_lags = [50, 75, 100, 125, 150]
        autocorrs = []
        for lag in autocorr_lags:
            if T > lag:
                autocorr = np.corrcoef(knee_signal[:-lag], knee_signal[lag:])[0, 1]
                autocorrs.append(autocorr)
        max_autocorr = max(autocorrs) if autocorrs else 0.0
    else:
        max_autocorr = 0.0
    
    metrics = {
        'max_z_score': float(max_z_score),
        'outlier_ratio': float(outlier_ratio),
        'avg_smoothness': {
            'mean': float(np.mean(avg_diff)),
            'max': float(np.max(avg_diff))
        },
        'max_jump': {
            'mean': float(np.mean(max_jump)),
            'max': float(np.max(max_jump))
        },
        'periodicity_score': float(max_autocorr)
    }
    
    return metrics


def plot_generated_sample(sample: np.ndarray, channel_names: List[str], 
                          sample_id: int, output_path: Path):
    """Plot generated sample."""
    T, C = sample.shape
    
    # Plot up to 12 channels
    num_to_plot = min(12, C)
    
    fig, axes = plt.subplots(num_to_plot, 1, figsize=(12, 2 * num_to_plot))
    if num_to_plot == 1:
        axes = [axes]
    
    for i in range(num_to_plot):
        ax = axes[i]
        ax.plot(sample[:, i], 'b-', linewidth=1.5)
        ax.set_ylabel(channel_names[i], fontsize=10)
        ax.grid(True, alpha=0.3)
        if i == num_to_plot - 1:
            ax.set_xlabel('Frame')
    
    plt.suptitle(f'Generated Sample {sample_id}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_diversity_comparison(samples: List[np.ndarray], channel_names: List[str],
                               output_path: Path):
    """Plot multiple samples overlaid to visualize diversity."""
    num_samples = len(samples)
    C = samples[0].shape[1]
    
    # Plot up to 6 channels
    num_to_plot = min(6, C)
    
    fig, axes = plt.subplots(num_to_plot, 1, figsize=(12, 2 * num_to_plot))
    if num_to_plot == 1:
        axes = [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, num_samples))
    
    for i in range(num_to_plot):
        ax = axes[i]
        for j, sample in enumerate(samples):
            ax.plot(sample[:, i], color=colors[j], alpha=0.6, linewidth=1.0)
        ax.set_ylabel(channel_names[i], fontsize=10)
        ax.grid(True, alpha=0.3)
        if i == num_to_plot - 1:
            ax.set_xlabel('Frame')
    
    plt.suptitle(f'Diversity Visualization ({num_samples} samples)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def test_generation(exp_name: str, num_samples: int = 10, seq_len: int = 150):
    """Main testing function for generation."""
    # Setup
    opt = parse_opt()
    opt = setup_ld_args(opt)
    opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 60)
    print(f"LD03: Generation Test - {exp_name}")
    print(f"Generating {num_samples} samples of length {seq_len}")
    print("=" * 60)
    print()
    
    # Load dataset (just for normalizer)
    print("Loading dataset for normalizer...")
    train_dataset = LDGDPDataset(
        data_path=opt.data_path_train,
        train=True,
        opt=opt
    )
    normalizer = train_dataset.normalizer
    
    # Load model
    print("\nLoading trained model...")
    model, checkpoint_path = load_trained_model(exp_name, opt, normalizer)
    
    # Create output directory
    output_dir = Path(f"results/{exp_name}/3_generation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model info
    model_info = {
        'checkpoint_path': str(checkpoint_path),
        'num_samples': num_samples,
        'seq_len': seq_len,
        'model_params': sum(p.numel() for p in model.diffusion.model.parameters()),
    }
    with open(output_dir / 'model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    # Generate samples
    print(f"\nGenerating {num_samples} samples...")
    generated_samples = []
    all_realism_metrics = []
    
    for i in tqdm(range(num_samples), desc="Generating"):
        sample_dir = output_dir / f"sample_{i:03d}"
        sample_dir.mkdir(exist_ok=True)
        
        # Generate
        generated = generate_sample(model, opt, seq_len).cpu().numpy()
        generated_samples.append(generated)
        
        # Check realism
        realism_metrics = check_realism(generated, opt.model_states_column_names)
        all_realism_metrics.append({'sample_id': i, **realism_metrics})
        
        # Save
        np.savez_compressed(sample_dir / 'generated.npz', data=generated)
        
        with open(sample_dir / 'realism_metrics.json', 'w') as f:
            json.dump(realism_metrics, f, indent=2)
        
        # Plot
        plot_generated_sample(generated, opt.model_states_column_names, i,
                            sample_dir / 'plot.png')
    
    # Compute diversity metrics
    print("\nComputing diversity metrics...")
    diversity_metrics = compute_diversity_metrics(generated_samples)
    
    # Plot diversity
    plot_diversity_comparison(generated_samples, opt.model_states_column_names,
                             output_dir / 'diversity_comparison.png')
    
    # Summary
    avg_pairwise_dist = diversity_metrics['avg_pairwise_distance']
    avg_outlier_ratio = np.mean([m['outlier_ratio'] for m in all_realism_metrics])
    avg_periodicity = np.mean([m['periodicity_score'] for m in all_realism_metrics])
    
    summary = {
        'num_samples': num_samples,
        'seq_len': seq_len,
        'diversity_metrics': diversity_metrics,
        'avg_realism_metrics': {
            'avg_outlier_ratio': float(avg_outlier_ratio),
            'avg_periodicity_score': float(avg_periodicity)
        },
        'per_sample_realism': all_realism_metrics
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("GENERATION RESULTS SUMMARY")
    print("=" * 60)
    print(f"Experiment: {exp_name}")
    print(f"Samples generated: {num_samples}")
    print(f"Sequence length: {seq_len}")
    print(f"\nDiversity:")
    print(f"  Average pairwise distance: {avg_pairwise_dist:.4f}")
    print(f"  Variance (mean): {diversity_metrics['avg_variance_per_channel']['mean']:.4f}")
    print(f"\nRealism:")
    print(f"  Average outlier ratio: {avg_outlier_ratio:.4f}")
    print(f"  Average periodicity score: {avg_periodicity:.4f}")
    print(f"\nResults saved to: {output_dir}")
    print("=" * 60)


def main():
    # Custom argument parser
    parser = argparse.ArgumentParser(description='LD03: Generation Test', add_help=False)
    parser.add_argument('--exp_name', type=str, required=True, help='Experiment name')
    parser.add_argument('--num_samples', type=int, default=10, 
                       help='Number of samples to generate')
    parser.add_argument('--seq_len', type=int, default=150,
                       help='Sequence length (frames)')
    
    our_args, remaining = parser.parse_known_args()
    
    # Remove our arguments from sys.argv
    sys.argv = [sys.argv[0]] + remaining
    
    test_generation(our_args.exp_name, our_args.num_samples, our_args.seq_len)


if __name__ == "__main__":
    main()
