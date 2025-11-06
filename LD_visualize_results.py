"""
LD Results Visualization & Validation

Visualize and validate LD02/LD03 results:
1. Convert .npz to .mot files for OpenSim visualization
2. Create detailed plots (joint angles, GRF, per gait cycle)
3. Validate plausibility (ROM, symmetry, periodicity)

Usage:
    # Visualize LD02 inpainting results
    python LD_visualize_results.py --result_dir results/LD_proper/2_inpainting/mask_kinematics/S011_accel_sine_trial_01
    
    # Visualize LD03 generation results
    python LD_visualize_results.py --result_dir results/LD_proper/3_generation/sample_000
    
    # Batch process all generation samples
    python LD_visualize_results.py --result_dir results/LD_proper/3_generation --batch
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from consts import MODEL_STATES_COLUMN_NAMES_NO_ARM, OSIM_DOF_ALL


def npz_to_mot(data: np.ndarray, channel_names: List[str], output_path: Path, fps: int = 100):
    """
    Convert npz data to OpenSim .mot file.
    
    Args:
        data: [T, C] array
        channel_names: List of channel names (must match MODEL_STATES_COLUMN_NAMES_NO_ARM)
        output_path: Output .mot file path
        fps: Sampling rate (Hz)
    """
    T, C = data.shape
    
    # Create time column
    time = np.arange(T) / fps
    
    # Map LD channels to OSIM coordinates
    # LD has: pelvis_tx/ty/tz, knee/ankle/subtalar angles (6), GRF/COP (12), 6DOF (30+)
    # OSIM needs: pelvis translations/rotations, hip/knee/ankle/subtalar angles
    
    # Extract joint angles from LD data
    pelvis_tx_idx = channel_names.index('pelvis_tx')
    pelvis_ty_idx = channel_names.index('pelvis_ty')
    pelvis_tz_idx = channel_names.index('pelvis_tz')
    
    knee_r_idx = channel_names.index('knee_angle_r')
    ankle_r_idx = channel_names.index('ankle_angle_r')
    subtalar_r_idx = channel_names.index('subtalar_angle_r')
    knee_l_idx = channel_names.index('knee_angle_l')
    ankle_l_idx = channel_names.index('ankle_angle_l')
    subtalar_l_idx = channel_names.index('subtalar_angle_l')
    
    # Create DataFrame with OSIM coordinate names
    # For minimal working .mot, we need pelvis + lower limb
    mot_data = {
        'time': time,
        'pelvis_tx': data[:, pelvis_tx_idx],
        'pelvis_ty': data[:, pelvis_ty_idx],
        'pelvis_tz': data[:, pelvis_tz_idx],
        'pelvis_tilt': np.zeros(T),  # LD doesn't have pelvis rotations
        'pelvis_list': np.zeros(T),
        'pelvis_rotation': np.zeros(T),
        'hip_flexion_r': np.zeros(T),  # LD doesn't have hip angles directly
        'hip_adduction_r': np.zeros(T),
        'hip_rotation_r': np.zeros(T),
        'knee_angle_r': data[:, knee_r_idx],
        'ankle_angle_r': data[:, ankle_r_idx],
        'subtalar_angle_r': data[:, subtalar_r_idx],
        'hip_flexion_l': np.zeros(T),
        'hip_adduction_l': np.zeros(T),
        'hip_rotation_l': np.zeros(T),
        'knee_angle_l': data[:, knee_l_idx],
        'ankle_angle_l': data[:, ankle_l_idx],
        'subtalar_angle_l': data[:, subtalar_l_idx],
    }
    
    df = pd.DataFrame(mot_data)
    
    # Write .mot file
    with open(output_path, 'w') as f:
        f.write('Coordinates\n')
        f.write('version=1\n')
        f.write(f'nRows={T}\n')
        f.write(f'nColumns={len(mot_data)}\n')
        f.write('inDegrees=no\n')
        f.write('\n')
        f.write("If the header above contains a line with 'inDegrees', this indicates whether rotational values are in degrees (yes) or radians (no).\n")
        f.write('\n')
        f.write('endheader\n')
        
        # Write header
        f.write('\t'.join(df.columns) + '\n')
        
        # Write data
        for i in range(T):
            row = [f"{df.iloc[i, j]:.8f}" for j in range(len(df.columns))]
            f.write('\t'.join(row) + '\n')
    
    print(f"  Saved .mot file: {output_path}")


def plot_detailed_comparison(true_data: np.ndarray, pred_data: np.ndarray, 
                             channel_names: List[str], output_dir: Path, 
                             trial_name: str = "result"):
    """
    Create detailed plots for better visualization.
    """
    T, C = true_data.shape
    time = np.arange(T) / 100  # 100 Hz
    
    # 1. Joint Angles Plot
    joint_indices = {
        'Knee (R)': channel_names.index('knee_angle_r'),
        'Ankle (R)': channel_names.index('ankle_angle_r'),
        'Subtalar (R)': channel_names.index('subtalar_angle_r'),
        'Knee (L)': channel_names.index('knee_angle_l'),
        'Ankle (L)': channel_names.index('ankle_angle_l'),
        'Subtalar (L)': channel_names.index('subtalar_angle_l'),
    }
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, (joint_name, idx) in enumerate(joint_indices.items()):
        ax = axes[i]
        ax.plot(time, np.rad2deg(true_data[:, idx]), 'b-', label='True', linewidth=2, alpha=0.8)
        ax.plot(time, np.rad2deg(pred_data[:, idx]), 'r--', label='Predicted', linewidth=2, alpha=0.8)
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Angle (deg)', fontsize=11)
        ax.set_title(joint_name, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)
        
        # Add error shading
        error = np.rad2deg(np.abs(true_data[:, idx] - pred_data[:, idx]))
        ax2 = ax.twinx()
        ax2.fill_between(time, 0, error, alpha=0.2, color='orange', label='Error')
        ax2.set_ylabel('Error (deg)', fontsize=9, color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        ax2.set_ylim(0, max(10, np.max(error)))
    
    plt.suptitle(f'Joint Angles: {trial_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'joints.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # 2. GRF Plot
    grf_r_vx_idx = channel_names.index('calcn_r_force_vx')
    grf_r_vy_idx = channel_names.index('calcn_r_force_vy')
    grf_r_vz_idx = channel_names.index('calcn_r_force_vz')
    grf_l_vx_idx = channel_names.index('calcn_l_force_vx')
    grf_l_vy_idx = channel_names.index('calcn_l_force_vy')
    grf_l_vz_idx = channel_names.index('calcn_l_force_vz')
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    for i, (side, indices) in enumerate([
        ('Right', [grf_r_vx_idx, grf_r_vy_idx, grf_r_vz_idx]),
        ('Left', [grf_l_vx_idx, grf_l_vy_idx, grf_l_vz_idx])
    ]):
        for j, (comp, idx) in enumerate(zip(['Vx (A-P)', 'Vy (M-L)', 'Vz (Vertical)'], indices)):
            ax = axes[j, i]
            ax.plot(time, true_data[:, idx], 'b-', label='True', linewidth=2, alpha=0.8)
            ax.plot(time, pred_data[:, idx], 'r--', label='Predicted', linewidth=2, alpha=0.8)
            ax.set_xlabel('Time (s)', fontsize=11)
            ax.set_ylabel('Force (N)', fontsize=11)
            ax.set_title(f'{side} GRF {comp}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            if j == 0:
                ax.legend(loc='upper right', fontsize=10)
    
    plt.suptitle(f'Ground Reaction Forces: {trial_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'grf.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # 3. Pelvis Trajectory
    pelvis_tx_idx = channel_names.index('pelvis_tx')
    pelvis_ty_idx = channel_names.index('pelvis_ty')
    pelvis_tz_idx = channel_names.index('pelvis_tz')
    
    fig = plt.figure(figsize=(14, 5))
    
    # X-Z trajectory (side view)
    ax1 = fig.add_subplot(131)
    ax1.plot(true_data[:, pelvis_tx_idx], true_data[:, pelvis_ty_idx], 'b-', 
             label='True', linewidth=2, alpha=0.8)
    ax1.plot(pred_data[:, pelvis_tx_idx], pred_data[:, pelvis_ty_idx], 'r--', 
             label='Predicted', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Tx (m)', fontsize=11)
    ax1.set_ylabel('Ty (m)', fontsize=11)
    ax1.set_title('Pelvis Trajectory (Top View)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.axis('equal')
    
    # Time series
    ax2 = fig.add_subplot(132)
    ax2.plot(time, true_data[:, pelvis_tx_idx], 'b-', label='True Tx', linewidth=1.5)
    ax2.plot(time, pred_data[:, pelvis_tx_idx], 'r--', label='Pred Tx', linewidth=1.5)
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Position (m)', fontsize=11)
    ax2.set_title('Pelvis Forward Position', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    ax3 = fig.add_subplot(133)
    ax3.plot(time, true_data[:, pelvis_ty_idx], 'b-', label='True Ty', linewidth=1.5)
    ax3.plot(time, pred_data[:, pelvis_ty_idx], 'r--', label='Pred Ty', linewidth=1.5)
    ax3.set_xlabel('Time (s)', fontsize=11)
    ax3.set_ylabel('Height (m)', fontsize=11)
    ax3.set_title('Pelvis Vertical Position', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.suptitle(f'Pelvis Motion: {trial_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'pelvis.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved detailed plots: joints.png, grf.png, pelvis.png")


def validate_plausibility(data: np.ndarray, channel_names: List[str]) -> Dict:
    """
    Check if generated data is plausible gait.
    
    Returns:
        Dictionary with validation results
    """
    validation = {}
    
    # 1. Joint ROM (Range of Motion) check
    joint_indices = {
        'knee_r': channel_names.index('knee_angle_r'),
        'ankle_r': channel_names.index('ankle_angle_r'),
        'knee_l': channel_names.index('knee_angle_l'),
        'ankle_l': channel_names.index('ankle_angle_l'),
    }
    
    # Expected ROM for normal gait (in degrees)
    expected_rom = {
        'knee_r': (0, 70),   # Knee: 0-70 deg flexion
        'ankle_r': (-20, 30),  # Ankle: -20 to +30 deg
        'knee_l': (0, 70),
        'ankle_l': (-20, 30),
    }
    
    rom_valid = {}
    for joint, idx in joint_indices.items():
        angle_deg = np.rad2deg(data[:, idx])
        actual_range = (np.min(angle_deg), np.max(angle_deg))
        expected = expected_rom[joint]
        
        # Check if within reasonable bounds (allowing some margin)
        lower_ok = actual_range[0] >= expected[0] - 20
        upper_ok = actual_range[1] <= expected[1] + 20
        
        rom_valid[joint] = {
            'actual_range': actual_range,
            'expected_range': expected,
            'valid': lower_ok and upper_ok
        }
    
    validation['rom'] = rom_valid
    
    # 2. GRF pattern check
    grf_vz_r_idx = channel_names.index('calcn_r_force_vz')
    grf_vz_l_idx = channel_names.index('calcn_l_force_vz')
    
    grf_r = data[:, grf_vz_r_idx]
    grf_l = data[:, grf_vz_l_idx]
    
    # Check if GRF is mostly positive (can't push into ground)
    grf_r_negative_ratio = np.sum(grf_r < -10) / len(grf_r)
    grf_l_negative_ratio = np.sum(grf_l < -10) / len(grf_l)
    
    validation['grf'] = {
        'right_negative_ratio': float(grf_r_negative_ratio),
        'left_negative_ratio': float(grf_l_negative_ratio),
        'valid': grf_r_negative_ratio < 0.1 and grf_l_negative_ratio < 0.1
    }
    
    # 3. Periodicity check (using knee angle)
    knee_r_idx = channel_names.index('knee_angle_r')
    knee_signal = data[:, knee_r_idx]
    
    # Autocorrelation to detect gait cycles
    from scipy import signal as sp_signal
    autocorr = np.correlate(knee_signal - np.mean(knee_signal), 
                           knee_signal - np.mean(knee_signal), mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr / autocorr[0]
    
    # Find peaks (gait cycle repeats)
    peaks, _ = sp_signal.find_peaks(autocorr, height=0.3, distance=50)
    
    if len(peaks) > 1:
        avg_cycle_length = np.mean(np.diff(peaks))
        cycle_std = np.std(np.diff(peaks))
        periodicity_score = autocorr[peaks[0]] if len(peaks) > 0 else 0
    else:
        avg_cycle_length = 0
        cycle_std = 0
        periodicity_score = 0
    
    validation['periodicity'] = {
        'avg_cycle_length_frames': float(avg_cycle_length),
        'cycle_std_frames': float(cycle_std),
        'periodicity_score': float(periodicity_score),
        'valid': periodicity_score > 0.3
    }
    
    # 4. Symmetry check
    knee_r = data[:, channel_names.index('knee_angle_r')]
    knee_l = data[:, channel_names.index('knee_angle_l')]
    
    symmetry_corr = np.corrcoef(knee_r, knee_l)[0, 1]
    
    validation['symmetry'] = {
        'knee_lr_correlation': float(symmetry_corr),
        'valid': abs(symmetry_corr) > 0.5  # Should have some anti-correlation (alternating)
    }
    
    # Overall validation
    all_valid = (
        all(v['valid'] for v in rom_valid.values()) and
        validation['grf']['valid'] and
        validation['periodicity']['valid']
    )
    
    validation['overall_valid'] = all_valid
    
    return validation


def visualize_result_dir(result_dir: Path):
    """Process a single result directory."""
    print(f"\n{'='*60}")
    print(f"Processing: {result_dir}")
    print('='*60)
    
    # Check what type of result this is
    if (result_dir / 'inpainted.npz').exists():
        # LD02 inpainting
        true_data = np.load(result_dir / 'true.npz')['data']
        pred_data = np.load(result_dir / 'inpainted.npz')['data']
        trial_name = result_dir.name
        
        print(f"Type: Inpainting result")
        print(f"Data shape: {true_data.shape}")
        
        # Convert to .mot
        npz_to_mot(true_data, MODEL_STATES_COLUMN_NAMES_NO_ARM, 
                  result_dir / 'true.mot')
        npz_to_mot(pred_data, MODEL_STATES_COLUMN_NAMES_NO_ARM, 
                  result_dir / 'inpainted.mot')
        
        # Detailed plots
        plot_detailed_comparison(true_data, pred_data, 
                                MODEL_STATES_COLUMN_NAMES_NO_ARM,
                                result_dir, trial_name)
        
        # Validate predicted data
        validation = validate_plausibility(pred_data, MODEL_STATES_COLUMN_NAMES_NO_ARM)
        with open(result_dir / 'plausibility.json', 'w') as f:
            json.dump(validation, f, indent=2)
        
        print(f"Plausibility: {'✓ PASS' if validation['overall_valid'] else '✗ FAIL'}")
        
    elif (result_dir / 'generated.npz').exists():
        # LD03 generation
        gen_data = np.load(result_dir / 'generated.npz')['data']
        trial_name = result_dir.name
        
        print(f"Type: Generated sample")
        print(f"Data shape: {gen_data.shape}")
        
        # Convert to .mot
        npz_to_mot(gen_data, MODEL_STATES_COLUMN_NAMES_NO_ARM, 
                  result_dir / 'generated.mot')
        
        # Create self-visualization (no ground truth)
        T, C = gen_data.shape
        time = np.arange(T) / 100
        
        # Joint angles
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        joint_names = ['knee_angle_r', 'ankle_angle_r', 'subtalar_angle_r',
                      'knee_angle_l', 'ankle_angle_l', 'subtalar_angle_l']
        
        for i, joint_name in enumerate(joint_names):
            idx = MODEL_STATES_COLUMN_NAMES_NO_ARM.index(joint_name)
            ax = axes[i]
            ax.plot(time, np.rad2deg(gen_data[:, idx]), 'b-', linewidth=2)
            ax.set_xlabel('Time (s)', fontsize=11)
            ax.set_ylabel('Angle (deg)', fontsize=11)
            ax.set_title(joint_name.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Generated Joints: {trial_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(result_dir / 'joints.png', dpi=200, bbox_inches='tight')
        plt.close()
        
        # GRF
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        grf_channels = ['calcn_r_force_vx', 'calcn_r_force_vy', 'calcn_r_force_vz',
                       'calcn_l_force_vx', 'calcn_l_force_vy', 'calcn_l_force_vz']
        
        for i, grf_ch in enumerate(grf_channels):
            idx = MODEL_STATES_COLUMN_NAMES_NO_ARM.index(grf_ch)
            ax = axes[i // 2, i % 2]
            ax.plot(time, gen_data[:, idx], 'b-', linewidth=2)
            ax.set_xlabel('Time (s)', fontsize=11)
            ax.set_ylabel('Force (N)', fontsize=11)
            ax.set_title(grf_ch.replace('_', ' ').title(), fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Generated GRF: {trial_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(result_dir / 'grf.png', dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved plots: joints.png, grf.png")
        
        # Validate
        validation = validate_plausibility(gen_data, MODEL_STATES_COLUMN_NAMES_NO_ARM)
        with open(result_dir / 'plausibility.json', 'w') as f:
            json.dump(validation, f, indent=2)
        
        print(f"Plausibility: {'✓ PASS' if validation['overall_valid'] else '✗ FAIL'}")
        for key, val in validation.items():
            if key != 'overall_valid' and isinstance(val, dict) and 'valid' in val:
                status = '✓' if val['valid'] else '✗'
                print(f"  {status} {key}")
    
    else:
        print("Unknown result type (no .npz files found)")


def main():
    parser = argparse.ArgumentParser(description='Visualize LD results')
    parser.add_argument('--result_dir', type=str, required=True,
                       help='Path to result directory')
    parser.add_argument('--batch', action='store_true',
                       help='Process all subdirectories')
    
    args = parser.parse_args()
    
    result_path = Path(args.result_dir)
    
    if args.batch:
        # Process all subdirectories
        subdirs = [d for d in result_path.iterdir() if d.is_dir()]
        print(f"Found {len(subdirs)} subdirectories to process")
        for subdir in sorted(subdirs):
            try:
                visualize_result_dir(subdir)
            except Exception as e:
                print(f"ERROR processing {subdir}: {e}")
    else:
        # Process single directory
        visualize_result_dir(result_path)
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
