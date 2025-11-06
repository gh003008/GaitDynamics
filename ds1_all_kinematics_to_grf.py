"""
Downstream Task 1: Full-body kinematics to GRF estimation
Estimate ground reaction forces using complete kinematic inputs from LD data.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
import os
import sys
from inspect import isfunction
from einops import rearrange, repeat
import math

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from args import parse_opt
from consts import *
from model.model import BaselineModel

# ============================================================================
# TransformerEncoderArchitecture and dependencies from gait_dynamics.py
# (Copied to avoid import issues with usr_inputs() execution)
# ============================================================================

def exists(val):
    return val is not None

def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")

def apply_rotary_emb(freqs, t, start_index=0):
    freqs = freqs.to(t)
    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim
    assert (
            rot_dim <= t.shape[-1]
    ), f"feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}"
    t_left, t, t_right = (
        t[..., :start_index],
        t[..., start_index:end_index],
        t[..., end_index:],
    )
    t = (t * freqs.cos()) + (rotate_half(t) * freqs.sin())
    return torch.cat((t_left, t, t_right), dim=-1)

class RotaryEmbedding(nn.Module):
    def __init__(
            self,
            dim,
            custom_freqs=None,
            freqs_for="lang",
            theta=10000,
            max_freq=10,
            num_freqs=1,
            learned_freq=False,
    ):
        super().__init__()
        if exists(custom_freqs):
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / (
                    theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
            )
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * math.pi
        elif freqs_for == "constant":
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f"unknown modality {freqs_for}")

        self.cache = dict()

        if learned_freq:
            self.freqs = nn.Parameter(freqs)
        else:
            self.register_buffer("freqs", freqs)

    def rotate_queries_or_keys(self, t, seq_dim=-2):
        device = t.device
        seq_len = t.shape[seq_dim]
        freqs = self.forward(
            lambda: torch.arange(seq_len, device=device), cache_key=seq_len
        )
        return apply_rotary_emb(freqs, t)

    def forward(self, t, cache_key=None):
        if exists(cache_key) and cache_key in self.cache:
            return self.cache[cache_key]

        if isfunction(t):
            t = t()

        freqs = self.freqs

        freqs = torch.einsum("..., f -> ... f", t.type(freqs.dtype), freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)

        if exists(cache_key):
            self.cache[cache_key] = freqs

        return freqs

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads=8, d_ff=512, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.rotary = RotaryEmbedding(dim=d_model)
        self.use_rotary = self.rotary is not None

        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        qk = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        attn_output, _ = self.self_attn(qk, qk, x, need_weights=False)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class TransformerEncoderArchitecture(nn.Module):
    def __init__(self, repr_dim, opt, nlayers=6):
        super(TransformerEncoderArchitecture, self).__init__()
        self.input_dim = len(opt.kinematic_diffusion_col_loc)
        self.output_dim = repr_dim - self.input_dim
        embedding_dim = 192
        self.input_to_embedding = nn.Linear(self.input_dim, embedding_dim)
        self.encoder_layers = nn.Sequential(*[EncoderLayer(embedding_dim) for _ in range(nlayers)])
        self.embedding_to_output = nn.Linear(embedding_dim, self.output_dim)
        self.opt = opt
        self.input_col_loc = opt.kinematic_diffusion_col_loc
        self.output_col_loc = [i for i in range(repr_dim) if i not in self.input_col_loc]

    def loss_fun(self, output_pred, output_true):
        return F.mse_loss(output_pred, output_true, reduction='none')

    def end_to_end_prediction(self, x):
        input = x[0][:, :, self.input_col_loc]
        sequence = self.input_to_embedding(input)
        sequence = self.encoder_layers(sequence)
        output_pred = self.embedding_to_output(sequence)
        return output_pred
    
    def get_optimizer(self):
        from model.adan import Adan
        return Adan(self.parameters(), lr=4e-4, weight_decay=0.02)

    def __str__(self):
        return 'tf'

# ============================================================================
# End of copied code from gait_dynamics.py
# ============================================================================


class Args:
    pass


def load_gdp_data(gdp_path):
    """Load GDP npz file and extract model states."""
    print(f"Loading GDP file: {gdp_path}")
    data = np.load(gdp_path, allow_pickle=True)
    
    model_states = data['model_states']  # T x D
    model_states_columns = data['model_states_columns']
    sampling_rate = float(data['sampling_rate'])
    
    print(f"  Model states shape: {model_states.shape}")
    print(f"  Sampling rate: {sampling_rate} Hz")
    print(f"  Duration: {model_states.shape[0] / sampling_rate:.2f} seconds")
    
    return model_states, model_states_columns, sampling_rate


def prepare_windows(model_states, window_len=150, stride=75):
    """Prepare sliding windows from the model states."""
    num_frames = model_states.shape[0]
    num_channels = model_states.shape[1]
    
    windows = []
    start_indices = []
    
    for start_idx in range(0, num_frames - window_len + 1, stride):
        window = model_states[start_idx:start_idx + window_len, :]
        windows.append(window)
        start_indices.append(start_idx)
    
    windows = np.stack(windows)  # N x window_len x channels
    print(f"Created {len(windows)} windows (window_len={window_len}, stride={stride})")
    
    return torch.from_numpy(windows).float(), start_indices


def estimate_grf_with_refinement(opt, model_states_windows):
    """
    Estimate GRF using the refinement model with full-body kinematics.
    """
    print("\nLoading GaitDynamicsRefinement model...")
    
    refinement_model = BaselineModel(opt, TransformerEncoderArchitecture)
    
    # Create mask: all kinematics are known (1), all kinetics are unknown (0)
    masks = torch.zeros_like(model_states_windows)
    masks[:, :, opt.kinematic_diffusion_col_loc] = 1.0
    
    print(f"Input shape: {model_states_windows.shape}")
    print(f"Kinematic channels: {len(opt.kinematic_diffusion_col_loc)}")
    print(f"Kinetic channels: {len(opt.kinetic_diffusion_col_loc)}")
    
    # Predict forces
    print("Estimating GRF from full-body kinematics...")
    state_pred = refinement_model.eval_loop(
        opt, 
        model_states_windows, 
        masks,
        num_of_generation_per_window=1,
        mode="inpaint"
    )
    
    # state_pred shape: [1, N, window_len, channels]
    state_pred = state_pred[0]  # Remove generation dimension
    
    return state_pred


def plot_grf_comparison(measured_states, predicted_states, opt, start_indices, sampling_rate, save_path):
    """
    Plot comparison between measured and predicted GRF.
    """
    # Find kinetic column indices
    kinetic_cols = opt.kinetic_diffusion_col_loc
    kinetic_names = [opt.model_states_column_names[i] for i in kinetic_cols]
    
    # Reconstruct full time series from windows (use first window, then non-overlapping parts)
    window_len = measured_states.shape[1]
    
    # Use first window completely, then add stride portions from subsequent windows
    measured_series = measured_states[0].numpy()
    predicted_series = predicted_states[0].numpy()
    
    stride = start_indices[1] - start_indices[0] if len(start_indices) > 1 else window_len
    
    for i in range(1, len(measured_states)):
        measured_series = np.vstack([measured_series, measured_states[i, -stride:].numpy()])
        predicted_series = np.vstack([predicted_series, predicted_states[i, -stride:].numpy()])
    
    time = np.arange(len(measured_series)) / sampling_rate
    
    # Group by foot (calcn_r, calcn_l)
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle('GRF Estimation: Full Kinematics → Forces (Refinement Model)', fontsize=16, fontweight='bold')
    
    # Right foot (top row)
    r_cols = [i for i, name in enumerate(kinetic_names) if 'calcn_r' in name]
    # Left foot (bottom row)
    l_cols = [i for i, name in enumerate(kinetic_names) if 'calcn_l' in name]
    
    for row, (foot_cols, foot_label) in enumerate([(r_cols, 'Right Foot'), (l_cols, 'Left Foot')]):
        for col_idx, kinetic_idx in enumerate(foot_cols[:3]):  # Force components
            ax = axes[row, col_idx]
            kinetic_name = kinetic_names[kinetic_idx]
            global_idx = kinetic_cols[kinetic_idx]
            
            # Extract data
            measured = measured_series[:, global_idx]
            predicted = predicted_series[:, global_idx]
            
            # Plot
            ax.plot(time, measured, 'b-', linewidth=1.5, label='Measured', alpha=0.7)
            ax.plot(time, predicted, 'r--', linewidth=1.5, label='Predicted', alpha=0.7)
            
            # Calculate error
            mae = np.mean(np.abs(measured - predicted))
            rmse = np.sqrt(np.mean((measured - predicted)**2))
            
            # Labels
            if 'force_vx' in kinetic_name:
                title = f'{foot_label} - AP Force'
                ylabel = 'Force (N)'
            elif 'force_vy' in kinetic_name:
                title = f'{foot_label} - Vertical Force'
                ylabel = 'Force (N)'
            elif 'force_vz' in kinetic_name:
                title = f'{foot_label} - ML Force'
                ylabel = 'Force (N)'
            elif 'cop_x' in kinetic_name:
                title = f'{foot_label} - CoP X'
                ylabel = 'Position (normalized)'
            elif 'cop_y' in kinetic_name:
                title = f'{foot_label} - CoP Y'
                ylabel = 'Position (normalized)'
            elif 'cop_z' in kinetic_name:
                title = f'{foot_label} - CoP Z'
                ylabel = 'Position (normalized)'
            else:
                title = kinetic_name
                ylabel = 'Value'
            
            ax.set_title(f'{title}\nMAE: {mae:.3f}, RMSE: {rmse:.3f}', fontsize=10)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(ylabel)
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison plot to: {save_path}")
    plt.close()
    
    # Print summary statistics
    print("\n" + "="*80)
    print("GRF ESTIMATION SUMMARY (Full Kinematics)")
    print("="*80)
    for kinetic_idx, kinetic_name in enumerate(kinetic_names):
        global_idx = kinetic_cols[kinetic_idx]
        measured = measured_series[:, global_idx]
        predicted = predicted_series[:, global_idx]
        
        mae = np.mean(np.abs(measured - predicted))
        rmse = np.sqrt(np.mean((measured - predicted)**2))
        
        print(f"{kinetic_name:40s} | MAE: {mae:8.4f} | RMSE: {rmse:8.4f}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Downstream Task 1: Full kinematics to GRF')
    parser.add_argument('--gdp_path', type=str, required=True,
                       help='Path to GDP npz file (LD data)')
    parser.add_argument('--save_dir', type=str, default='plots/downstream_task1',
                       help='Directory to save plots')
    parser.add_argument('--window_len', type=int, default=150,
                       help='Window length in frames')
    parser.add_argument('--stride', type=int, default=75,
                       help='Window stride in frames')
    
    script_args = parser.parse_args()
    
    # Load data
    model_states, model_states_columns, sampling_rate = load_gdp_data(script_args.gdp_path)
    
    # Prepare windows
    model_states_windows, start_indices = prepare_windows(
        model_states, 
        window_len=script_args.window_len, 
        stride=script_args.stride
    )
    
    # Create options using parse_opt from args.py (with empty args to avoid conflicts)
    import sys
    original_argv = sys.argv
    sys.argv = [sys.argv[0]]  # Only keep script name
    opt = parse_opt()
    sys.argv = original_argv  # Restore original argv
    
    opt.window_len = script_args.window_len
    opt.checkpoint_bl = os.path.join(project_root, 'example_usage', 'GaitDynamicsRefinement.pt')
    
    # Estimate GRF using refinement model
    predicted_states = estimate_grf_with_refinement(opt, model_states_windows)
    
    # Generate save path
    trial_name = os.path.splitext(os.path.basename(script_args.gdp_path))[0]
    save_path = os.path.join(script_args.save_dir, f'{trial_name}_all_kinematics.png')
    
    # Plot comparison
    plot_grf_comparison(
        model_states_windows, 
        predicted_states, 
        opt, 
        start_indices, 
        sampling_rate,
        save_path
    )
    
    print("\nTask completed successfully!")


if __name__ == '__main__':
    main()
