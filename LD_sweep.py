"""
Hyperparameter Sweep for LD Training

Test different configurations to find optimal settings for plausible gait generation.

Key hypotheses to test:
1. More epochs needed (100 → 300-500)
2. Learning rate might be too high/low
3. Diffusion steps might need adjustment
4. Model size might need to be larger

Sweep configurations:
- Config 1 (Baseline+): More epochs, same LR
- Config 2 (Lower LR): Slower learning, more stable
- Config 3 (Higher capacity): Larger model
- Config 4 (More diffusion steps): Better sampling
- Config 5 (Aggressive): High LR, many epochs

Usage:
    python LD_sweep.py --config 1
    python LD_sweep.py --config 2
    # ... etc
"""

import os
import sys
import argparse
from pathlib import Path

# Sweep configurations
# Note: Current train_models_only_LD.py only supports: epochs, pseudo_dataset_len, batch_size
# LR and model architecture are hardcoded in model code
CONFIGS = {
    1: {
        'name': 'baseline_plus',
        'description': 'More epochs with same settings',
        'epochs': 300,
        'pseudo_dataset_len': 10000,
        'batch_size': 32,
        'notes': 'Test if 100 epochs was just too short'
    },
    
    2: {
        'name': 'long_training',
        'description': 'Very long training for full convergence',
        'epochs': 500,
        'pseudo_dataset_len': 10000,
        'batch_size': 32,
        'notes': 'Give model plenty of time to converge'
    },
    
    3: {
        'name': 'smaller_batch',
        'description': 'Smaller batch for more gradient updates',
        'epochs': 300,
        'pseudo_dataset_len': 10000,
        'batch_size': 16,  # Half the default
        'notes': 'More frequent updates, might help escape bad modes'
    },
    
    4: {
        'name': 'larger_pseudo',
        'description': 'Larger pseudo dataset for more diversity',
        'epochs': 300,
        'pseudo_dataset_len': 20000,  # 2x larger
        'batch_size': 32,
        'notes': 'More data variation per epoch'
    },
    
    5: {
        'name': 'ultra_long',
        'description': 'Very long training with large dataset',
        'epochs': 1000,
        'pseudo_dataset_len': 20000,
        'batch_size': 32,
        'notes': 'Maximum training time to ensure full convergence'
    },
}


def print_config_summary():
    """Print summary of all configurations."""
    print("\n" + "="*80)
    print("HYPERPARAMETER SWEEP CONFIGURATIONS")
    print("="*80)
    
    for config_id, config in CONFIGS.items():
        print(f"\nConfig {config_id}: {config['name']}")
        print(f"  Description: {config['description']}")
        print(f"  Epochs: {config['epochs']}")
        print(f"  Pseudo dataset length: {config['pseudo_dataset_len']}")
        print(f"  Batch size: {config['batch_size']}")
        print(f"  Notes: {config['notes']}")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS:")
    print("  - Start with Config 1 (baseline_plus) - safest option")
    print("  - If Config 1 works but slow → try Config 5 (aggressive)")
    print("  - If Config 1 fails → try Config 2 (lower_lr)")
    print("  - If still mode collapse → try Config 3 (higher_capacity)")
    print("  - Config 4 for final high-quality model")
    print("="*80 + "\n")


def run_config(config_id: int):
    """Run training with specified configuration."""
    if config_id not in CONFIGS:
        raise ValueError(f"Invalid config ID: {config_id}. Choose from {list(CONFIGS.keys())}")
    
    config = CONFIGS[config_id]
    exp_name = f"LD_sweep_{config['name']}"
    
    print("\n" + "="*80)
    print(f"RUNNING CONFIG {config_id}: {config['name']}")
    print("="*80)
    print(f"Experiment name: {exp_name}")
    print(f"Description: {config['description']}")
    print(f"Notes: {config['notes']}")
    print("="*80 + "\n")
    
    # Build command
    cmd_parts = [
        "python train_models_only_LD.py",
        f"--exp_name {exp_name}",
        f"--epochs {config['epochs']}",
        f"--pseudo_dataset_len {config['pseudo_dataset_len']}",
        f"--batch_size {config['batch_size']}",
        "--log_with_wandb",  # Always log to wandb for comparison
    ]
    
    cmd = " ".join(cmd_parts)
    
    print(f"Command: {cmd}\n")
    
    # For Windows, need to set env var
    full_cmd = f'$env:KMP_DUPLICATE_LIB_OK="TRUE"; {cmd}'
    
    print("To run this configuration manually:")
    print(f"  conda activate gaitdyn2")
    print(f"  {full_cmd}")
    print()
    
    # Ask for confirmation
    response = input("Run this configuration now? [y/N]: ")
    if response.lower() == 'y':
        import subprocess
        result = subprocess.run(
            ['powershell', '-Command', f'conda activate gaitdyn2; {full_cmd}'],
            cwd=os.getcwd()
        )
        return result.returncode == 0
    else:
        print("Skipped. Run manually with the command above.")
        return False


def generate_batch_script():
    """Generate a batch script to run all configs."""
    script_lines = [
        "# Hyperparameter Sweep Batch Script",
        "# Run all configurations sequentially",
        "",
        "conda activate gaitdyn2",
        '$env:KMP_DUPLICATE_LIB_OK="TRUE"',
        "",
    ]
    
    for config_id, config in CONFIGS.items():
        exp_name = f"LD_sweep_{config['name']}"
        script_lines.append(f"# Config {config_id}: {config['description']}")
        
        cmd_parts = [
            "python train_models_only_LD.py",
            f"--exp_name {exp_name}",
            f"--epochs {config['epochs']}",
            f"--pseudo_dataset_len {config['pseudo_dataset_len']}",
            f"--batch_size {config['batch_size']}",
            "--log_with_wandb",
        ]
        
        script_lines.append(" ".join(cmd_parts))
        script_lines.append("")
    
    script_content = "\n".join(script_lines)
    
    with open('run_sweep.ps1', 'w') as f:
        f.write(script_content)
    
    print("Generated: run_sweep.ps1")
    print("To run all configs: powershell -File run_sweep.ps1")


def quick_test_all():
    """Generate quick test versions (2 epochs each) for validation."""
    print("\n" + "="*80)
    print("QUICK TEST MODE")
    print("="*80)
    print("Running 2-epoch tests of all configs to validate setup...\n")
    
    for config_id, config in CONFIGS.items():
        exp_name = f"LD_sweep_test_{config['name']}"
        
        cmd_parts = [
            "python train_models_only_LD.py",
            f"--exp_name {exp_name}",
            "--epochs 2",  # Quick test
            "--pseudo_dataset_len 100",  # Small dataset
            f"--batch_size {config['batch_size']}",
        ]
        
        cmd = " ".join(cmd_parts)
        print(f"Config {config_id} test: {cmd}")
    
    print("\nRun these commands to validate each config works before full training.")


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter sweep for LD training')
    parser.add_argument('--config', type=int, choices=list(CONFIGS.keys()),
                       help='Configuration ID to run')
    parser.add_argument('--list', action='store_true',
                       help='List all configurations')
    parser.add_argument('--generate-batch', action='store_true',
                       help='Generate batch script to run all configs')
    parser.add_argument('--quick-test', action='store_true',
                       help='Show quick test commands (2 epochs each)')
    
    args = parser.parse_args()
    
    if args.list or (not args.config and not args.generate_batch and not args.quick_test):
        print_config_summary()
    
    if args.config:
        run_config(args.config)
    
    if args.generate_batch:
        generate_batch_script()
    
    if args.quick_test:
        quick_test_all()


if __name__ == "__main__":
    main()
