#!/usr/bin/env python3
"""
Post-processing script to rescale pelvis_tx to match target velocity.

Usage:
    python rescale_pelvis_tx.py input.csv --target-speed 1.2 --output output.csv

This is useful when multi-window stitching causes velocity errors due to overlap-add.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def rescale_pelvis_tx(input_csv, target_speed_ms, output_csv=None, height_m=1.75):
    """
    Rescale pelvis_tx in a CSV file to match target speed.
    
    Args:
        input_csv: Path to input CSV file
        target_speed_ms: Target speed in m/s
        output_csv: Output path (default: input_rescaled.csv)
        height_m: Subject height in meters (for velocity calculation)
    """
    # Load data
    df = pd.read_csv(input_csv)
    
    if 'pelvis_tx' not in df.columns or 'time' not in df.columns:
        raise ValueError("CSV must contain 'pelvis_tx' and 'time' columns")
    
    # Calculate current velocity
    ptx = df['pelvis_tx'].values
    time = df['time'].values
    current_vel_ms = np.diff(ptx).sum() / (time[-1] - time[0])  # Average velocity
    
    # Calculate scale factor
    scale_factor = target_speed_ms / current_vel_ms
    
    # Rescale pelvis_tx (preserve starting position)
    ptx_start = ptx[0]
    df['pelvis_tx'] = ptx_start + (ptx - ptx_start) * scale_factor
    
    # Verify result
    new_vel_ms = np.diff(df['pelvis_tx'].values).sum() / (time[-1] - time[0])
    
    # Save
    if output_csv is None:
        input_path = Path(input_csv)
        output_csv = input_path.parent / f"{input_path.stem}_rescaled{input_path.suffix}"
    
    df.to_csv(output_csv, index=False)
    
    # Print summary
    print("=" * 70)
    print("Pelvis TX Rescale Summary")
    print("=" * 70)
    print(f"Input:  {input_csv}")
    print(f"Output: {output_csv}")
    print()
    print(f"Original velocity: {current_vel_ms:.3f} m/s")
    print(f"Target velocity:   {target_speed_ms:.3f} m/s")
    print(f"Scale factor:      {scale_factor:.4f}")
    print(f"Result velocity:   {new_vel_ms:.3f} m/s")
    print(f"Error:             {abs(new_vel_ms - target_speed_ms) / target_speed_ms * 100:.2f}%")
    print("=" * 70)
    
    return output_csv


def main():
    parser = argparse.ArgumentParser(
        description="Rescale pelvis_tx to match target velocity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Rescale to 1.2 m/s
    python rescale_pelvis_tx.py data.csv --target-speed 1.2
    
    # Specify output path
    python rescale_pelvis_tx.py data.csv --target-speed 1.5 --output corrected.csv
    
    # Different subject height
    python rescale_pelvis_tx.py data.csv --target-speed 1.2 --height 1.60
        """
    )
    
    parser.add_argument('input_csv', type=str,
                        help='Input CSV file with pelvis_tx and time columns')
    parser.add_argument('--target-speed', type=float, required=True,
                        help='Target walking speed in m/s')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output CSV path (default: input_rescaled.csv)')
    parser.add_argument('--height', type=float, default=1.75,
                        help='Subject height in meters (default: 1.75)')
    
    args = parser.parse_args()
    
    rescale_pelvis_tx(
        args.input_csv,
        args.target_speed,
        args.output,
        args.height
    )


if __name__ == '__main__':
    main()
