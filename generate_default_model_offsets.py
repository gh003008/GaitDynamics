"""
Generate default model offsets from example OpenSim model
This can be used for LD dataset which doesn't have skeleton information
"""
import torch
import nimblephysics as nimble
from data.osim_fk import get_model_offsets
import numpy as np

# Load example OpenSim model
osim_path = "example_usage/example_opensim_model.osim"
print(f"Loading OpenSim model from: {osim_path}")

skel = nimble.biomechanics.OpenSimParser.parseOsim(osim_path)
print(f"Skeleton loaded: {skel.getNumBodyNodes()} body nodes, {skel.getNumJoints()} joints")

# Get model offsets (without arm since LD data doesn't have arm)
model_offsets = get_model_offsets(skel, with_arm=False)
print(f"Model offsets shape: {model_offsets.shape}")
print(f"Model offsets type: {model_offsets.dtype}")

# Save as numpy file
output_path = "data/default_model_offsets_no_arm.npy"
np.save(output_path, model_offsets.numpy())
print(f"\nSaved model offsets to: {output_path}")
print("This file can be loaded in LDGDPDataset for FK calculations")
