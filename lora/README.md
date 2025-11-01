# LoRA Adaptor Training for GaitDynamics

This folder contains a minimal training script to specialize the pretrained diffusion model to a target gait style (e.g., crouch) using Low-Rank Adaptation (LoRA).

## What it does
- Loads the pretrained DanceDecoder + GaussianDiffusion from a checkpoint (.pt) that includes the embedded `Normalizer`.
- Injects LoRA into the model's linear layers (input projection, FFN linear1/linear2, and MultiheadAttention out_proj) and freezes all base weights.
- Trains only the LoRA parameters on crouch windows read from an NPZ dataset tree.
- Saves only LoRA weights (plus metadata) to `runs/adapter_train/<exp>/weights/*.pt` and a `config.json`.

## Expected NPZ format
Each trial is an `.npz` with the following keys:
- `model_states`: float32 array of shape (T, C)
- `model_states_columns`: list/array of C column names (must match `opt.model_states_column_names` ordering; if not, script reorders)
- `probably_missing`: bool array of shape (T,) indicating unreliable kinetics windows
- `height_m`: scalar
- `weight_kg`: scalar
- `sampling_rate`: scalar (Hz)
- `pos_vec`: float32 array (3,)

Example tree (crouch):
```
gdp_ld_crouch/
  S006/
    crouch/
      trial_01.npz
  S007/
    crouch/
      trial_03.npz
...
```

## How to run
Use the repository's main CLI to pass the base checkpoint (for the embedded `Normalizer` and model state):

```bash
# Minimal run (single GPU). Adjust envs as needed.
python lora/train_lora_adapter.py \
  --checkpoint example_usage/GaitDynamicsDiffusion.pt \
  --window_len 150 --target_sampling_rate 100
```

Environment variables (optional) control training details:
- `LORA_EXP_NAME` (default: `lora_crouch`)
- `LORA_DATA_ROOT` (default: `gdp_ld_crouch`)
- `LORA_SAVE_DIR` (default: `runs/adapter_train`)
- `LORA_R` (default: `16`)
- `LORA_ALPHA` (default: `32`)
- `LORA_DROPOUT` (default: `0.05`)
- `LORA_EPOCHS` (default: `5`)
- `LORA_BATCH` (default: `args.batch_size // 2`)
- `LORA_LR` (default: `1e-3`)

The script prints how many LoRA modules were injected and the number of trainable parameters, then trains and saves a LoRA checkpoint after each epoch.

## Using the LoRA adaptor for sampling
At inference time (e.g., in your generation script), after building the base model and loading the base checkpoint, call `inject_lora(...)` with the same hyperparameters and then `load_lora_weights(...)` to load the saved adapter weights. Only LoRA deltas are loaded; the base weights remain frozen.

## Notes
- FK-related losses are zero-weighted in the base training objective, but the diffusion loss path still runs FK. This script provides identity `model_offsets` to keep that code path valid without requiring Nimble skeletons for each trial.
- The script reorders NPZ columns to match `opt.model_states_column_names` to ensure compatibility with the pretrained scaler and model.
- Saves include the embedded `Normalizer` alongside LoRA weights for convenient reuse (optional).
