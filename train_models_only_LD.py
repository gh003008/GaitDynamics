"""
Training script for LD dataset only (GDP format)
Based on train_models.py but adapted for pre-converted GDP .npz files
"""
from args import parse_opt
from model.model import MotionModel
from data.preprocess import Normalizer
from torch.utils.data import Dataset
import torch
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import wandb
from torch.nn import functional as F
from model.utils import fix_seed
from consts import *
import copy

fix_seed()


class LDGDPDataset(Dataset):
    """Dataset loader for pre-converted LD GDP .npz files"""
    
    class Trial:
        def __init__(self, converted_pose, dset_name, sub_and_trial_name, height_m, weight_kg, length):
            self.converted_pose = converted_pose
            self.dset_name = dset_name
            self.sub_and_trial_name = sub_and_trial_name
            self.height_m = height_m
            self.weight_kg = weight_kg
            self.length = length
    
    def __init__(self, data_path: str, train: bool, opt, normalizer=None):
        self.data_path = data_path
        self.train = train
        self.opt = opt
        self.window_len = opt.window_len
        self.target_sampling_rate = opt.target_sampling_rate
        self.name = "Train" if self.train else "Test"
        self.trials = []
        
        print(f"Loading {self.name} dataset from {data_path}...")
        self.load_gdp_data(opt)
        
        if len(self.trials) == 0:
            print("No trials loaded")
            return
        
        # Collect dataset statistics
        self.dset_and_trials = {}
        for i_trial, trial in enumerate(self.trials):
            if trial.dset_name not in self.dset_and_trials:
                self.dset_and_trials[trial.dset_name] = [i_trial]
            else:
                self.dset_and_trials[trial.dset_name].append(i_trial)
        
        self.dset_num = len(self.dset_and_trials)
        self.dset_set = set(self.dset_and_trials.keys())  # For wandb logging compatibility
        
        for dset_name in self.dset_and_trials:
            print(f"{dset_name}: {len(self.dset_and_trials[dset_name])} trials")
        
        total_hour = sum([trial.length for trial in self.trials]) / self.target_sampling_rate / 60 / 60
        total_clip_num = sum([trial.length for trial in self.trials]) / self.target_sampling_rate / 3
        print(f"Total: {len(self.trials)} trials, {total_hour:.2f} hours, {total_clip_num:.0f} clips (not considering overlap)")
        
        # Normalize data
        if train:
            data_concat = torch.cat([trial.converted_pose for trial in self.trials], dim=0)
            print("Normalizing training data...")
            self.normalizer = Normalizer(data_concat, range(data_concat.shape[1]))
        else:
            self.normalizer = normalizer
        
        for i in range(len(self.trials)):
            self.trials[i].converted_pose = self.normalizer.normalize(
                self.trials[i].converted_pose
            ).clone().detach().float()
    
    def load_gdp_data(self, opt):
        """Load all .npz files from LD_gdp directory"""
        data_root = Path(self.data_path)
        
        if not data_root.exists():
            print(f"Error: Data path {data_root} does not exist")
            return
        
        # Get all subjects
        subjects = sorted([d for d in data_root.iterdir() if d.is_dir()])
        
        # Split train/test (use last subject for test)
        if self.train:
            subject_list = subjects[:-1]  # S004-S010 for training
            print(f"Training subjects: {[s.name for s in subject_list]}")
        else:
            subject_list = subjects[-1:]  # S011 for testing
            print(f"Test subjects: {[s.name for s in subject_list]}")
        
        for subject_dir in subject_list:
            subject_name = subject_dir.name
            
            # Iterate through conditions
            for condition_dir in sorted(subject_dir.iterdir()):
                if not condition_dir.is_dir():
                    continue
                
                condition_name = condition_dir.name
                
                # Load all trials in this condition
                for npz_file in sorted(condition_dir.glob("*.npz")):
                    trial_name = npz_file.stem
                    
                    try:
                        data = np.load(npz_file)
                        
                        # Extract data from npz
                        model_states = torch.from_numpy(data['model_states']).float()
                        height_m = float(data['height_m'])
                        weight_kg = float(data['weight_kg'])
                        
                        # Create trial object
                        dset_name = f"LD_{subject_name}"
                        sub_and_trial_name = f"{subject_name}_{condition_name}_{trial_name}"
                        
                        trial = self.Trial(
                            converted_pose=model_states,
                            dset_name=dset_name,
                            sub_and_trial_name=sub_and_trial_name,
                            height_m=height_m,
                            weight_kg=weight_kg,
                            length=model_states.shape[0]
                        )
                        
                        self.trials.append(trial)
                        
                    except Exception as e:
                        print(f"Error loading {npz_file}: {e}")
                        continue
    
    def __len__(self):
        return self.opt.pseudo_dataset_len if self.train else len(self.trials)
    
    def get_attributes_of_trials(self):
        """Return trial and dataset names for logging"""
        sub_and_trial_names = [trial.sub_and_trial_name for trial in self.trials]
        dset_names = [trial.dset_name for trial in self.trials]
        return sub_and_trial_names, dset_names
    
    def __getitem__(self, idx):
        """Sample a random window from a random trial
        Returns tuple matching addb_dataset.py format:
        (converted_pose, model_offsets, trial_idx, slice_index, height_m, cond)
        """
        if self.train:
            # Random trial selection
            trial_idx = np.random.randint(0, len(self.trials))
            trial = self.trials[trial_idx]
            
            # Random window
            if trial.length > self.window_len:
                start_idx = np.random.randint(0, trial.length - self.window_len)
                data = trial.converted_pose[start_idx:start_idx + self.window_len]
            else:
                # If trial is shorter than window, repeat it
                start_idx = 0
                data = trial.converted_pose
                while data.shape[0] < self.window_len:
                    data = torch.cat([data, trial.converted_pose], dim=0)
                data = data[:self.window_len]
        else:
            # For test, use sequential indexing
            trial_idx = idx % len(self.trials)
            trial = self.trials[trial_idx]
            
            if trial.length > self.window_len:
                start_idx = 0
                data = trial.converted_pose[start_idx:start_idx + self.window_len]
            else:
                start_idx = 0
                data = trial.converted_pose
                while data.shape[0] < self.window_len:
                    data = torch.cat([data, trial.converted_pose], dim=0)
                data = data[:self.window_len]
        
        # Create dummy model_offsets (not used in LD training)
        model_offsets = torch.zeros(1, 4, 4)  # Dummy offsets
        
        # Condition (trial.cond would be walking condition, use trial name as placeholder)
        cond = trial.sub_and_trial_name
        
        # Return in the format expected by model.py:
        # Line 191: cond = x[5]
        # Line 713: x_start, model_offsets, _, _, height_m, _ = x
        # Matching addb_dataset.py: (converted_pose, model_offsets, i_trial, slice_index, height_m, cond)
        return (data, model_offsets, trial_idx, start_idx, trial.height_m, cond)


def train_ld(opt):
    """Training function for LD dataset"""
    
    # Load training dataset first
    print("Loading training dataset...")
    train_dataset = LDGDPDataset(
        data_path=opt.data_path_train,
        train=True,
        opt=opt
    )
    
    # Create model
    print("Creating model...")
    model = MotionModel(
        opt,
        learning_rate=opt.learning_rate,
        weight_decay=opt.weight_decay
    )
    
    # Initialize wandb if enabled (with error handling for metadata issues)
    if opt.log_with_wandb:
        try:
            # Set environment variable to skip package metadata collection (workaround for wandb bug)
            os.environ["WANDB_DISABLE_CODE"] = "true"
            
            wandb.init(
                project=opt.wandb_pj_name,
                name=opt.exp_name,
                dir="wandb_logs",
                config=vars(opt),
                settings=wandb.Settings(_disable_meta=True)  # Disable metadata collection
            )
            
            # Log hyperparameters explicitly for sweep tracking
            wandb.config.update({
                'learning_rate': opt.learning_rate,
                'weight_decay': opt.weight_decay,
                'batch_size': opt.batch_size,
                'pseudo_dataset_len': opt.pseudo_dataset_len,
                'epochs': opt.epochs,
                'window_len': opt.window_len,
                'num_params': sum(p.numel() for p in model.diffusion.parameters()),
                'num_train_trials': len(train_dataset.trials),
            }, allow_val_change=True)
            
            # Log initial data statistics for reference
            sample_data = train_dataset.trials[0].converted_pose  # Normalized data
            wandb.log({
                'data_stats/normalized_mean': sample_data.mean().item(),
                'data_stats/normalized_std': sample_data.std().item(),
                'data_stats/normalized_min': sample_data.min().item(),
                'data_stats/normalized_max': sample_data.max().item(),
            })
            
            # Log channel-wise statistics for key joints
            for i, col_name in enumerate(opt.model_states_column_names):
                if any(key in col_name for key in ['knee_angle', 'hip_flexion', 'ankle_angle', 'ground_force']):
                    col_data = sample_data[:, i]
                    wandb.log({
                        f'data_stats/{col_name}_mean': col_data.mean().item(),
                        f'data_stats/{col_name}_std': col_data.std().item(),
                    })
            
            wandb.watch(model.diffusion, F.mse_loss, log='all', log_freq=200)
            print("✓ Wandb initialized successfully")
            print(f"  - Learning Rate: {opt.learning_rate}")
            print(f"  - Batch Size: {opt.batch_size}")
            print(f"  - Pseudo Dataset Length: {opt.pseudo_dataset_len}")
            print(f"  - Epochs: {opt.epochs}")
            print(f"  - Model Parameters: {sum(p.numel() for p in model.diffusion.parameters()):,}")
        except Exception as e:
            print(f"⚠ Warning: wandb initialization failed: {e}")
            print("Continuing without wandb logging...")
            opt.log_with_wandb = False
    
    # Train
    print("\n" + "="*50)
    print("Starting training on LD dataset...")
    print("="*50 + "\n")
    
    model.train_loop(opt, train_dataset)
    
    print("\n" + "="*50)
    print("Training complete!")
    print("="*50 + "\n")


def setup_ld_args(opt):
    """Setup arguments specific to LD dataset"""
    
    # LD data uses GDP format (no arm)
    opt.with_arm = False
    opt.osim_dof_columns = copy.deepcopy(OSIM_DOF_ALL[:23] + KINETICS_ALL)
    opt.joints_3d = {
        key_: value_ for key_, value_ in JOINTS_3D_ALL.items() 
        if key_ in ['pelvis', 'hip_r', 'hip_l', 'lumbar']
    }
    
    # Set LD data paths
    opt.data_path_train = 'data/LD_gdp/'
    opt.data_path_test = 'data/LD_gdp/'
    
    # OpenSim model path (using generic unscaled model)
    opt.data_path_osim_model = opt.data_path_parent + 'osim_model/unscaled_generic_no_arm.osim'
    
    # Model states columns
    opt.model_states_column_names = copy.deepcopy(MODEL_STATES_COLUMN_NAMES_NO_ARM)
    
    # Add angular velocity columns for 3DOF joints
    for joint_name in opt.joints_3d:
        opt.model_states_column_names += [
            joint_name + '_' + axis + '_angular_vel' for axis in ['x', 'y', 'z']
        ]
    
    # Add velocity columns if enabled
    if opt.with_kinematics_vel:
        opt.model_states_column_names += [
            f'{col}_vel' for col in opt.model_states_column_names
            if not any(term in col for term in ['force', 'pelvis_', '_vel', '_0', '_1', '_2', '_3', '_4', '_5'])
        ]
    
    # Column location indices
    opt.knee_diffusion_col_loc = [
        i for i, col in enumerate(opt.model_states_column_names) if 'knee' in col
    ]
    opt.ankle_diffusion_col_loc = [
        i for i, col in enumerate(opt.model_states_column_names) if 'ankle' in col
    ]
    opt.hip_diffusion_col_loc = [
        i for i, col in enumerate(opt.model_states_column_names) if 'hip' in col
    ]
    opt.kinematic_diffusion_col_loc = [
        i for i, col in enumerate(opt.model_states_column_names) if 'force' not in col
    ]
    opt.kinetic_diffusion_col_loc = [
        i for i, col in enumerate(opt.model_states_column_names) 
        if i not in opt.kinematic_diffusion_col_loc
    ]
    opt.grf_osim_col_loc = [
        i for i, col in enumerate(opt.osim_dof_columns) 
        if 'force' in col and '_cop_' not in col
    ]
    opt.cop_osim_col_loc = [
        i for i, col in enumerate(opt.osim_dof_columns) if '_cop_' in col
    ]
    opt.kinematic_osim_col_loc = [
        i for i, col in enumerate(opt.osim_dof_columns) if 'force' not in col
    ]
    
    # Adjust hyperparameters for smaller dataset
    print("\nLD Dataset Training Configuration:")
    print(f"  - Batch size: {opt.batch_size}")
    print(f"  - Pseudo dataset length: {opt.pseudo_dataset_len}")
    print(f"  - Epochs: {opt.epochs}")
    print(f"  - Window length: {opt.window_len}")
    print(f"  - Target sampling rate: {opt.target_sampling_rate} Hz")
    print(f"  - Train data path: {opt.data_path_train}")
    print(f"  - Test data path: {opt.data_path_test}")
    print(f"  - Wandb logging: {opt.log_with_wandb}")
    print()
    
    return opt


if __name__ == "__main__":
    # Parse arguments
    opt = parse_opt()
    
    # Setup LD-specific arguments
    opt = setup_ld_args(opt)
    
    # Start training
    train_ld(opt)
