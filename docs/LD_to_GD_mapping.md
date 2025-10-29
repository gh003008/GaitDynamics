# Laboratory Data (LD) → GaitDynamics (GD) format mapping

This document summarizes how to transform our lab HDF5 data (subjects S004–S011) into the standardized training window format used by GaitDynamics.

## GD data window (target)
- Sampling: 100 Hz
- Window length: 150 frames (1.5 s)
- Columns (no-arm model): see `consts.py`
  - Kinematics base:
    - Pelvis translations: pelvis_tx, pelvis_ty, pelvis_tz (meters, Earth frame, later converted to velocities inside pipeline)
    - Selected Euler angles: pelvis, hip_r, hip_l, lumbar (ZXY order)
    - Joint angles: knee, ankle, subtalar, mtp (arms removed)
  - Kinetic targets per foot (six channels per foot):
    - calcn_<r|l>_force_vx, _vy, _vz (ground reaction force in BW)
    - calcn_<r|l>_force_normed_cop_x, _y, _z (normalized CoP; see below)
  - 6D orientation representation for pelvis, hip_r, hip_l, lumbar: each 3-DoF Euler converted to 6 numbers (first 2 rows of rotation matrix)
  - Angular velocities for pelvis, hip_r, hip_l, lumbar (x,y,z), low-pass filtered (~15 Hz)
  - Optional additional kinematics first derivatives (enabled in this repo)
- Normalization in training: MinMax to [-10,10] learned from training set; not applied when exporting raw windows.
- Alignment & pruning (from code/paper):
  - Align moving direction: rotate yaw so median pelvis yaw = 0; drop trials with >45° variation.
  - Treadmill speed handling when computing pelvis velocities.
  - CoP normalization: (CoP_world - calcn_world) * Fy / height_m; zero when GRF below threshold; sanity-check CoP distance.

References: `args.py`, `consts.py`, `data/addb_dataset.py`, `model/utils.py`, paper (Methods: Datasets section).

## LD HDF5 (source)
- Subjects: S004–S011 focus.
- For each trial (example: `/S004/accel_sine/trial_01`):
  - MoCap:
    - `/MoCap/ik_data/*` (OpenSim IK): pelvis_* (tilt/list/rotation/tx/ty/tz), hip/knee/ankle/subtalar/mtp, lumbar_*.
    - `/MoCap/kin_q`, `/MoCap/kin_qdot` (redundant to IK, both available). Units: radians and meters.
    - `/MoCap/body_pos_global/*` and `/MoCap/body_vel_global/*`: segment/world positions and velocities (e.g., `calcn_l_X/Y/Z`, `pelvis_X/Y/Z`).
    - `/MoCap/grf_measured/left|right/force/Fx|Fy|Fz` (N), `cop/x|y|z` (m), `moment/*` (Nm). Includes `/MoCap/grf_measured/time`.
    - `/MoCap/markers/*` for marker positions.
  - Meta: `/MoCap/gaitStage`, `trial_id`, `trial_label`.
  - Treadmill: `/Treadmill_data/*` contains belt speed and control signals (used for treadmill correction).

Source: `h5viewer/combined_data_S004plus_summary.txt`.

## Field-by-field mapping
Below, “GD col → LD path / transform”. BW = body weight in Newtons (mass*9.81). Height_m = subject height in meters.

Kinematics (positions and Euler angles)
- pelvis_tx/ty/tz → `/MoCap/ik_data/pelvis_tx|ty|tz` (meters). Later used to compute velocities; also shifted so the first frame is at (0,0,0) as in GD.
- pelvis_tilt/list/rotation → `/MoCap/ik_data/pelvis_tilt|list|rotation` (rad; ZXY order assumed in GD conversion).
- hip_*_r/l, knee_angle_r/l, ankle_angle_r/l, subtalar_angle_r/l, mtp_angle_r/l → `/MoCap/ik_data/*` (rad).
- lumbar_extension/bending/rotation → `/MoCap/ik_data/*` (rad).

Kinetics (forces and CoP)
- calcn_r_force_vx/vy/vz → `/MoCap/grf_measured/right/force/Fx|Fy|Fz` divided by BW (i.e., force [N] / (mass*9.81)).
- calcn_l_force_vx/vy/vz → `/MoCap/grf_measured/left/force/Fx|Fy|Fz` divided by BW.
- calcn_r_force_normed_cop_{x,y,z} → Normalize CoP using:
  - CoP_world = `/MoCap/grf_measured/right/cop/{x,y,z}` (m)
  - calcn_world = from `/MoCap/body_pos_global/calcn_r_{X,Y,Z}` (m)
  - Fy_BW = (right Fy [N])/(mass*9.81)
  - normed = (CoP_world - calcn_world) * Fy_BW / Height_m
  - When Fy < threshold (≈ 20 N), set normed CoP = 0 (stance off); clip extreme distances if needed.
- calcn_l_force_normed_cop_{x,y,z} → same using left signals and `calcn_l`.

Orientation and angular velocities
- Convert Euler (ZXY) for pelvis/hip_r/hip_l/lumbar to 6D orientation using GD’s `euler_to_6v`.
- Angular velocities for pelvis/hip_r/hip_l/lumbar:
  - Prefer `/MoCap/kin_qdot/*` if synchronized and clean; otherwise finite-difference Euler (then low-pass ~15 Hz).
  - Units: rad/s.

Velocities of kinematics columns (optional feature block in this repo)
- For non-force, non-6D, non-pelvis_* columns, add first derivatives using spline differentiation (see `convert_addb_state_to_model_input`).

Auxiliary needed from LD
- Subject height_m, mass_kg: if available in H5 metadata; otherwise a config or per-subject lookup. Used for BW normalization and CoP normalization.
- Treadmill belt speed: `/Treadmill_data/*` to correct pelvis velocities as in paper (Earth vs belt frame). If not available, infer from foot motion during stance (as in GD training pipeline).

## Processing steps and rules
1) Load one trial and gather arrays and time vectors from LD.
2) Resample all channels to 100 Hz (linear), after optional low-pass filtering (cutoff ~15 Hz) for angles, positions; maintain time alignment.
3) Compose a states matrix in “AddBiomechanics-like” order: [OpenSim kinematics (positions and Euler), forces (BW), CoP (world)] then:
   - Normalize CoP to calcn frame and scale by Fy/height as GD does.
   - Convert to GD model input ordering and representation with `convert_addb_state_to_model_input()`.
4) Align moving direction: `align_moving_direction()` rotates yaw so median pelvis yaw = 0; also rotates GRF and normed CoP accordingly. Drop trials with >45° direction change.
5) Compute foot marker-based velocities if needed (for treadmill correction) and adjust pelvis_tx/ty/tz velocities.
6) Prune bad data per GD heuristics:
   - Unreasonable lumbar rotation (>45° mean) → drop.
   - Too short (<1.5 s + small buffer) → drop.
   - CoP far from calcn (>0.3 m during stance for >0.2% samples) → drop.
   - Jittery frames with extreme angular velocities (>2000 deg/s) → exclude frames or split trials.
7) Windowing: slice into 150-frame windows with chosen stride; compute available starts where GRF not missing for the whole window.

Notes:
- Frame conventions: Forces and CoP are in Earth coordinates in LD; we rotate/align consistently with pelvis yaw alignment.
- Zero CoP during swing: set to 0 vector when vertical force below a small threshold.
- Overground vs treadmill: pelvis velocities are Earth-frame for overground; for treadmill, subtract belt speed in AP direction after alignment (see paper Methods).

## Open questions to confirm
- Mass and height per subject: where to read from LD (H5 metadata) vs external roster; we need both for BW and CoP normalization.
- Exact Euler sequence in LD IK data: assumed ZXY (OpenSim default for many joints). If different, update conversion.
- Treadmill belt speed availability and sign convention in `/Treadmill_data`.
- Any known lab-specific coordinate rotations needed to match OpenSim world frame used in LD processing.

## Next steps
- Implement `lab_preprocessing/G_LD_to_GD.py` using the mapping above, reusing:
  - Filtering, resampling, alignment, CoP normalization utilities in `model/utils.py`.
- Run a first pass on S004 and inspect basic sanity plots:
  - GRF magnitudes in %BW; zero during swing; symmetric profiles.
  - Normed CoP within reasonable range and bounded by foot size during stance.
  - Yaw alignment verified by forward AP motion.
- Then perform windowing to `.npz` for adaptor training inputs.
