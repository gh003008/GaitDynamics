"""Check if raw data has normal variance and distribution"""
import numpy as np
import glob

files = glob.glob('data/LD_gdp/S*/level_*/*.npz')
print(f'Total files: {len(files)}')

# Check first file
data = np.load(files[0])
states = data['model_states']
print(f'\nFirst file: {files[0]}')
print(f'Shape: {states.shape}')
print(f'Value range: [{states.min():.3f}, {states.max():.3f}]')
print(f'Mean: {states.mean():.3f}, Std: {states.std():.3f}')

# Check channel-wise statistics
print(f'\nChannel-wise statistics (first 10 channels):')
for i in range(min(10, states.shape[1])):
    print(f'  Ch{i:02d}: mean={states[:,i].mean():7.3f}, std={states[:,i].std():7.3f}, '
          f'range=[{states[:,i].min():7.3f}, {states[:,i].max():7.3f}]')

# Check all files
print(f'\nChecking all {len(files)} files...')
all_stats = []
for f in files:
    d = np.load(f)['model_states']
    all_stats.append([d.mean(), d.std(), d.min(), d.max()])

stats = np.array(all_stats)
print(f'Mean of means: {stats[:,0].mean():.3f}')
print(f'Mean of stds: {stats[:,1].mean():.3f}')
print(f'Global min: {stats[:,2].min():.3f}')
print(f'Global max: {stats[:,3].max():.3f}')

# Check if data variance collapsed
if stats[:,1].mean() < 0.1:
    print('\n⚠️ WARNING: Data has very low variance! Possible normalization issue.')
else:
    print(f'\n✓ Data variance looks normal (std ~{stats[:,1].mean():.2f})')
