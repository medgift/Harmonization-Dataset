import numpy as np
import pandas as pd
import pingouin as pg

# Parameters
n_subjects = 10
n_raters = 3
base_measurements = [50, 60, 70]  # Multiple true values
noise_levels = list(range(15))  # Different levels of noise for each rater
noise_amplitude = 100000

# Simulate data
data = []
for subject in range(n_subjects):
    for rater in range(n_raters):
        true_value = base_measurements[rater] + noise_amplitude*np.random.normal(0, 5)  # True value with some inherent variability
        measurement = true_value + noise_amplitude*np.random.normal(0, noise_levels[rater])  # Add different noise for each rater
        data.append([f'S{subject+1}', f'R{rater+1}', measurement])

# Create DataFrame
df = pd.DataFrame(data, columns=['subject', 'rater', 'measurement'])
print(df)

# Calculate ICC
icc_results = pg.intraclass_corr(data=df, targets='subject', raters='rater', ratings='measurement')
print(icc_results)

print('')
print('')
# Create DataFrame
df = pd.DataFrame(data, columns=['rater', 'subject', 'measurement'])

# Calculate ICC
icc_results = pg.intraclass_corr(data=df, targets='subject', raters='rater', ratings='measurement')
print(icc_results)