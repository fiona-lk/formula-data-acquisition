# Fiona Keaney
# Underclassmen member
# CSU Fullerton Formula SAE Team

import pandas as pd
import numpy as np

# Load the data from .dat files, skip first 3 rows and specify tab as the delimiter
data_8psi = pd.read_csv('B2356run6.dat', delimiter='\t', skiprows=3, header=None)
data_10psi = pd.read_csv('B2356run4.dat', delimiter='\t', skiprows=3, header=None)

# Display the first 10 rows from run6 (8psi) and run4 (10psi)
print("First 10 rows from run6 (8psi):")
print(data_8psi.head(10))

print("\nFirst 10 rows from run4 (10psi):")
print(data_10psi.head(10))

# Convert the dataframes to numpy arrays
allvar_8psi = data_8psi.to_numpy()
allvar_10psi = data_10psi.to_numpy()

# Check the sizes of both datasets
size_8psi = allvar_8psi.shape
size_10psi = allvar_10psi.shape

# Trim the 8psi data to match the 10psi row count if necessary
if size_8psi[0] > size_10psi[0]:
    allvar_8psi = allvar_8psi[:size_10psi[0], :]

# Define the pressure levels (8psi and 10psi)
pressures = np.array([8, 10])

# Pressure for interpolation
pressure_interp = 9

# Initialize the interpolated 9psi matrix (same size as the trimmed 8psi data)
allvar_9psi = np.zeros_like(allvar_10psi)

# Perform interpolation row by row (across pressures)
for row in range(allvar_10psi.shape[0]):
    for col in range(allvar_10psi.shape[1]):
        allvar_9psi[row, col] = np.interp(pressure_interp, pressures, [allvar_8psi[row, col], allvar_10psi[row, col]])

# Display the interpolated data for 9psi (first 10 rows)
print("\nInterpolated FX-MZ data for 9psi:")
print(allvar_9psi[:10, :])

# Export input into csv
output_filename = 'allvar_8psi.csv'
np.savetxt(output_filename, allvar_8psi, delimiter=',')

output_filename = 'allvar_10psi.csv'
np.savetxt(output_filename, allvar_10psi, delimiter=',')

# Export the interpolated data to CSV for viewing in Excel
output_filename = 'allvar_9psi.csv'
np.savetxt(output_filename, allvar_9psi, delimiter=',')
print(f"\nInterpolated FX-MZ data for 9psi has been exported to: {output_filename}")